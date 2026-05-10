import os
import json
import numpy as np
import pandas as pd


# ----------------------------
# Metrics for 1-positive setting
# ----------------------------
def metrics_for_one_positive(rank_1based: int, K: int):
    """
    rank_1based: rank position of the positive item in the ranked list (1 = best).
                If not in top-K, return zeros.
    """
    if rank_1based is None or rank_1based > K:
        return {
            f"Precision@{K}": 0.0,
            f"Recall@{K}": 0.0,
            f"NDCG@{K}": 0.0,
            f"HitRate@{K}": 0.0,
        }

    # With one relevant item:
    # Precision@K = 1/K if hit else 0
    # Recall@K = 1 if hit else 0
    # HitRate@K = 1 if hit else 0
    # NDCG@K = 1/log2(rank+1) if hit else 0
    return {
        f"Precision@{K}": 1.0 / K,
        f"Recall@{K}": 1.0,
        f"NDCG@{K}": 1.0 / np.log2(rank_1based + 1.0),
        f"HitRate@{K}": 1.0,
    }


# ----------------------------
# Load model artifacts
# ----------------------------
def load_itemknn_model(model_dir: str):
    user_ids = np.load(os.path.join(model_dir, "user_ids.npy"))
    item_ids = np.load(os.path.join(model_dir, "item_ids.npy"))
    knn_indices = np.load(os.path.join(model_dir, "knn_indices.npy"))  # shape (n_items, topKmax)
    knn_sims = np.load(os.path.join(model_dir, "knn_sims.npy"))        # shape (n_items, topKmax)

    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    user2idx = {int(u): i for i, u in enumerate(user_ids)}
    movieId_to_itemIndex = {int(m): i for i, m in enumerate(item_ids)}

    return {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "user2idx": user2idx,
        "movieId_to_itemIndex": movieId_to_itemIndex,
        "knn_indices": knn_indices,
        "knn_sims": knn_sims,
        "config": config,
    }


# ----------------------------
# Build train history (positives only)
# ----------------------------
def build_train_history(train_path: str, movieId_to_itemIndex: dict, rating_threshold: float = 4.0):
    df = pd.read_csv(train_path, usecols=["userId", "movieId", "rating", "timestamp"])
    df = df[df["rating"] >= rating_threshold].copy()

    # Map movieId -> itemIndex (drop items not present in model)
    df["itemIndex"] = df["movieId"].map(movieId_to_itemIndex)
    df = df.dropna(subset=["itemIndex"]).copy()
    df["itemIndex"] = df["itemIndex"].astype(np.int32)

    # Keep latest per (user,item) if duplicates exist
    df = df.sort_values("timestamp").drop_duplicates(["userId", "itemIndex"], keep="last")

    # Group into history lists
    history = df.groupby("userId")["itemIndex"].apply(lambda x: x.to_numpy()).to_dict()
    return history


# ----------------------------
# Prepare validation positives (1 per user)
# ----------------------------
def load_val_positives(val_path: str, movieId_to_itemIndex: dict, rating_threshold: float = 4.0):
    val = pd.read_csv(val_path, usecols=["userId", "movieId", "rating", "timestamp"])
    val = val[val["rating"] >= rating_threshold].copy()

    val["posItemIndex"] = val["movieId"].map(movieId_to_itemIndex)
    val = val.dropna(subset=["posItemIndex"]).copy()
    val["posItemIndex"] = val["posItemIndex"].astype(np.int32)

    # val.csv already has 1 row per user from your split, so no need to dedup,
    # but just in case:
    val = val.sort_values("timestamp").drop_duplicates(["userId"], keep="last")

    return val[["userId", "posItemIndex"]]


# ----------------------------
# Scoring function using neighbor lists
# ----------------------------
def score_candidates_for_user(
    user_history: np.ndarray,
    candidates: np.ndarray,
    knn_indices: np.ndarray,
    knn_sims: np.ndarray,
    topk_neighbors: int,
    n_items: int,
):
    """
    Score only the candidate items using neighbor expansion:
      score(candidate) = sum over history items h of sim(h -> candidate) if candidate is in h's neighbors.

    Efficient approach:
      - Build an array pos_of_item[item] = position in candidates (or -1)
      - For each history item h:
          neigh = knn_indices[h, :topk_neighbors]
          sims  = knn_sims[h, :topk_neighbors]
          pos = pos_of_item[neigh]
          add sims to scores[pos != -1]
    """
    scores = np.zeros(len(candidates), dtype=np.float32)

    pos_of_item = np.full(n_items, -1, dtype=np.int32)
    pos_of_item[candidates] = np.arange(len(candidates), dtype=np.int32)

    for h in user_history:
        neigh = knn_indices[h, :topk_neighbors]
        sims = knn_sims[h, :topk_neighbors]
        pos = pos_of_item[neigh]
        mask = pos != -1
        if np.any(mask):
            # add sims for matched neighbors into candidate score positions
            np.add.at(scores, pos[mask], sims[mask])

    return scores


# ----------------------------
# Evaluate on validation
# ----------------------------
def evaluate_itemknn_on_val(
    val_df: pd.DataFrame,
    train_history: dict,
    knn_indices: np.ndarray,
    knn_sims: np.ndarray,
    n_items: int,
    topk_neighbors: int,
    K_list=(10, 20),
    num_negatives: int = 499,
    seed: int = 42,
    max_users: int = None,   # set e.g. 20000 if you want faster trial
):
    rng = np.random.default_rng(seed)

    # For metrics accumulation
    metric_sums = {f"{m}@{K}": 0.0 for K in K_list for m in ["Precision", "Recall", "NDCG", "HitRate"]}
    used_users = 0

    # Optional subsample (for speed)
    if max_users is not None and len(val_df) > max_users:
        val_df = val_df.sample(n=max_users, random_state=seed).reset_index(drop=True)

    for row in val_df.itertuples(index=False):
        user = int(row.userId)
        pos_item = int(row.posItemIndex)

        hist = train_history.get(user, None)
        if hist is None or len(hist) == 0:
            continue

        # Candidate negatives: items not in history and not the positive
        hist_set = set(hist.tolist())
        hist_set.add(pos_item)

        # Sample negatives
        # (n_items is only ~20k, so this is okay)
        negatives = []
        while len(negatives) < num_negatives:
            cand = int(rng.integers(0, n_items))
            if cand not in hist_set:
                negatives.append(cand)

        candidates = np.array([pos_item] + negatives, dtype=np.int32)

        # Score candidates
        scores = score_candidates_for_user(
            user_history=hist,
            candidates=candidates,
            knn_indices=knn_indices,
            knn_sims=knn_sims,
            topk_neighbors=topk_neighbors,
            n_items=n_items,
        )

        # Rank candidates by score (descending)
        order = np.argsort(-scores)
        ranked_candidates = candidates[order]

        # Find positive rank (1-based)
        # positive is candidates[0] => pos_item
        pos_rank = None
        where = np.where(ranked_candidates == pos_item)[0]
        if len(where) > 0:
            pos_rank = int(where[0]) + 1  # 1-based

        # Update metrics for each K
        for K in K_list:
            mets = metrics_for_one_positive(pos_rank, K)
            metric_sums[f"Precision@{K}"] += mets[f"Precision@{K}"]
            metric_sums[f"Recall@{K}"] += mets[f"Recall@{K}"]
            metric_sums[f"NDCG@{K}"] += mets[f"NDCG@{K}"]
            metric_sums[f"HitRate@{K}"] += mets[f"HitRate@{K}"]

        used_users += 1

    # Average
    if used_users == 0:
        raise RuntimeError("No users were evaluated. (Likely due to filtering / missing histories.)")

    results = {k: v / used_users for k, v in metric_sums.items()}
    results["users_evaluated"] = used_users
    results["topk_neighbors"] = topk_neighbors
    results["num_negatives"] = num_negatives
    return results


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    VAL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "val.csv")

    MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model")
    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model", "tuning_reports")
    os.makedirs(OUT_DIR, exist_ok=True)

    rating_threshold = 4.0
    num_negatives = 499
    K_list = (10, 20)

    print("Loading model from:", MODEL_DIR)
    model = load_itemknn_model(MODEL_DIR)

    knn_indices = model["knn_indices"]
    knn_sims = model["knn_sims"]
    n_items = knn_indices.shape[0]
    topKmax = knn_indices.shape[1]

    print("Model topKmax:", topKmax, "| n_items:", n_items)

    print("\nBuilding train history from:", TRAIN_PATH)
    train_history = build_train_history(TRAIN_PATH, model["movieId_to_itemIndex"], rating_threshold=rating_threshold)

    print("Loading validation positives from:", VAL_PATH)
    val_pos = load_val_positives(VAL_PATH, model["movieId_to_itemIndex"], rating_threshold=rating_threshold)

    # Keep only users that have history
    val_pos = val_pos[val_pos["userId"].isin(train_history.keys())].reset_index(drop=True)
    print("Users with valid val+history:", len(val_pos))

    # Neighbor cutoffs to test (must be <= topKmax)
    neighbor_grid = [50, 100]
    neighbor_grid = [k for k in neighbor_grid if k <= topKmax]

    # Optional: speed knob
    # For full run, set max_users = None
    max_users = None  # e.g., 20000 if you want a quick dry run first

    all_results = []
    for nk in neighbor_grid:
        print(f"\nEvaluating topk_neighbors = {nk} on validation...")
        res = evaluate_itemknn_on_val(
            val_df=val_pos,
            train_history=train_history,
            knn_indices=knn_indices,
            knn_sims=knn_sims,
            n_items=n_items,
            topk_neighbors=nk,
            K_list=K_list,
            num_negatives=num_negatives,
            seed=42,
            max_users=max_users,
        )
        all_results.append(res)
        print("Done:", res)

    results_df = pd.DataFrame(all_results)
    results_path = os.path.join(OUT_DIR, "val_tuning_results.csv")
    results_df.to_csv(results_path, index=False)

    print("\n✅ Saved validation tuning results to:")
    print(results_path)
    print("\nSummary:")
    print(results_df.sort_values(by="NDCG@20", ascending=False))


if __name__ == "__main__":
    main()
