import os
import json
import numpy as np
import pandas as pd


# ----------------------------
# Metrics (1 positive per user)
# ----------------------------
def metrics_for_one_positive(rank_1based, K):
    if rank_1based is None or rank_1based > K:
        return {
            f"Precision@{K}": 0.0,
            f"Recall@{K}": 0.0,
            f"NDCG@{K}": 0.0,
            f"HitRate@{K}": 0.0,
        }

    return {
        f"Precision@{K}": 1.0 / K,
        f"Recall@{K}": 1.0,
        f"NDCG@{K}": 1.0 / np.log2(rank_1based + 1.0),
        f"HitRate@{K}": 1.0,
    }


# ----------------------------
# Load trained model
# ----------------------------
def load_model(model_dir):
    user_ids = np.load(os.path.join(model_dir, "user_ids.npy"))
    item_ids = np.load(os.path.join(model_dir, "item_ids.npy"))
    knn_indices = np.load(os.path.join(model_dir, "knn_indices.npy"))
    knn_sims = np.load(os.path.join(model_dir, "knn_sims.npy"))

    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)

    movieId_to_itemIndex = {int(m): i for i, m in enumerate(item_ids)}

    return {
        "item_ids": item_ids,
        "movieId_to_itemIndex": movieId_to_itemIndex,
        "knn_indices": knn_indices,
        "knn_sims": knn_sims,
        "config": config,
    }


# ----------------------------
# Build train history
# ----------------------------
def build_train_history(train_path, movieId_to_itemIndex, rating_threshold=4.0):
    df = pd.read_csv(train_path)
    df = df[df["rating"] >= rating_threshold]

    df["itemIndex"] = df["movieId"].map(movieId_to_itemIndex)
    df = df.dropna(subset=["itemIndex"])
    df["itemIndex"] = df["itemIndex"].astype(np.int32)

    df = df.sort_values("timestamp").drop_duplicates(["userId", "itemIndex"], keep="last")

    history = df.groupby("userId")["itemIndex"].apply(lambda x: x.to_numpy()).to_dict()
    return history


# ----------------------------
# Load test positives
# ----------------------------
def load_test_positives(test_path, movieId_to_itemIndex, rating_threshold=4.0):
    df = pd.read_csv(test_path)
    df = df[df["rating"] >= rating_threshold]

    df["posItemIndex"] = df["movieId"].map(movieId_to_itemIndex)
    df = df.dropna(subset=["posItemIndex"])
    df["posItemIndex"] = df["posItemIndex"].astype(np.int32)

    df = df.sort_values("timestamp").drop_duplicates(["userId"], keep="last")

    return df[["userId", "posItemIndex"]]


# ----------------------------
# Score candidates
# ----------------------------
def score_user(hist, candidates, knn_indices, knn_sims, topk_neighbors, n_items):

    scores = np.zeros(len(candidates), dtype=np.float32)

    pos_of_item = np.full(n_items, -1, dtype=np.int32)
    pos_of_item[candidates] = np.arange(len(candidates))

    for h in hist:
        neigh = knn_indices[h, :topk_neighbors]
        sims = knn_sims[h, :topk_neighbors]

        pos = pos_of_item[neigh]
        mask = pos != -1
        if np.any(mask):
            np.add.at(scores, pos[mask], sims[mask])

    return scores


# ----------------------------
# Evaluation
# ----------------------------
def evaluate(
    test_df,
    train_history,
    knn_indices,
    knn_sims,
    n_items,
    topk_neighbors=50,
    K_list=(10, 20),
    num_negatives=499,
    seed=42,
):

    rng = np.random.default_rng(seed)

    metric_sums = {f"{m}@{K}": 0.0 for K in K_list for m in ["Precision", "Recall", "NDCG", "HitRate"]}
    users_evaluated = 0

    for row in test_df.itertuples(index=False):
        user = int(row.userId)
        pos_item = int(row.posItemIndex)

        hist = train_history.get(user)
        if hist is None or len(hist) == 0:
            continue

        hist_set = set(hist.tolist())
        hist_set.add(pos_item)

        negatives = []
        while len(negatives) < num_negatives:
            cand = int(rng.integers(0, n_items))
            if cand not in hist_set:
                negatives.append(cand)

        candidates = np.array([pos_item] + negatives, dtype=np.int32)

        scores = score_user(hist, candidates, knn_indices, knn_sims, topk_neighbors, n_items)

        ranked = candidates[np.argsort(-scores)]

        where = np.where(ranked == pos_item)[0]
        pos_rank = int(where[0]) + 1 if len(where) > 0 else None

        for K in K_list:
            mets = metrics_for_one_positive(pos_rank, K)
            for key in mets:
                metric_sums[key] += mets[key]

        users_evaluated += 1

    results = {k: v / users_evaluated for k, v in metric_sums.items()}
    results["users_evaluated"] = users_evaluated
    results["topk_neighbors"] = topk_neighbors
    results["num_negatives"] = num_negatives

    return results


# ----------------------------
# Main
# ----------------------------
def main():

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model")
    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model", "final_report")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model...")
    model = load_model(MODEL_DIR)

    print("Building train history...")
    train_history = build_train_history(TRAIN_PATH, model["movieId_to_itemIndex"])

    print("Loading test positives...")
    test_pos = load_test_positives(TEST_PATH, model["movieId_to_itemIndex"])
    test_pos = test_pos[test_pos["userId"].isin(train_history.keys())].reset_index(drop=True)

    print("Users evaluated:", len(test_pos))

    print("Evaluating on TEST set...")
    results = evaluate(
        test_pos,
        train_history,
        model["knn_indices"],
        model["knn_sims"],
        n_items=len(model["item_ids"]),
        topk_neighbors=50,
    )

    print("\nFinal Baseline Test Results:")
    print(results)

    results_df = pd.DataFrame([results])
    results_df.to_csv(os.path.join(OUT_DIR, "baseline_test_results.csv"), index=False)

    print("\n✅ Saved final baseline report to:")
    print(os.path.join(OUT_DIR, "baseline_test_results.csv"))


if __name__ == "__main__":
    main()
