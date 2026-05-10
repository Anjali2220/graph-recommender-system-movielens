import os
import json
import numpy as np
import pandas as pd
import torch

# ✅ matches your current structure (mf_model.py is in src/eda/)
from src.eda.mf_model import MFModel


# ----------------------------
# Metrics for 1 positive per user
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


def build_user_pos_sets_from_pairs(pairs: np.ndarray, n_users: int):
    user_pos_sets = [set() for _ in range(n_users)]
    for u, i in pairs:
        user_pos_sets[int(u)].add(int(i))
    return user_pos_sets


def evaluate_mf_leave_one_out(
    user_emb: np.ndarray,          # (n_users, dim)
    item_emb: np.ndarray,          # (n_items, dim)
    test_df: pd.DataFrame,         # columns: user_idx, pos_item_idx
    user_pos_sets: list,
    n_items: int,
    K_list=(10, 20),
    num_negatives=499,
    seed=42,
):
    rng = np.random.default_rng(seed)

    sums = {f"{m}@{K}": 0.0 for K in K_list for m in ["Precision", "Recall", "NDCG", "HitRate"]}
    used = 0

    for row in test_df.itertuples(index=False):
        u = int(row.user_idx)
        pos = int(row.pos_item_idx)

        pos_set = user_pos_sets[u]

        # sample negatives not in train positives, and not the test positive
        negatives = []
        while len(negatives) < num_negatives:
            cand = int(rng.integers(0, n_items))
            if cand != pos and cand not in pos_set:
                negatives.append(cand)

        candidates = np.array([pos] + negatives, dtype=np.int32)

        # score = dot(user_emb[u], item_emb[candidates])
        uvec = user_emb[u]               # (dim,)
        scores = item_emb[candidates] @ uvec  # (500,)

        order = np.argsort(-scores)
        ranked = candidates[order]

        where = np.where(ranked == pos)[0]
        rank_1based = int(where[0]) + 1 if len(where) > 0 else None

        for K in K_list:
            mets = metrics_for_one_positive(rank_1based, K)
            for k, v in mets.items():
                sums[k] += v

        used += 1

    results = {k: v / used for k, v in sums.items()}
    results["users_evaluated"] = used
    results["num_negatives"] = num_negatives
    return results


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    MF_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")
    TUNE_DIR = os.path.join(MF_DIR, "tuning")

    # ✅ Best tuned MF checkpoint from MF-4
    BEST_TAG = "dim64_lr0.001_ep5"
    MODEL_PATH = os.path.join(TUNE_DIR, f"mf_{BEST_TAG}.pt")

    # MF-1 artifacts
    PAIRS_PATH = os.path.join(MF_DIR, "train_pairs.npy")
    USER_IDS_PATH = os.path.join(MF_DIR, "user_ids.npy")
    ITEM_IDS_PATH = os.path.join(MF_DIR, "item_ids.npy")

    TEST_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

    OUT_DIR = os.path.join(MF_DIR, "final_report")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_CSV = os.path.join(OUT_DIR, "mf_test_results.csv")

    # Evaluation settings
    rating_threshold = 4.0
    K_list = (10, 20)
    num_negatives = 499
    seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Project root :", PROJECT_ROOT)
    print("MF dir       :", MF_DIR)
    print("Device       :", device)
    print("Best model   :", MODEL_PATH)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Best tuned MF checkpoint not found: {MODEL_PATH}")

    # Load MF-1 artifacts
    pairs = np.load(PAIRS_PATH)
    user_ids = np.load(USER_IDS_PATH)
    item_ids = np.load(ITEM_IDS_PATH)

    n_users = len(user_ids)
    n_items = len(item_ids)

    user2idx = {int(u): i for i, u in enumerate(user_ids)}
    item2idx = {int(m): i for i, m in enumerate(item_ids)}

    # Build train positives for negative sampling filter
    print("\nBuilding train positive sets...")
    user_pos_sets = build_user_pos_sets_from_pairs(pairs, n_users=n_users)

    # Load MF model (to get embeddings)
    print("Loading MF model checkpoint...")
    model, extra = MFModel.load(
        MODEL_PATH,
        n_users=n_users,
        n_items=n_items,
        emb_dim=64,            # matches best config
        map_location=device
    )
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        user_emb = model.user_emb.weight.detach().cpu().numpy()
        item_emb = model.item_emb.weight.detach().cpu().numpy()

    # Load test positives and map to indices
    print("\nLoading TEST positives...")
    test = pd.read_csv(TEST_CSV, usecols=["userId", "movieId", "rating"])
    test = test[test["rating"] >= rating_threshold].copy()

    test["user_idx"] = test["userId"].map(user2idx)
    test["pos_item_idx"] = test["movieId"].map(item2idx)

    test = test.dropna(subset=["user_idx", "pos_item_idx"]).copy()
    test["user_idx"] = test["user_idx"].astype(np.int32)
    test["pos_item_idx"] = test["pos_item_idx"].astype(np.int32)

    # One test item per user
    test = test.drop_duplicates(subset=["user_idx"], keep="last").reset_index(drop=True)
    print("Users evaluated:", len(test))

    # Evaluate
    print("\nEvaluating MF on TEST (1 pos + 499 neg)...")
    res = evaluate_mf_leave_one_out(
        user_emb=user_emb,
        item_emb=item_emb,
        test_df=test[["user_idx", "pos_item_idx"]],
        user_pos_sets=user_pos_sets,
        n_items=n_items,
        K_list=K_list,
        num_negatives=num_negatives,
        seed=seed,
    )

    final = {
        "model": "MF_BPR",
        "best_tag": BEST_TAG,
        "emb_dim": 64,
        "lr": 0.001,
        "epochs": 5,
        **res
    }

    print("\n✅ Final MF TEST results:")
    print(final)

    pd.DataFrame([final]).to_csv(OUT_CSV, index=False)

    print("\n✅ Saved MF test report to:")
    print(OUT_CSV)


if __name__ == "__main__":
    main()
