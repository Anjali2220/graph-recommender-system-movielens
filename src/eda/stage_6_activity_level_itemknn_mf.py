import os
import numpy as np
import pandas as pd
import torch

from src.eda.mf_model import MFModel


# ============================================
# SETTINGS
# ============================================
RATING_THRESHOLD = 4.0
NUM_NEGATIVES = 499
KS = (10, 20)
SEED = 42

ACTIVITY_BINS = [
    (1, 5, "1-5"),
    (6, 20, "6-20"),
    (21, 50, "21-50"),
    (51, 100, "51-100"),
    (101, 10**9, "101+"),
]
# ============================================


def ndcg(rank_1based, k):
    if rank_1based <= k:
        return 1.0 / np.log2(rank_1based + 1)
    return 0.0


def compute_metrics(rank_1based, k):
    hit = 1.0 if rank_1based <= k else 0.0
    precision = hit / k
    recall = hit
    nd = ndcg(rank_1based, k)
    return precision, recall, nd, hit


def evaluate_candidates(scores):
    order = np.argsort(-scores)
    pos_position = np.where(order == 0)[0][0]
    return pos_position + 1


def main():

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # -------- Paths --------
    MF_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")
    MF_MODEL_PATH = os.path.join(MF_DIR, "tuning", "mf_dim64_lr0.001_ep5.pt")

    PAIRS_PATH = os.path.join(MF_DIR, "train_pairs.npy")
    USER_IDS_PATH = os.path.join(MF_DIR, "user_ids.npy")
    ITEM_IDS_PATH = os.path.join(MF_DIR, "item_ids.npy")

    TEST_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

    ITEMKNN_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model")
    KNN_INDICES_PATH = os.path.join(ITEMKNN_DIR, "knn_indices.npy")
    KNN_SIMS_PATH = os.path.join(ITEMKNN_DIR, "knn_sims.npy")

    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "analysis_activity")
    os.makedirs(OUT_DIR, exist_ok=True)
    OUT_CSV = os.path.join(OUT_DIR, "activity_level_itemknn_mf_test.csv")

    print("Output file:", OUT_CSV)

    # -------- Load Data --------
    pairs = np.load(PAIRS_PATH)
    user_ids = np.load(USER_IDS_PATH)
    item_ids = np.load(ITEM_IDS_PATH)

    n_users = len(user_ids)
    n_items = len(item_ids)

    # User activity
    activity_counts = np.bincount(pairs[:, 0].astype(np.int32), minlength=n_users)

    # Build user positive sets
    user_pos = [set() for _ in range(n_users)]
    for u, i in pairs:
        user_pos[int(u)].add(int(i))

    # -------- Load Test Positives --------
    test = pd.read_csv(TEST_CSV)
    test = test[test["rating"] >= RATING_THRESHOLD]

    user2idx = {int(u): i for i, u in enumerate(user_ids)}
    item2idx = {int(m): i for i, m in enumerate(item_ids)}

    test["u_idx"] = test["userId"].map(user2idx)
    test["i_idx"] = test["movieId"].map(item2idx)
    test = test.dropna(subset=["u_idx", "i_idx"])
    test = test.drop_duplicates(subset=["u_idx"], keep="last")

    test_pos = [None] * n_users
    for row in test.itertuples():
        test_pos[int(row.u_idx)] = int(row.i_idx)

    # -------- Load Models --------
    knn_indices = np.load(KNN_INDICES_PATH)
    knn_sims = np.load(KNN_SIMS_PATH)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mf_model, _ = MFModel.load(
        MF_MODEL_PATH,
        n_users=n_users,
        n_items=n_items,
        emb_dim=64,
        map_location=device
    )
    mf_model = mf_model.to(device).eval()

    with torch.no_grad():
        user_emb = mf_model.user_emb.weight.detach().cpu().numpy()
        item_emb = mf_model.item_emb.weight.detach().cpu().numpy()

    rng = np.random.default_rng(SEED)

    rows = []

    # -------- Evaluate Per Activity Bin --------
    for lo, hi, label in ACTIVITY_BINS:

        users_bin = np.where((activity_counts >= lo) & (activity_counts <= hi))[0]
        users_bin = [u for u in users_bin if test_pos[u] is not None]

        if len(users_bin) == 0:
            continue

        print(f"\nEvaluating bin {label}, users:", len(users_bin))

        metrics = {
            "ItemKNN": {k: {"Recall": 0, "NDCG": 0} for k in KS},
            "MF": {k: {"Recall": 0, "NDCG": 0} for k in KS},
        }

        for u in users_bin:

            pos = test_pos[u]
            seen = user_pos[u]

            # Sample negatives
            negs = []
            while len(negs) < NUM_NEGATIVES:
                cand = int(rng.integers(0, n_items))
                if cand == pos or cand in seen:
                    continue
                negs.append(cand)

            candidates = np.array([pos] + negs)

            # -------- ItemKNN --------
            scores_knn = np.zeros(n_items, dtype=np.float32)
            for it in seen:
                scores_knn[knn_indices[it]] += knn_sims[it]
            cand_scores_knn = scores_knn[candidates]

            rank_knn = evaluate_candidates(cand_scores_knn)

            # -------- MF --------
            cand_scores_mf = item_emb[candidates] @ user_emb[u]
            rank_mf = evaluate_candidates(cand_scores_mf)

            for k in KS:
                _, rec_knn, nd_knn, _ = compute_metrics(rank_knn, k)
                _, rec_mf, nd_mf, _ = compute_metrics(rank_mf, k)

                metrics["ItemKNN"][k]["Recall"] += rec_knn
                metrics["ItemKNN"][k]["NDCG"] += nd_knn

                metrics["MF"][k]["Recall"] += rec_mf
                metrics["MF"][k]["NDCG"] += nd_mf

        # Average
        row = {
            "activity_bin": label,
            "users_in_bin": len(users_bin),
            "avg_train_interactions": round(activity_counts[users_bin].mean(), 2)
        }

        for model in ["ItemKNN", "MF"]:
            for k in KS:
                row[f"{model}_Recall@{k}"] = round(
                    metrics[model][k]["Recall"] / len(users_bin), 4
                )
                row[f"{model}_NDCG@{k}"] = round(
                    metrics[model][k]["NDCG"] / len(users_bin), 4
                )

        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)

    print("\nSaved results to:", OUT_CSV)
    print("\nPreview:\n")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()