import os
import numpy as np
import pandas as pd


# ============================================
# SETTINGS
# ============================================
NUM_NEGATIVES = 499
KS = (10, 20)
SEED = 42
# ============================================


def ndcg_at_k(rank_1based: int, k: int) -> float:
    """NDCG@k with 1 relevant item."""
    if rank_1based <= 0 or rank_1based > k:
        return 0.0
    return 1.0 / np.log2(rank_1based + 1)


def metrics_from_pos_rank(pos_pos_0based: int, k: int):
    """Given 0-based position of positive item in ranked list."""
    rank_1based = pos_pos_0based + 1
    hit = 1.0 if rank_1based <= k else 0.0
    precision = hit / float(k)
    recall = hit  # only 1 positive
    ndcg = ndcg_at_k(rank_1based, k)
    return precision, recall, ndcg, hit


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    POP_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Cpopularity_model")
    RANKING_NPY = os.path.join(POP_DIR, "popularity_ranking.npy")

    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    TEST_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "test.csv")

    OUT_DIR = os.path.join(POP_DIR, "final_report")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_CSV = os.path.join(OUT_DIR, "popularity_test_results.csv")

    print("============================================")
    print("📊 Popularity Baseline — TEST Evaluation")
    print("============================================")
    print("Popularity ranking:", RANKING_NPY)
    print("Train file        :", TRAIN_PATH)
    print("Test file         :", TEST_PATH)
    print("Negatives/user    :", NUM_NEGATIVES)
    print("K values          :", KS)
    print("Output CSV        :", OUT_CSV)
    print("============================================\n")

    # --------------------------------------------
    # Load popularity ranking
    # --------------------------------------------
    pop_ranking = np.load(RANKING_NPY)
    pop_rank_index = {int(mid): idx for idx, mid in enumerate(pop_ranking)}
    all_movies = pop_ranking.astype(int)

    # --------------------------------------------
    # Train history per user (movies seen in train)
    # --------------------------------------------
    train = pd.read_csv(TRAIN_PATH, usecols=["userId", "movieId"])
    train_hist = train.groupby("userId")["movieId"].apply(set).to_dict()

    # --------------------------------------------
    # Load test positives (1 per user in your split)
    # --------------------------------------------
    test = pd.read_csv(TEST_PATH, usecols=["userId", "movieId"])
    test_pos = test.groupby("userId")["movieId"].last().to_dict()

    users = sorted(set(test_pos.keys()) & set(train_hist.keys()))
    print("Users with test + train history:", len(users))

    rng = np.random.default_rng(SEED)

    totals = {k: {"Precision": 0.0, "Recall": 0.0, "NDCG": 0.0, "HitRate": 0.0} for k in KS}
    users_evaluated = 0

    for u in users:
        pos_item = int(test_pos[u])
        seen = train_hist[u]

        # Build candidate set: 1 positive + negatives
        negs = []
        while len(negs) < NUM_NEGATIVES:
            cand = int(rng.choice(all_movies))
            if cand == pos_item:
                continue
            if cand in seen:
                continue
            negs.append(cand)

        candidates = [pos_item] + negs

        # Popularity score: higher if more popular (lower rank index)
        scores = np.empty(len(candidates), dtype=np.int64)
        for i, mid in enumerate(candidates):
            rank_idx = pop_rank_index.get(int(mid), 10**12)
            scores[i] = -rank_idx

        order = np.argsort(-scores)
        ranked = [candidates[i] for i in order]

        pos_pos = ranked.index(pos_item)

        for k in KS:
            p, r, nd, h = metrics_from_pos_rank(pos_pos, k)
            totals[k]["Precision"] += p
            totals[k]["Recall"] += r
            totals[k]["NDCG"] += nd
            totals[k]["HitRate"] += h

        users_evaluated += 1
        if users_evaluated % 20000 == 0:
            print(f"Evaluated {users_evaluated} users...")

    # Average
    results = {
        "model": "Popularity",
        "users_evaluated": users_evaluated,
        "num_negatives": NUM_NEGATIVES
    }
    for k in KS:
        results[f"Precision@{k}"] = totals[k]["Precision"] / users_evaluated
        results[f"Recall@{k}"] = totals[k]["Recall"] / users_evaluated
        results[f"NDCG@{k}"] = totals[k]["NDCG"] / users_evaluated
        results[f"HitRate@{k}"] = totals[k]["HitRate"] / users_evaluated

    pd.DataFrame([results]).to_csv(OUT_CSV, index=False)

    print("\n✅ Popularity TEST Results:")
    print(results)
    print("\n✅ Saved to:", OUT_CSV)


if __name__ == "__main__":
    main()
