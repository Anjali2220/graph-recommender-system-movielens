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


def ndcg_at_k(rank: int, k: int) -> float:
    """If positive item is at 1-based rank 'rank', compute NDCG@k."""
    if rank <= 0 or rank > k:
        return 0.0
    # DCG = 1/log2(rank+1), IDCG = 1 (since only 1 relevant item)
    return 1.0 / np.log2(rank + 1)


def eval_user(pop_rank_pos: int, k: int) -> dict:
    """
    Given the rank position (0-based) of the positive item in the scored candidate list,
    compute metrics for this user at cutoff k.
    """
    rank_1based = pop_rank_pos + 1

    hit = 1.0 if rank_1based <= k else 0.0
    precision = hit / float(k)
    recall = hit  # only 1 positive
    ndcg = ndcg_at_k(rank_1based, k)

    return {"Precision": precision, "Recall": recall, "NDCG": ndcg, "HitRate": hit}


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    POP_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Cpopularity_model")
    RANKING_NPY = os.path.join(POP_DIR, "popularity_ranking.npy")

    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    VAL_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "val.csv")

    OUT_DIR = os.path.join(POP_DIR, "tuning_reports")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_CSV = os.path.join(OUT_DIR, "popularity_val_results.csv")

    print("============================================")
    print("📊 Popularity Baseline — Validation Evaluation")
    print("============================================")
    print("Popularity ranking:", RANKING_NPY)
    print("Train file        :", TRAIN_PATH)
    print("Val file          :", VAL_PATH)
    print("Negatives/user    :", NUM_NEGATIVES)
    print("K values          :", KS)
    print("Output CSV        :", OUT_CSV)
    print("============================================\n")

    # --------------------------------------------
    # Load popularity ranking
    # --------------------------------------------
    pop_ranking = np.load(RANKING_NPY)
    pop_rank_index = {int(mid): idx for idx, mid in enumerate(pop_ranking)}

    # --------------------------------------------
    # Build train history (movies seen in train per user)
    # --------------------------------------------
    train = pd.read_csv(TRAIN_PATH, usecols=["userId", "movieId"])
    train_hist = train.groupby("userId")["movieId"].apply(set).to_dict()

    # --------------------------------------------
    # Load validation positives (1 per user in your split)
    # --------------------------------------------
    val = pd.read_csv(VAL_PATH, usecols=["userId", "movieId"])
    # If multiple rows per user, keep last (should be 1 anyway)
    val_pos = val.groupby("userId")["movieId"].last().to_dict()

    users = sorted(set(val_pos.keys()) & set(train_hist.keys()))
    print("Users with val + train history:", len(users))

    rng = np.random.default_rng(SEED)

    # We will sample negatives from the movie universe = pop_ranking list
    all_movies = pop_ranking.astype(int)

    # Accumulators
    totals = {k: {"Precision": 0.0, "Recall": 0.0, "NDCG": 0.0, "HitRate": 0.0} for k in KS}
    users_evaluated = 0

    for u in users:
        pos_item = int(val_pos[u])
        seen = train_hist[u]

        # Candidates: 1 positive + 499 negatives (unseen & not positive)
        # Negatives drawn from all_movies excluding seen and pos
        # Efficient sampling: loop until we collect enough
        negs = []
        while len(negs) < NUM_NEGATIVES:
            cand = int(rng.choice(all_movies))
            if cand == pos_item:
                continue
            if cand in seen:
                continue
            negs.append(cand)

        candidates = [pos_item] + negs

        # Score candidates by popularity rank (smaller rank index = more popular)
        # If a movie not found in ranking dict, treat as very unpopular (push to bottom)
        scores = []
        for mid in candidates:
            rank_idx = pop_rank_index.get(int(mid), 10**12)
            scores.append(-rank_idx)  # higher score = more popular

        # Sort candidates by score descending
        order = np.argsort(-np.array(scores))
        ranked_candidates = [candidates[i] for i in order]

        # Find positive position
        pos_pos = ranked_candidates.index(pos_item)

        # Metrics for each K
        for k in KS:
            m = eval_user(pos_pos, k)
            totals[k]["Precision"] += m["Precision"]
            totals[k]["Recall"] += m["Recall"]
            totals[k]["NDCG"] += m["NDCG"]
            totals[k]["HitRate"] += m["HitRate"]

        users_evaluated += 1

        if users_evaluated % 20000 == 0:
            print(f"Evaluated {users_evaluated} users...")

    # Average
    results = {"model": "Popularity", "users_evaluated": users_evaluated, "num_negatives": NUM_NEGATIVES}
    for k in KS:
        results[f"Precision@{k}"] = totals[k]["Precision"] / users_evaluated
        results[f"Recall@{k}"] = totals[k]["Recall"] / users_evaluated
        results[f"NDCG@{k}"] = totals[k]["NDCG"] / users_evaluated
        results[f"HitRate@{k}"] = totals[k]["HitRate"] / users_evaluated

    # Save CSV
    out_df = pd.DataFrame([results])
    out_df.to_csv(OUT_CSV, index=False)

    print("\n✅ Popularity Validation Results:")
    print(results)
    print("\n✅ Saved to:", OUT_CSV)


if __name__ == "__main__":
    main()
