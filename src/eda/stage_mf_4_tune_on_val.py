import os
import json
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# ✅ import matches your current structure (mf_model.py is in src/eda/)
from src.eda.mf_model import MFModel, MFConfig


# ----------------------------
# BPR Dataset (negative sampled on-the-fly)
# ----------------------------
class BPRDataset(Dataset):
    def __init__(self, pairs: np.ndarray, user_pos_sets: list, n_items: int, seed: int = 42):
        self.pairs = pairs
        self.user_pos_sets = user_pos_sets
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, idx):
        u = int(self.pairs[idx, 0])
        pos = int(self.pairs[idx, 1])

        pos_set = self.user_pos_sets[u]
        while True:
            neg = int(self.rng.integers(0, self.n_items))
            if neg not in pos_set:
                break

        return u, pos, neg


def collate_bpr(batch):
    u, p, n = zip(*batch)
    return (
        torch.tensor(u, dtype=torch.long),
        torch.tensor(p, dtype=torch.long),
        torch.tensor(n, dtype=torch.long),
    )


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8))


def build_user_pos_sets(pairs: np.ndarray, n_users: int):
    user_pos_sets = [set() for _ in range(n_users)]
    for u, i in pairs:
        user_pos_sets[int(u)].add(int(i))
    return user_pos_sets


# ----------------------------
# Metrics for leave-one-out (1 positive)
# ----------------------------
def metrics_for_one_positive(rank_1based: int, K: int):
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
# Train one MF model (BPR)
# ----------------------------
def train_one_mf(
    pairs: np.ndarray,
    n_users: int,
    n_items: int,
    user_pos_sets: list,
    emb_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
    reg_lambda: float,
    device: str,
    seed: int,
    num_workers: int = 0,
):
    torch.manual_seed(seed)
    model = MFModel(n_users=n_users, n_items=n_items, emb_dim=emb_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    ds = BPRDataset(pairs, user_pos_sets=user_pos_sets, n_items=n_items, seed=seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_bpr,
        drop_last=True
    )

    train_hist = []
    for ep in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        batches = 0

        for u, pos, neg in loader:
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)

            opt.zero_grad()
            pos_s = model(u, pos)
            neg_s = model(u, neg)

            loss = bpr_loss(pos_s, neg_s)
            if reg_lambda > 0:
                loss = loss + reg_lambda * model.l2_reg(u, pos, neg)

            loss.backward()
            opt.step()

            running += float(loss.item())
            batches += 1

        ep_loss = running / max(1, batches)
        ep_time = time.time() - t0
        train_hist.append({"epoch": ep, "loss": ep_loss, "seconds": ep_time})

    return model, train_hist


# ----------------------------
# Evaluate MF on val (1 pos + 499 neg)
# Efficient scoring by using embedding matrices directly
# ----------------------------
def evaluate_mf_on_val(
    user_emb: np.ndarray,      # (n_users, dim)
    item_emb: np.ndarray,      # (n_items, dim)
    val_df: pd.DataFrame,      # columns: user_idx, pos_item_idx
    user_pos_sets: list,
    n_items: int,
    K_list=(10, 20),
    num_negatives=499,
    seed=42,
    max_users=None,
):
    rng = np.random.default_rng(seed)

    if max_users is not None and len(val_df) > max_users:
        val_df = val_df.sample(n=max_users, random_state=seed).reset_index(drop=True)

    sums = {f"{m}@{K}": 0.0 for K in K_list for m in ["Precision", "Recall", "NDCG", "HitRate"]}
    used = 0

    for row in val_df.itertuples(index=False):
        u = int(row.user_idx)
        pos = int(row.pos_item_idx)

        pos_set = user_pos_sets[u]
        # sample negatives not in train positives, and not the pos
        negatives = []
        while len(negatives) < num_negatives:
            cand = int(rng.integers(0, n_items))
            if cand != pos and cand not in pos_set:
                negatives.append(cand)

        candidates = np.array([pos] + negatives, dtype=np.int32)

        # scores = dot(user_emb[u], item_emb[candidates])
        uvec = user_emb[u]  # (dim,)
        scores = item_emb[candidates] @ uvec  # (500,)

        order = np.argsort(-scores)
        ranked = candidates[order]

        where = np.where(ranked == pos)[0]
        pos_rank = int(where[0]) + 1 if len(where) > 0 else None

        for K in K_list:
            mets = metrics_for_one_positive(pos_rank, K)
            for k, v in mets.items():
                sums[k] += v

        used += 1

    results = {k: v / used for k, v in sums.items()}
    results["users_evaluated"] = used
    results["num_negatives"] = num_negatives
    return results


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    VAL_CSV = os.path.join(PROJECT_ROOT, "data", "processed", "val.csv")

    MF_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")
    PAIRS_PATH = os.path.join(MF_DIR, "train_pairs.npy")
    USER_IDS_PATH = os.path.join(MF_DIR, "user_ids.npy")
    ITEM_IDS_PATH = os.path.join(MF_DIR, "item_ids.npy")

    TUNE_DIR = os.path.join(MF_DIR, "tuning")
    os.makedirs(TUNE_DIR, exist_ok=True)

    RESULTS_PATH = os.path.join(TUNE_DIR, "mf_val_tuning_results.csv")

    # ----------------------------
    # Tuning grid (small & practical)
    # ----------------------------
    EMB_DIMS = [32, 64]
    LRS = [1e-3, 5e-4]
    EPOCHS_LIST = [3, 5]

    # Training settings
    batch_size = 4096
    reg_lambda = 1e-6
    seed = 42
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0  # safest on Mac

    # Evaluation settings
    num_negatives = 499
    K_list = (10, 20)

    # If you want faster first run:
    max_users_eval = None  # e.g. 20000 for a quick tune

    print("Device:", device)
    print("Loading MF-1 data from:", MF_DIR)

    pairs = np.load(PAIRS_PATH)
    user_ids = np.load(USER_IDS_PATH)
    item_ids = np.load(ITEM_IDS_PATH)
    n_users = len(user_ids)
    n_items = len(item_ids)

    print("Pairs:", pairs.shape, "| Users:", n_users, "| Items:", n_items)

    # Build user positives from train pairs (for negative sampling)
    user_pos_sets = build_user_pos_sets(pairs, n_users=n_users)

    # Build mappings from ids for val conversion
    user2idx = {int(u): i for i, u in enumerate(user_ids)}
    item2idx = {int(m): i for i, m in enumerate(item_ids)}

    # Load val positives (rating>=4) and map to indices
    val = pd.read_csv(VAL_CSV, usecols=["userId", "movieId", "rating"])
    val = val[val["rating"] >= 4.0].copy()
    val["user_idx"] = val["userId"].map(user2idx)
    val["pos_item_idx"] = val["movieId"].map(item2idx)
    val = val.dropna(subset=["user_idx", "pos_item_idx"]).copy()
    val["user_idx"] = val["user_idx"].astype(np.int32)
    val["pos_item_idx"] = val["pos_item_idx"].astype(np.int32)

    # Keep one per user (should already be one, but safe)
    val = val.drop_duplicates(subset=["user_idx"], keep="last").reset_index(drop=True)

    print("Val users available for eval:", len(val))

    all_rows = []

    for emb_dim in EMB_DIMS:
        for lr in LRS:
            for epochs in EPOCHS_LIST:
                tag = f"dim{emb_dim}_lr{lr}_ep{epochs}"
                model_path = os.path.join(TUNE_DIR, f"mf_{tag}.pt")

                print(f"\n=== Training MF ({tag}) ===")
                model, train_hist = train_one_mf(
                    pairs=pairs,
                    n_users=n_users,
                    n_items=n_items,
                    user_pos_sets=user_pos_sets,
                    emb_dim=emb_dim,
                    lr=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    reg_lambda=reg_lambda,
                    device=device,
                    seed=seed,
                    num_workers=num_workers,
                )

                # Save checkpoint
                extra = {
                    "tag": tag,
                    "train_history": train_hist,
                    "config": {
                        "n_users": n_users,
                        "n_items": n_items,
                        "emb_dim": emb_dim,
                        "lr": lr,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "reg_lambda": reg_lambda,
                        "seed": seed,
                        "device": device,
                    }
                }
                model.save(model_path, extra=extra)
                print("Saved:", model_path)

                # Extract embeddings for fast eval
                model.eval()
                with torch.no_grad():
                    user_emb = model.user_emb.weight.detach().cpu().numpy()
                    item_emb = model.item_emb.weight.detach().cpu().numpy()

                print(f"Evaluating MF ({tag}) on val (1 pos + 499 neg)...")
                res = evaluate_mf_on_val(
                    user_emb=user_emb,
                    item_emb=item_emb,
                    val_df=val[["user_idx", "pos_item_idx"]],
                    user_pos_sets=user_pos_sets,
                    n_items=n_items,
                    K_list=K_list,
                    num_negatives=num_negatives,
                    seed=seed,
                    max_users=max_users_eval,
                )

                row = {
                    "tag": tag,
                    "emb_dim": emb_dim,
                    "lr": lr,
                    "epochs": epochs,
                    **res
                }
                all_rows.append(row)

                print("Done:", row)

    df = pd.DataFrame(all_rows)
    df.to_csv(RESULTS_PATH, index=False)

    print("\n✅ Saved tuning results to:")
    print(RESULTS_PATH)

    print("\nLeaderboard (sorted by NDCG@20):")
    print(df.sort_values("NDCG@20", ascending=False).head(10))


if __name__ == "__main__":
    main()
