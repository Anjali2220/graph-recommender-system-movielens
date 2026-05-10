import os
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ✅ Updated import (because your mf_model.py is in src/eda/)
from src.eda.mf_model import MFModel, MFConfig


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


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    MF_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")

    PAIRS_PATH = os.path.join(MF_DIR, "train_pairs.npy")
    USER_IDS_PATH = os.path.join(MF_DIR, "user_ids.npy")
    ITEM_IDS_PATH = os.path.join(MF_DIR, "item_ids.npy")

    OUT_MODEL_PATH = os.path.join(MF_DIR, "mf_bpr_model.pt")
    OUT_TRAIN_LOG = os.path.join(MF_DIR, "train_log.json")

    # Hyperparameters
    emb_dim = 64
    lr = 1e-3
    batch_size = 4096
    epochs = 5
    reg_lambda = 1e-6
    num_workers = 0  # ✅ safer for Mac
    seed = 42

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Project root:", PROJECT_ROOT)
    print("MF dir      :", MF_DIR)
    print("Device      :", device)

    if not os.path.exists(PAIRS_PATH):
        raise FileNotFoundError(f"Missing {PAIRS_PATH}. Run MF-1 first.")

    pairs = np.load(PAIRS_PATH)
    user_ids = np.load(USER_IDS_PATH)
    item_ids = np.load(ITEM_IDS_PATH)

    n_users = len(user_ids)
    n_items = len(item_ids)

    print("\nLoaded MF-1 data:")
    print("Pairs:", pairs.shape)
    print("Users:", n_users, "| Items:", n_items)

    print("\nBuilding user positive sets...")
    user_pos_sets = build_user_pos_sets(pairs, n_users=n_users)

    dataset = BPRDataset(pairs, user_pos_sets=user_pos_sets, n_items=n_items, seed=seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_bpr,
        drop_last=True,
    )

    config = MFConfig(
        n_users=n_users,
        n_items=n_items,
        emb_dim=emb_dim,
        lr=lr,
        weight_decay=0.0,
        seed=seed,
        device=device,
    )

    torch.manual_seed(seed)
    model = MFModel(n_users=n_users, n_items=n_items, emb_dim=emb_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print("\nMFModel ready.")
    train_history = []
    global_step = 0

    print("\nStarting BPR training...")
    for epoch in range(1, epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        batches = 0

        for u, pos, neg in loader:
            u, pos, neg = u.to(device), pos.to(device), neg.to(device)

            optimizer.zero_grad()
            pos_scores = model(u, pos)
            neg_scores = model(u, neg)

            loss = bpr_loss(pos_scores, neg_scores)

            if reg_lambda > 0:
                loss = loss + reg_lambda * model.l2_reg(u, pos, neg)

            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            batches += 1
            global_step += 1

            if global_step % 200 == 0:
                print(f"Epoch {epoch} | step {global_step} | avg_loss {running_loss / batches:.4f}")

        epoch_loss = running_loss / max(1, batches)
        epoch_time = time.time() - t0

        print(f"\n✅ Epoch {epoch}/{epochs} finished | loss={epoch_loss:.6f} | time={epoch_time:.1f}s")

        train_history.append({"epoch": epoch, "loss": epoch_loss, "seconds": epoch_time})

        extra = {"config": config.__dict__, "epoch": epoch, "train_history": train_history}
        model.save(OUT_MODEL_PATH, extra=extra)
        print("Saved checkpoint:", OUT_MODEL_PATH)

    with open(OUT_TRAIN_LOG, "w") as f:
        json.dump({"config": config.__dict__, "train_history": train_history}, f, indent=2)

    print("\n✅ Training complete.")
    print("Model:", OUT_MODEL_PATH)
    print("Log  :", OUT_TRAIN_LOG)


if __name__ == "__main__":
    main()
