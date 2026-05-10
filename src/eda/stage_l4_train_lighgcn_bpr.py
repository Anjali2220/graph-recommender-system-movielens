# src/eda/stage_l4_train_lightgcn_bpr.py

import json
import time
import random
import argparse
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.optim as optim


# =========================
# Utils
# =========================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "src").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pretty(obj) -> str:
    return json.dumps(obj, indent=4, ensure_ascii=False)


def load_pos_sets_npz(npz_path: Path, n_users: int) -> List[np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    keys = set(data.files)

    # Your format (user_ptr, pos_items_flat)
    if {"user_ptr", "pos_items_flat"}.issubset(keys):
        user_ptr = data["user_ptr"].astype(np.int64)
        pos_items_flat = data["pos_items_flat"].astype(np.int64)
        if len(user_ptr) != n_users + 1:
            raise ValueError(f"user_ptr length {len(user_ptr)} != n_users+1 ({n_users+1})")
        out = []
        for u in range(n_users):
            s, e = user_ptr[u], user_ptr[u + 1]
            out.append(pos_items_flat[s:e])
        return out

    # CSR style (indptr, indices)
    if {"indptr", "indices"}.issubset(keys):
        indptr = data["indptr"].astype(np.int64)
        indices = data["indices"].astype(np.int64)
        if len(indptr) != n_users + 1:
            raise ValueError(f"indptr length {len(indptr)} != n_users+1 ({n_users+1})")
        out = []
        for u in range(n_users):
            s, e = indptr[u], indptr[u + 1]
            out.append(indices[s:e])
        return out

    raise ValueError(f"Unsupported NPZ keys: {sorted(list(keys))}")


def import_lightgcn(project_root: Path):
    """
    Robust import to your L3 file.
    Must expose class LightGCN with:
      - get_all_embeddings(A_hat)
      - score_user_item(...)
    We'll build A_hat in this file.
    """
    candidates = [
        project_root / "src" / "eda" / "stage_l3_lightgcn_model.py",
        project_root / "src" / "gnn" / "lightgcn_model.py",
    ]

    errors = []
    for file_path in candidates:
        if file_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("lightgcn_module", str(file_path))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "LightGCN") and hasattr(mod, "build_norm_adj"):
                    return mod.LightGCN, mod.build_norm_adj, str(file_path)
                errors.append(f"{file_path} exists but missing LightGCN/build_norm_adj")
            except Exception as e:
                errors.append(f"Failed importing {file_path}: {e}")

    raise ImportError("Could not import LightGCN:\n" + "\n".join(errors))


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()


# =========================
# Sampler with hard negatives
# =========================
class BPRTripletSampler:
    def __init__(
        self,
        pos_lists: List[np.ndarray],
        n_items: int,
        popular_items: np.ndarray,
        hard_prob: float = 0.30,
        seed: int = 42,
    ):
        self.pos_lists = pos_lists
        self.n_users = len(pos_lists)
        self.n_items = int(n_items)
        self.popular_items = popular_items.astype(np.int64)
        self.hard_prob = float(hard_prob)

        self.rng = np.random.default_rng(seed)
        self.valid_users = [u for u in range(self.n_users) if len(pos_lists[u]) > 0]
        if not self.valid_users:
            raise ValueError("No users with positive interactions in pos_lists.")

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        users = self.rng.choice(self.valid_users, size=batch_size, replace=True)
        pos = np.empty(batch_size, dtype=np.int64)
        neg = np.empty(batch_size, dtype=np.int64)

        for idx, u in enumerate(users):
            pos_items = self.pos_lists[u]
            pos[idx] = pos_items[self.rng.integers(0, len(pos_items))]

            # negative: 30% popular(hard), 70% uniform(random)
            while True:
                if self.rng.random() < self.hard_prob:
                    j = int(self.rng.choice(self.popular_items))
                else:
                    j = int(self.rng.integers(0, self.n_items))
                if j not in pos_items:
                    neg[idx] = j
                    break

        return users.astype(np.int64), pos, neg


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tag", type=str, default="lightgcn_strong_run")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--edge_index_pt", type=str, default="")
    parser.add_argument("--edge_info_json", type=str, default="")
    parser.add_argument("--pos_npz", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")

    # Strong M4 CPU-safe defaults
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--K", type=int, default=3)

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # hard negatives
    parser.add_argument("--hard_prob", type=float, default=0.30)
    parser.add_argument("--popular_topk", type=int, default=5000)

    parser.add_argument("--log_every", type=int, default=50)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_root = find_project_root(Path(__file__).resolve().parent)

    edge_index_pt = Path(args.edge_index_pt) if args.edge_index_pt else (
        project_root / "outputs" / "gnn" / "lightgcn" / "train_graph" / "edge_index.pt"
    )
    edge_info_json = Path(args.edge_info_json) if args.edge_info_json else (
        project_root / "outputs" / "gnn" / "lightgcn" / "train_graph" / "edge_index_info.json"
    )
    pos_npz = Path(args.pos_npz) if args.pos_npz else (
        project_root / "outputs" / "gnn" / "data" / "train_pos_sets.npz"
    )
    out_dir = Path(args.out_dir) if args.out_dir else (
        project_root / "outputs" / "gnn" / "lightgcn" / "runs" / args.tag
    )
    ensure_dir(out_dir)

    print("============================================")
    print("✅ TASK L4 — Train LightGCN (BPR) — STRONG CPU MODE")
    print("============================================")
    print(f"edge_index.pt  : {edge_index_pt}")
    print(f"edge_info.json : {edge_info_json}")
    print(f"pos_npz        : {pos_npz}")
    print(f"Out dir        : {out_dir}")
    print("--------------------------------------------")
    print(f"emb_dim        : {args.emb_dim}")
    print(f"K              : {args.K}")
    print(f"epochs         : {args.epochs}")
    print(f"batch_size     : {args.batch_size}")
    print(f"steps/epoch    : {args.steps_per_epoch}")
    print(f"lr             : {args.lr}")
    print(f"hard_prob      : {args.hard_prob}")
    print(f"popular_topk   : {args.popular_topk}")
    print("============================================")

    if not edge_index_pt.exists():
        raise FileNotFoundError(f"Missing: {edge_index_pt}")
    if not edge_info_json.exists():
        raise FileNotFoundError(f"Missing: {edge_info_json}")
    if not pos_npz.exists():
        raise FileNotFoundError(f"Missing: {pos_npz}")

    edge_info = json.loads(edge_info_json.read_text(encoding="utf-8"))
    n_users = int(edge_info["n_users"])
    n_items = int(edge_info["n_items"])
    num_nodes = int(edge_info["num_nodes"])

    # import model + adj builder from L3
    LightGCN, build_norm_adj, imported_from = import_lightgcn(project_root)
    print(f"✅ Imported LightGCN from: {imported_from}")

    edge_index = torch.load(edge_index_pt, map_location="cpu")

    # Build normalized adjacency once (CPU)
    print("Building normalized adjacency A_hat (CPU)...")
    A_hat = build_norm_adj(edge_index, num_nodes=num_nodes)

    model = LightGCN(n_users=n_users, n_items=n_items, emb_dim=args.emb_dim, K=args.K)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # positives
    pos_lists = load_pos_sets_npz(pos_npz, n_users=n_users)

    # build popularity list for hard negatives
    flat_all = np.concatenate([arr for arr in pos_lists if len(arr) > 0])
    item_pop = np.bincount(flat_all, minlength=n_items)
    popular_items = np.argsort(-item_pop)[: args.popular_topk]

    sampler = BPRTripletSampler(
        pos_lists=pos_lists,
        n_items=n_items,
        popular_items=popular_items,
        hard_prob=args.hard_prob,
        seed=args.seed,
    )

    # save config
    run_cfg = {
        "tag": args.tag,
        "seed": args.seed,
        "imported_from": imported_from,
        "edge_index_pt": str(edge_index_pt),
        "pos_npz": str(pos_npz),
        "n_users": n_users,
        "n_items": n_items,
        "emb_dim": args.emb_dim,
        "K": args.K,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "steps_per_epoch": args.steps_per_epoch,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "hard_prob": args.hard_prob,
        "popular_topk": args.popular_topk,
    }
    (out_dir / "run_config.json").write_text(pretty(run_cfg), encoding="utf-8")

    model.train()
    log_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = 0.0

        for step in range(1, args.steps_per_epoch + 1):
            u, pos_i, neg_j = sampler.sample_batch(args.batch_size)

            u = torch.from_numpy(u).long()
            pos_i = torch.from_numpy(pos_i).long()
            neg_j = torch.from_numpy(neg_j).long()

            optimizer.zero_grad()

            Z_user, Z_item = model.get_all_embeddings(A_hat)

            pos_scores = model.score_user_item(Z_user, Z_item, u, pos_i)
            neg_scores = model.score_user_item(Z_user, Z_item, u, neg_j)

            loss = bpr_loss(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()

            running += float(loss.item())

            if step % args.log_every == 0:
                print(f"Epoch {epoch:02d} | step {step:04d}/{args.steps_per_epoch} | loss {running/args.log_every:.5f}")
                running = 0.0

        epoch_time = time.time() - t0
        ckpt = out_dir / f"model_epoch_{epoch}.pt"
        torch.save(model.state_dict(), ckpt)
        log_rows.append({"epoch": epoch, "time_sec": round(epoch_time, 2), "ckpt": ckpt.name})
        print(f"✅ Epoch {epoch} done | time={epoch_time:.1f}s | saved {ckpt.name}")

    final_path = out_dir / "model_final.pt"
    torch.save(model.state_dict(), final_path)

    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)

    print("============================================")
    print("✅ LightGCN STRONG training finished")
    print(f"Final model: {final_path}")
    print(f"Log CSV    : {out_dir / 'train_log.csv'}")
    print("============================================")


if __name__ == "__main__":
    main()