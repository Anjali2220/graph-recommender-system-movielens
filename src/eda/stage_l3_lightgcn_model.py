import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


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


def build_norm_adj(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """
    LightGCN uses normalized adjacency:
      A_hat = D^{-1/2} A D^{-1/2}

    We build A_hat as a torch sparse COO tensor.
    edge_index: [2, E] with integer node indices in [0, num_nodes)
    """
    assert edge_index.dim() == 2 and edge_index.size(0) == 2

    row = edge_index[0].long()
    col = edge_index[1].long()

    # unweighted adjacency -> values = 1
    val = torch.ones(row.size(0), dtype=torch.float32)

    # degree
    deg = torch.zeros(num_nodes, dtype=torch.float32)
    deg.scatter_add_(0, row, val)

    deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)  # avoid div by 0
    norm_val = deg_inv_sqrt[row] * val * deg_inv_sqrt[col]

    A_hat = torch.sparse_coo_tensor(
        indices=torch.stack([row, col], dim=0),
        values=norm_val,
        size=(num_nodes, num_nodes),
        dtype=torch.float32,
    ).coalesce()

    return A_hat


# =========================
# LightGCN Model
# =========================
class LightGCN(nn.Module):
    """
    Bipartite LightGCN on a single graph with nodes:
      [0 ... n_users-1]             = users
      [n_users ... n_users+n_items-1] = items (movies)
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64, K: int = 3):
        super().__init__()
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.num_nodes = self.n_users + self.n_items
        self.emb_dim = int(emb_dim)
        self.K = int(K)

        self.user_emb = nn.Embedding(self.n_users, self.emb_dim)
        self.item_emb = nn.Embedding(self.n_items, self.emb_dim)

        # LightGCN standard init
        nn.init.normal_(self.user_emb.weight, std=0.1)
        nn.init.normal_(self.item_emb.weight, std=0.1)

    def get_all_embeddings(self, A_hat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          Z_user: [n_users, emb_dim]
          Z_item: [n_items, emb_dim]
        """
        # E0: concatenate user + item embeddings into one matrix
        E0 = torch.cat([self.user_emb.weight, self.item_emb.weight], dim=0)  # [N, d]
        E_list = [E0]

        Ek = E0
        for _ in range(self.K):
            Ek = torch.sparse.mm(A_hat, Ek)  # [N, d]
            E_list.append(Ek)

        # final = mean over 0..K
        E_final = torch.stack(E_list, dim=0).mean(dim=0)  # [N, d]

        Z_user = E_final[: self.n_users]
        Z_item = E_final[self.n_users :]
        return Z_user, Z_item

    def score_user_item(self, Z_user: torch.Tensor, Z_item: torch.Tensor,
                        user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Dot-product scoring for (user_idx, item_idx).
        user_idx: [B]
        item_idx: [B] in [0..n_items-1] (movie_idx space)
        """
        u = Z_user[user_idx]
        i = Z_item[item_idx]
        return (u * i).sum(dim=-1)  # [B]


# =========================
# Quick sanity runner (optional)
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge_index_pt", type=str, default="")
    parser.add_argument("--n_users", type=int, default=138238)
    parser.add_argument("--n_items", type=int, default=20639)
    parser.add_argument("--emb_dim", type=int, default=64)
    parser.add_argument("--K", type=int, default=3)
    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve().parent)

    edge_index_pt = Path(args.edge_index_pt) if args.edge_index_pt else (
        project_root / "outputs" / "gnn" / "lightgcn" / "train_graph" / "edge_index.pt"
    )

    print("============================================")
    print("✅ TASK L3 — LightGCN Model (Sanity Test)")
    print("============================================")
    print(f"edge_index.pt : {edge_index_pt}")
    print(f"n_users       : {args.n_users}")
    print(f"n_items       : {args.n_items}")
    print(f"emb_dim       : {args.emb_dim}")
    print(f"K             : {args.K}")
    print("============================================")

    edge_index = torch.load(edge_index_pt, map_location="cpu")
    num_nodes = args.n_users + args.n_items
    A_hat = build_norm_adj(edge_index, num_nodes=num_nodes)

    model = LightGCN(n_users=args.n_users, n_items=args.n_items, emb_dim=args.emb_dim, K=args.K)

    with torch.no_grad():
        Z_user, Z_item = model.get_all_embeddings(A_hat)
        print("✅ Embeddings computed:")
        print("Z_user:", tuple(Z_user.shape))
        print("Z_item:", tuple(Z_item.shape))

        # small scoring check
        u = torch.tensor([0, 1, 2], dtype=torch.long)
        i = torch.tensor([0, 10, 20], dtype=torch.long)
        scores = model.score_user_item(Z_user, Z_item, u, i)
        print("✅ Sample scores:", scores.tolist())

    print("✅ L3 done (model implemented). Next is L4 training loop (BPR).")


if __name__ == "__main__":
    main()