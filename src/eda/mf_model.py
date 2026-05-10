import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict


@dataclass
class MFConfig:
    n_users: int
    n_items: int
    emb_dim: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-6   # L2 regularization strength (we also use explicit reg in BPR)
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class MFModel(nn.Module):
    """
    Classic Matrix Factorization for implicit ranking:
      score(u, i) = dot(U[u], V[i])
    """

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 64):
        super().__init__()

        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)

        # Good initialization helps training stability
        nn.init.normal_(self.user_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_emb.weight, mean=0.0, std=0.01)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Returns predicted preference score for (user, item).
        user_idx: (B,)
        item_idx: (B,)
        output:   (B,)
        """
        u = self.user_emb(user_idx)  # (B, D)
        v = self.item_emb(item_idx)  # (B, D)
        return (u * v).sum(dim=1)    # dot product

    def l2_reg(self, user_idx: torch.Tensor, pos_idx: torch.Tensor, neg_idx: torch.Tensor) -> torch.Tensor:
        """
        L2 regularization over embeddings involved in a BPR batch.
        """
        u = self.user_emb(user_idx)
        p = self.item_emb(pos_idx)
        n = self.item_emb(neg_idx)
        return (u.pow(2).sum() + p.pow(2).sum() + n.pow(2).sum()) / user_idx.shape[0]

    def save(self, path: str, extra: Dict = None):
        payload = {
            "state_dict": self.state_dict(),
            "extra": extra or {}
        }
        torch.save(payload, path)

    @staticmethod
    def load(path: str, n_users: int, n_items: int, emb_dim: int, map_location=None):
        ckpt = torch.load(path, map_location=map_location)
        model = MFModel(n_users=n_users, n_items=n_items, emb_dim=emb_dim)
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt.get("extra", {})
