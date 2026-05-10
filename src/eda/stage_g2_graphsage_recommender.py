import argparse
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv, HeteroConv


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


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# =========================
# GraphSAGE Recommender
# =========================
class GraphSAGERecommender(nn.Module):
    """
    Heterogeneous GraphSAGE recommender using:
      - trainable node-type embeddings
      - hetero GraphSAGE message passing
      - layer-wise embedding averaging
      - user/movie projection heads

    Output:
      z_user  -> final user embeddings
      z_movie -> final movie embeddings

    Score:
      dot(z_user, z_movie)
    """

    def __init__(
        self,
        metadata,
        num_nodes_dict: Dict[str, int],
        emb_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.metadata = metadata
        self.num_nodes_dict = num_nodes_dict
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # -----------------------------------
        # Trainable embeddings per node type
        # -----------------------------------
        self.embeddings = nn.ModuleDict()
        for node_type, n_nodes in num_nodes_dict.items():
            emb = nn.Embedding(n_nodes, emb_dim)
            nn.init.xavier_uniform_(emb.weight)
            self.embeddings[node_type] = emb

        # -----------------------------------
        # Hetero GraphSAGE layers
        # -----------------------------------
        self.convs = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = emb_dim if layer_idx == 0 else hidden_dim
            out_dim = hidden_dim

            conv_dict = {}
            for edge_type in metadata[1]:
                conv_dict[edge_type] = SAGEConv(
                    (in_dim, in_dim),
                    out_dim,
                    aggr="mean",
                )

            self.convs.append(HeteroConv(conv_dict, aggr="sum"))

        # -----------------------------------
        # Projection heads
        # -----------------------------------
        self.user_proj = nn.Linear(hidden_dim, hidden_dim)
        self.movie_proj = nn.Linear(hidden_dim, hidden_dim)

        # project input embeddings to hidden_dim
        self.input_proj = nn.ModuleDict()
        for node_type in num_nodes_dict.keys():
            self.input_proj[node_type] = nn.Linear(emb_dim, hidden_dim)

    def forward(self, data):
        """
        data: PyG HeteroData

        returns:
            {
                "user": z_user,
                "movie": z_movie,
                "genre": z_genre,
                "tag": z_tag
            }
        """

        # -----------------------------------
        # Initial node embeddings
        # -----------------------------------
        x_dict = {}
        for node_type in data.node_types:
            x_dict[node_type] = self.embeddings[node_type].weight

        # store outputs from all layers
        layer_outputs = []

        # initial projected embeddings
        x0_dict = {
            node_type: self.input_proj[node_type](x_dict[node_type])
            for node_type in x_dict
        }
        layer_outputs.append(x0_dict)

        # -----------------------------------
        # Message passing
        # -----------------------------------
        x_cur = x_dict
        for conv in self.convs:
            x_cur = conv(x_cur, data.edge_index_dict)

            x_cur = {
                k: F.dropout(F.relu(v), p=self.dropout, training=self.training)
                for k, v in x_cur.items()
            }

            layer_outputs.append(x_cur)

        # -----------------------------------
        # Layer-wise averaging
        # -----------------------------------
        final_dict = {}
        for node_type in layer_outputs[0].keys():
            stacked = torch.stack(
                [layer[node_type] for layer in layer_outputs],
                dim=0
            )
            final_dict[node_type] = stacked.mean(dim=0)

        # -----------------------------------
        # Final projections for user/movie
        # -----------------------------------
        final_dict["user"] = self.user_proj(final_dict["user"])
        final_dict["movie"] = self.movie_proj(final_dict["movie"])

        return final_dict

    def score(
        self,
        z_user: torch.Tensor,
        z_movie: torch.Tensor,
        user_idx: torch.Tensor,
        movie_idx: torch.Tensor
    ) -> torch.Tensor:
        """
        Dot-product scoring between selected users and movies.
        """
        u = z_user[user_idx]
        m = z_movie[movie_idx]
        return (u * m).sum(dim=-1)


# =========================
# Sanity Test Runner
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_pt", type=str, default="")
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)

    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve().parent)
    graph_pt = Path(args.graph_pt) if args.graph_pt else (
        project_root / "outputs" / "gnn" / "graph" / "hetero_graph_train.pt"
    )

    print("============================================")
    print("✅ TASK G2 — GraphSAGE Recommender (Sanity Test)")
    print("============================================")
    print(f"Graph pt     : {graph_pt}")
    print(f"emb_dim      : {args.emb_dim}")
    print(f"hidden_dim   : {args.hidden_dim}")
    print(f"num_layers   : {args.num_layers}")
    print(f"dropout      : {args.dropout}")
    print("============================================")

    if not graph_pt.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_pt}")

    data = safe_torch_load(graph_pt, map_location="cpu")

    metadata = data.metadata()
    num_nodes_dict = {ntype: int(data[ntype].num_nodes) for ntype in data.node_types}

    model = GraphSAGERecommender(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    model.eval()
    with torch.no_grad():
        z_dict = model(data)

    print("✅ Output embedding shapes:")
    for k, v in z_dict.items():
        print(f"{k}: {tuple(v.shape)}")

    u = torch.tensor([0, 1, 2], dtype=torch.long)
    i = torch.tensor([0, 10, 20], dtype=torch.long)
    scores = model.score(z_dict["user"], z_dict["movie"], u, i)
    print("✅ Sample scores:", scores.tolist())

    print("============================================")
    print("✅ G2 complete. Next: G3 training loop (BPR)")
    print("============================================")


if __name__ == "__main__":
    main()