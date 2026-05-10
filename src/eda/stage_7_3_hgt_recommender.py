import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv


class HGTRecommender(nn.Module):
    """
    Heterogeneous Graph Transformer Recommender
    ------------------------------------------------
    Node types:
        - user
        - movie
        - genre
        - tag

    Edge types:
        - user -> movie (rates)
        - movie -> user (rev_rates)
        - movie -> genre
        - genre -> movie
        - movie -> tag
        - tag -> movie

    Scoring:
        dot(z_user, z_movie)
    """

    def __init__(
        self,
        metadata,
        num_nodes_dict,
        emb_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
    ):
        super().__init__()

        self.metadata = metadata
        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # --------------------------------------------------
        # 1️⃣ Trainable embeddings for each node type
        # --------------------------------------------------
        self.embeddings = nn.ModuleDict()
        for node_type, num_nodes in num_nodes_dict.items():
            self.embeddings[node_type] = nn.Embedding(
                num_nodes,
                emb_dim
            )

        # --------------------------------------------------
        # 2️⃣ HGT Layers
        # --------------------------------------------------
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(
                in_channels=emb_dim,
                out_channels=emb_dim,
                metadata=metadata,
                heads=num_heads,
            )
            self.convs.append(conv)

        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

    # ------------------------------------------------------
    # Initialization
    # ------------------------------------------------------
    def reset_parameters(self):
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

        for conv in self.convs:
            conv.reset_parameters()

    # ------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------
    def forward(self, data):
        """
        Returns:
            z_dict: dictionary of node embeddings after HGT
        """

        # Initial embedding lookup
        x_dict = {
            node_type: self.embeddings[node_type](
                torch.arange(
                    self.embeddings[node_type].num_embeddings,
                    device=data[node_type].num_nodes_device
                    if hasattr(data[node_type], "num_nodes_device")
                    else next(self.parameters()).device,
                )
            )
            for node_type in data.node_types
        }

        # HGT layers
        for conv in self.convs:
            x_dict = conv(x_dict, data.edge_index_dict)
            x_dict = {
                key: F.relu(val)
                for key, val in x_dict.items()
            }
            x_dict = {
                key: self.dropout_layer(val)
                for key, val in x_dict.items()
            }

        return x_dict

    # ------------------------------------------------------
    # Get user & movie embeddings
    # ------------------------------------------------------
    def get_user_movie_embeddings(self, data):
        z_dict = self.forward(data)
        z_user = z_dict["user"]
        z_movie = z_dict["movie"]
        return z_user, z_movie

    # ------------------------------------------------------
    # Score function (dot product)
    # ------------------------------------------------------
    def score(self, user_emb, movie_emb):
        return torch.sum(user_emb * movie_emb, dim=1)