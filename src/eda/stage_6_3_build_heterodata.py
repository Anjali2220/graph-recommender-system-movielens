import os
import json
import torch
import pandas as pd
from torch_geometric.data import HeteroData


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def load_edge_csv(path: str, src_col: str, dst_col: str):
    df = pd.read_csv(path)
    return torch.tensor(df[[src_col, dst_col]].values.T, dtype=torch.long)  # shape [2, num_edges]


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    EDGE_DIR = os.path.join(PROJECT_ROOT, "outputs", "gnn", "graph")
    META_PATH = os.path.join(EDGE_DIR, "edge_type_metadata.json")

    OUT_GRAPH = os.path.join(EDGE_DIR, "hetero_graph_train.pt")
    OUT_META = os.path.join(EDGE_DIR, "hetero_graph_metadata.json")

    print("Project root:", PROJECT_ROOT)
    print("Edge dir     :", EDGE_DIR)

    # --------------------------
    # Load metadata
    # --------------------------
    with open(META_PATH, "r") as f:
        meta = json.load(f)

    n_user = meta["node_counts"]["user"]
    n_movie = meta["node_counts"]["movie"]
    n_genre = meta["node_counts"]["genre"]
    n_tag = meta["node_counts"]["tag"]

    # --------------------------
    # Initialize HeteroData
    # --------------------------
    data = HeteroData()

    # Node counts (we set dummy features; embeddings will be learned later)
    data["user"].num_nodes = n_user
    data["movie"].num_nodes = n_movie
    data["genre"].num_nodes = n_genre
    data["tag"].num_nodes = n_tag

    # Optional: add dummy x so PyG is happy
    data["user"].x = torch.arange(n_user).view(-1, 1)
    data["movie"].x = torch.arange(n_movie).view(-1, 1)
    data["genre"].x = torch.arange(n_genre).view(-1, 1)
    data["tag"].x = torch.arange(n_tag).view(-1, 1)

    # --------------------------
    # Load edges
    # --------------------------
    # user -> movie
    um_path = os.path.join(EDGE_DIR, "edges_user_movie_train.csv")
    data[("user", "rates", "movie")].edge_index = load_edge_csv(
        um_path, src_col="user_idx", dst_col="movie_idx"
    )

    # movie -> user (reverse)
    mu_path = os.path.join(EDGE_DIR, "edges_movie_user_train_rev.csv")
    data[("movie", "rev_rates", "user")].edge_index = load_edge_csv(
        mu_path, src_col="src_movie_idx", dst_col="dst_user_idx"
    )

    # movie -> genre
    mg_path = os.path.join(EDGE_DIR, "edges_movie_genre.csv")
    data[("movie", "has_genre", "genre")].edge_index = load_edge_csv(
        mg_path, src_col="movie_idx", dst_col="genre_idx"
    )

    # genre -> movie (reverse)
    gm_path = os.path.join(EDGE_DIR, "edges_genre_movie_rev.csv")
    data[("genre", "rev_has_genre", "movie")].edge_index = load_edge_csv(
        gm_path, src_col="src_genre_idx", dst_col="dst_movie_idx"
    )

    # movie -> tag
    mt_path = os.path.join(EDGE_DIR, "edges_movie_tag.csv")
    data[("movie", "has_tag", "tag")].edge_index = load_edge_csv(
        mt_path, src_col="movie_idx", dst_col="tag_idx"
    )

    # tag -> movie (reverse)
    tm_path = os.path.join(EDGE_DIR, "edges_tag_movie_rev.csv")
    data[("tag", "rev_has_tag", "movie")].edge_index = load_edge_csv(
        tm_path, src_col="src_tag_idx", dst_col="dst_movie_idx"
    )

    # --------------------------
    # Print summary
    # --------------------------
    print("\n✅ HeteroData summary:")
    print(data)
    print("\nNode counts:")
    print("users :", data['user'].num_nodes)
    print("movies:", data['movie'].num_nodes)
    print("genres:", data['genre'].num_nodes)
    print("tags  :", data['tag'].num_nodes)

    print("\nEdge counts:")
    for etype in data.edge_types:
        ecount = data[etype].edge_index.size(1)
        print(f"{etype}: {ecount}")

    # --------------------------
    # Save graph
    # --------------------------
    torch.save(data, OUT_GRAPH)
    print("\n✅ Saved graph to:", OUT_GRAPH)

    # Save metadata for report
    summary = {
        "node_counts": {
            "user": int(data["user"].num_nodes),
            "movie": int(data["movie"].num_nodes),
            "genre": int(data["genre"].num_nodes),
            "tag": int(data["tag"].num_nodes),
        },
        "edge_counts": {
            str(etype): int(data[etype].edge_index.size(1)) for etype in data.edge_types
        },
        "edge_types": [str(e) for e in data.edge_types],
        "notes": "Graph is built from train interactions only; genre/tag edges are metadata; reverse edges added for message passing."
    }

    with open(OUT_META, "w") as f:
        json.dump(summary, f, indent=4)

    print("✅ Saved graph metadata to:", OUT_META)


if __name__ == "__main__":
    main()