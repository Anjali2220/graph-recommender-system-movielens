import argparse
import json
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    """
    Walk upwards until we find a folder that looks like the project root.
    We detect it by presence of 'data' and 'src' folders.
    """
    cur = start.resolve()
    for _ in range(10):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def safe_movie_label(title: str, max_len: int = 22) -> str:
    """Shorten long titles so the graph looks clean."""
    title = title.strip()
    if len(title) <= max_len:
        return title
    return title[: max_len - 3] + "..."


def load_movies_lookup(movies_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(movies_csv)
    # ensure movieId is int
    df["movieId"] = df["movieId"].astype(int)
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Stage 6.7 — Visualize genres-only subgraph for a chosen user."
    )

    # Choose one of these:
    parser.add_argument("--user_idx", type=int, default=None, help="Internal user index (user_idx).")
    parser.add_argument("--userId", type=int, default=None, help="Original MovieLens userId (will be mapped).")

    # Controls
    parser.add_argument("--top_movies", type=int, default=5, help="How many liked movies to show.")
    parser.add_argument("--max_genres_per_movie", type=int, default=3, help="Max genres shown per movie.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (used only if no user is provided).")

    args = parser.parse_args()

    print("============================================")
    print("✅ Stage 6.7 — Graph Visualization (Genres Only, Fixed User)")
    print("============================================")

    project_root = find_project_root(Path(__file__).parent)
    print(f"Project root: {project_root}")

    # Paths (adjusted to match your outputs)
    edges_dir = project_root / "data" / "processed" / "graph_edges"
    gnn_graph_dir = project_root / "outputs" / "gnn" / "graph"

    edges_user_movie = edges_dir / "edges_user_movie_train.csv"
    edges_movie_genre = edges_dir / "edges_movie_genre.csv"
    user_map_path = edges_dir / "user_id_mapping.csv"
    movie_map_path = edges_dir / "movie_id_mapping.csv"
    genre_map_path = edges_dir / "genre_mapping.csv"

    movies_csv = project_root / "data" / "raw" / "movies.csv"

    out_png = gnn_graph_dir / "graph_sample_genres_only_fixed_user.png"
    out_json = gnn_graph_dir / "graph_sample_genres_only_fixed_user.json"

    # sanity checks
    for p in [edges_user_movie, edges_movie_genre, user_map_path, movie_map_path, genre_map_path, movies_csv]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    gnn_graph_dir.mkdir(parents=True, exist_ok=True)

    # Load mappings
    user_map = pd.read_csv(user_map_path)      # columns: userId,user_idx
    movie_map = pd.read_csv(movie_map_path)    # columns: movieId,movie_idx
    genre_map = pd.read_csv(genre_map_path)    # columns: genre,genre_idx

    # Determine user_idx
    if args.user_idx is not None and args.userId is not None:
        raise ValueError("Provide only one: --user_idx OR --userId (not both).")

    if args.user_idx is not None:
        user_idx = int(args.user_idx)

    elif args.userId is not None:
        row = user_map[user_map["userId"] == int(args.userId)]
        if len(row) == 0:
            raise ValueError(f"userId={args.userId} not found in mapping file: {user_map_path}")
        user_idx = int(row["user_idx"].iloc[0])

    else:
        # fallback: pick a random user who has interactions
        rng = np.random.default_rng(args.seed)
        edges_um = pd.read_csv(edges_user_movie)
        user_idx = int(rng.choice(edges_um["user_idx"].unique()))

    # Load edges
    edges_um = pd.read_csv(edges_user_movie)       # user_idx, movie_idx
    edges_mg = pd.read_csv(edges_movie_genre)      # movie_idx, genre_idx

    # Filter to this user's liked movies
    user_movies = edges_um[edges_um["user_idx"] == user_idx]["movie_idx"].values
    if len(user_movies) == 0:
        raise ValueError(f"user_idx={user_idx} has 0 train interactions in edges_user_movie_train.csv")

    # pick top_movies (first N)
    user_movies = user_movies[: args.top_movies]

    # movies lookup (movie_idx -> movieId -> title)
    movies_df = load_movies_lookup(movies_csv)

    # build movie_idx -> movieId
    movie_idx_to_movieId = dict(zip(movie_map["movie_idx"].astype(int), movie_map["movieId"].astype(int)))

    # build genre_idx -> genre name
    genre_idx_to_name = dict(zip(genre_map["genre_idx"].astype(int), genre_map["genre"].astype(str)))

    # Build graph
    G = nx.Graph()

    # Add user node
    user_node = f"user:{user_idx}"
    G.add_node(user_node, ntype="user", label=f"user\n{user_idx}")

    # Add movie + genre nodes/edges
    kept_movies = []
    movie_labels = {}

    for m_idx in user_movies:
        m_idx = int(m_idx)
        kept_movies.append(m_idx)

        # title
        movieId = movie_idx_to_movieId.get(m_idx, None)
        if movieId is None:
            title = f"movie_idx {m_idx}"
        else:
            row = movies_df[movies_df["movieId"] == movieId]
            title = row["title"].iloc[0] if len(row) else f"movieId {movieId}"

        m_node = f"movie:{m_idx}"
        movie_labels[m_node] = safe_movie_label(title)

        G.add_node(m_node, ntype="movie", label=movie_labels[m_node])
        G.add_edge(user_node, m_node)

        # genres for this movie
        g_rows = edges_mg[edges_mg["movie_idx"] == m_idx]["genre_idx"].values
        g_rows = [int(x) for x in g_rows][: args.max_genres_per_movie]

        for g_idx in g_rows:
            g_name = genre_idx_to_name.get(g_idx, f"genre {g_idx}")
            g_node = f"genre:{g_idx}"
            G.add_node(g_node, ntype="genre", label=g_name)
            G.add_edge(m_node, g_node)

    # Layout + draw
    plt.figure(figsize=(18, 10))
    pos = nx.spring_layout(G, seed=args.seed, k=0.9)

    # node groups
    user_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "user"]
    movie_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "movie"]
    genre_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "genre"]

    nx.draw_networkx_edges(G, pos, alpha=0.35)

    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_shape="s", node_size=1200)
    nx.draw_networkx_nodes(G, pos, nodelist=movie_nodes, node_shape="o", node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=genre_nodes, node_shape="^", node_size=750)

    # labels
    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    title = f"Heterogeneous Graph Sample (Genres Only)\nuser_idx={user_idx} | user → liked movies → genres"
    plt.title(title)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

    # Save small json summary
    summary = {
        "user_idx": user_idx,
        "top_movies": int(args.top_movies),
        "max_genres_per_movie": int(args.max_genres_per_movie),
        "movies_in_plot": [int(x) for x in kept_movies],
        "output_png": str(out_png),
        "notes": "Genres-only visualization. User connected to liked movies; movies connected to genres.",
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"✅ Saved PNG : {out_png}")
    print(f"✅ Saved JSON: {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()