import os
import re
import json
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# ---------------------------
# Helpers
# ---------------------------
def find_project_root(start: Path) -> Path:
    """Find project root by locating folders like data/ or src/."""
    cur = start.resolve()
    for _ in range(8):
        if (cur / "src").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def make_movie_title_lookup(movies_csv: Path) -> dict:
    """movieId -> title (if movies.csv exists)"""
    if not movies_csv.exists():
        return {}
    df = pd.read_csv(movies_csv)
    return dict(zip(df["movieId"].astype(int), df["title"].astype(str)))


def make_movieidx_to_movieid(movie_map_csv: Path) -> dict:
    """movie_idx -> movieId (if mapping exists)"""
    if not movie_map_csv.exists():
        return {}
    df = pd.read_csv(movie_map_csv)
    return dict(zip(df["movie_idx"].astype(int), df["movieId"].astype(int)))


def load_genre_name_lookup(genre_mapping_csv: Path) -> dict:
    """genre_idx -> genre_name"""
    df = safe_read_csv(genre_mapping_csv)
    # expected columns: genre, genre_idx
    return dict(zip(df["genre_idx"].astype(int), df["genre"].astype(str)))


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user_idx", type=int, default=None, help="Optional: visualize specific user_idx")
    parser.add_argument("--top_movies", type=int, default=5, help="How many liked movies to show for user")
    parser.add_argument("--max_genres_per_movie", type=int, default=3, help="Limit genres per movie for readability")
    parser.add_argument("--out", type=str, default="graph_sample_genres_only.png")
    args = parser.parse_args()

    # Print which file is running (helps if PyCharm runs wrong file)
    print("RUNNING FILE:", os.path.abspath(__file__))

    project_root = find_project_root(Path(__file__).parent)
    print("============================================")
    print("✅ Stage 6.7 — Graph Visualization (Genres Only)")
    print("============================================")
    print("Project root:", project_root)

    # ---- Expected locations (adjusted to match your earlier outputs) ----
    # Your edges were saved in: data/processed/graph_edges/
    graph_edges_dir = project_root / "data" / "processed" / "graph_edges"

    edges_user_movie = graph_edges_dir / "edges_user_movie_train.csv"
    edges_movie_genre = graph_edges_dir / "edges_movie_genre.csv"
    genre_mapping = graph_edges_dir / "genre_mapping.csv"

    movie_map_csv = graph_edges_dir / "movie_id_mapping.csv"  # optional
    movies_csv = project_root / "data" / "raw" / "movies.csv"  # optional

    # ---- Load edge lists ----
    um = safe_read_csv(edges_user_movie)   # columns: user_idx, movie_idx
    mg = safe_read_csv(edges_movie_genre)  # columns: movie_idx, genre_idx

    # ---- Lookups ----
    genre_idx_to_name = load_genre_name_lookup(genre_mapping)
    movieidx_to_movieid = make_movieidx_to_movieid(movie_map_csv)
    movieid_to_title = make_movie_title_lookup(movies_csv)

    # ---- Choose user ----
    # compute how many movies per user for sampling
    counts = um.groupby("user_idx").size().reset_index(name="n_movies")
    # prefer users with at least 3 liked movies so visualization is meaningful
    counts = counts[counts["n_movies"] >= 3].copy()
    if counts.empty:
        raise RuntimeError("No users with >=3 liked movies found in edges_user_movie_train.csv")

    if args.user_idx is None:
        # ✅ FIXED LINE: iloc[0] so it's a scalar not a Series
        user_idx = int(counts.sample(1, random_state=args.seed)["user_idx"].iloc[0])
    else:
        user_idx = int(args.user_idx)

    print(f"Selected user_idx = {user_idx}")

    # ---- Get user's liked movies ----
    user_movies = um[um["user_idx"] == user_idx]["movie_idx"].astype(int).tolist()
    if len(user_movies) == 0:
        raise RuntimeError(f"user_idx={user_idx} has no movies in edges_user_movie_train.csv")

    # limit movies shown
    user_movies = user_movies[: args.top_movies]

    # ---- Build subgraph ----
    G = nx.Graph()

    user_node = f"U{user_idx}"
    G.add_node(user_node, ntype="user", label=f"user\n{user_idx}")

    # For each movie, add movie node and connect to user
    for m_idx in user_movies:
        # label movie with title if possible
        movie_label = f"movie\nidx={m_idx}"

        if m_idx in movieidx_to_movieid:
            mid = movieidx_to_movieid[m_idx]
            if mid in movieid_to_title:
                title = movieid_to_title[mid]
                # shorten long titles for plotting
                short_title = title if len(title) <= 22 else title[:19] + "..."
                movie_label = f"{short_title}"

        movie_node = f"M{m_idx}"
        G.add_node(movie_node, ntype="movie", label=movie_label)
        G.add_edge(user_node, movie_node)

        # attach genres
        genres_for_movie = mg[mg["movie_idx"] == m_idx]["genre_idx"].astype(int).tolist()
        # limit genres for readability
        genres_for_movie = genres_for_movie[: args.max_genres_per_movie]

        for g_idx in genres_for_movie:
            g_name = genre_idx_to_name.get(g_idx, f"genre_{g_idx}")
            genre_node = f"G{g_idx}"
            if not G.has_node(genre_node):
                G.add_node(genre_node, ntype="genre", label=g_name)
            G.add_edge(movie_node, genre_node)

    # ---- Draw ----
    # layout
    pos = nx.spring_layout(G, seed=args.seed, k=0.9)

    plt.figure(figsize=(14, 8))
    plt.title(f"Heterogeneous Graph Sample (Genres Only)\nuser_idx={user_idx}  |  user → liked movies → genres",
              fontsize=12)

    # draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.35)

    # nodes by type
    user_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "user"]
    movie_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "movie"]
    genre_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "genre"]

    # Draw nodes (different shapes)
    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_shape="s", node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=movie_nodes, node_shape="o", node_size=700)
    nx.draw_networkx_nodes(G, pos, nodelist=genre_nodes, node_shape="^", node_size=550)

    # labels
    labels = {n: d.get("label", n) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8)

    # legend
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    user_patch = mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10, label='User')
    movie_patch = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Movie')
    genre_patch = mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='Genre')
    plt.legend(handles=[user_patch, movie_patch, genre_patch], loc="lower left")

    plt.axis("off")

    out_path = (project_root / "outputs" / "gnn" / "graph" / args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✅ Saved visualization to: {out_path}")


if __name__ == "__main__":
    main()