import os
import re
import math
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import networkx as nx

# -----------------------------
# Helper: find project root
# -----------------------------
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return start.resolve()


# -----------------------------
# Helper: safe torch.load for HeteroData (PyTorch 2.6+)
# -----------------------------
def safe_load_graph(graph_path: Path):
    graph_path = graph_path.resolve()
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}")

    # Try the safest approach first (allowlist HeteroData)
    try:
        from torch_geometric.data.hetero_data import HeteroData
        torch.serialization.add_safe_globals([HeteroData])
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
        return data
    except Exception:
        # fallback: old behavior
        data = torch.load(graph_path, map_location="cpu", weights_only=False)
        return data


# -----------------------------
# Helper: layered positions for clean plot
# -----------------------------
def layered_positions(user_node, movie_nodes, outer_nodes):
    pos = {}
    pos[user_node] = (0.0, 0.0)

    # Movies circle
    r_movies = 2.3
    for i, n in enumerate(movie_nodes):
        angle = 2 * math.pi * i / max(1, len(movie_nodes))
        pos[n] = (r_movies * math.cos(angle), r_movies * math.sin(angle))

    # Outer nodes circle (genres + tags)
    r_outer = 4.5
    for i, n in enumerate(outer_nodes):
        angle = 2 * math.pi * i / max(1, len(outer_nodes))
        pos[n] = (r_outer * math.cos(angle), r_outer * math.sin(angle))

    return pos


def shorten(text, max_len=28):
    if text is None:
        return ""
    t = str(text)
    if len(t) <= max_len:
        return t
    return t[: max_len - 3] + "..."


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--userId", type=int, default=None, help="UserId (original) to visualize. If not provided, auto-picks.")
    parser.add_argument("--max_movies", type=int, default=5, help="Max liked movies to show for this user")
    parser.add_argument("--max_tags_per_movie", type=int, default=2, help="Max tags to show per movie")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--graph_path", type=str, default=None, help="Path to hetero_graph_train.pt")
    parser.add_argument("--edges_dir", type=str, default=None, help="Path to graph_edges directory containing mappings/edges")
    parser.add_argument("--movies_csv", type=str, default=None, help="Path to raw movies.csv")

    parser.add_argument("--out_png", type=str, default=None, help="Output PNG path")
    parser.add_argument("--out_csv", type=str, default=None, help="Optional: output CSV listing nodes in sample")
    args = parser.parse_args()

    np.random.seed(args.seed)

    project_root = find_project_root(Path(__file__).parent)
    print(f"Project root: {project_root}")

    # Defaults that match your project structure
    default_graph = project_root / "outputs" / "gnn" / "graph" / "hetero_graph_train.pt"
    default_edges_dir = project_root / "data" / "processed" / "graph_edges"
    default_movies_csv = project_root / "data" / "raw" / "movies.csv"
    default_out_png = project_root / "outputs" / "gnn" / "graph" / "graph_sample_pretty.png"

    graph_path = Path(args.graph_path) if args.graph_path else default_graph
    edges_dir = Path(args.edges_dir) if args.edges_dir else default_edges_dir
    movies_csv = Path(args.movies_csv) if args.movies_csv else default_movies_csv
    out_png = Path(args.out_png) if args.out_png else default_out_png
    out_png.parent.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv) if args.out_csv else None
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Files from Stage 6.4
    user_map_path = edges_dir / "user_id_mapping.csv"
    movie_map_path = edges_dir / "movie_id_mapping.csv"
    genre_map_path = edges_dir / "genre_mapping.csv"
    tag_map_path = edges_dir / "tag_mapping.csv"

    edges_user_movie_path = edges_dir / "edges_user_movie_train.csv"
    edges_movie_genre_path = edges_dir / "edges_movie_genre.csv"
    edges_movie_tag_path = edges_dir / "edges_movie_tag.csv"

    print("============================================")
    print("✅ Stage 6.7 — Graph Visualization (Pretty)")
    print("============================================")
    print("Graph path  :", graph_path)
    print("Edges dir   :", edges_dir)
    print("Movies CSV  :", movies_csv)
    print("Out PNG     :", out_png)
    if out_csv:
        print("Out CSV     :", out_csv)
    print("============================================")

    # Load graph (not strictly required for visualization, but confirms it exists)
    data = safe_load_graph(graph_path)
    print("Loaded graph OK. Node types:", list(data.node_types))
    print("Edge types:", [str(e) for e in data.edge_types])

    # Load mappings
    user_map = pd.read_csv(user_map_path)         # userId,user_idx
    movie_map = pd.read_csv(movie_map_path)       # movieId,movie_idx
    genre_map = pd.read_csv(genre_map_path)       # genre,genre_idx
    tag_map = pd.read_csv(tag_map_path)           # tag,tag_idx

    userId_to_idx = dict(zip(user_map["userId"], user_map["user_idx"]))
    idx_to_userId = dict(zip(user_map["user_idx"], user_map["userId"]))

    movieId_to_idx = dict(zip(movie_map["movieId"], movie_map["movie_idx"]))
    idx_to_movieId = dict(zip(movie_map["movie_idx"], movie_map["movieId"]))

    genre_idx_to_name = dict(zip(genre_map["genre_idx"], genre_map["genre"]))
    tag_idx_to_name = dict(zip(tag_map["tag_idx"], tag_map["tag"]))

    # Load movies titles
    movies_df = pd.read_csv(movies_csv)
    movieId_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))

    # Load edges
    um = pd.read_csv(edges_user_movie_path)       # user_idx,movie_idx
    mg = pd.read_csv(edges_movie_genre_path)      # movie_idx,genre_idx
    mt = pd.read_csv(edges_movie_tag_path) if edges_movie_tag_path.exists() else None

    # Pick user
    if args.userId is None:
        # choose a user with enough interactions for a pretty graph
        counts = um.groupby("user_idx")["movie_idx"].count().reset_index(name="cnt")
        counts = counts[counts["cnt"] >= min(3, args.max_movies)]
        if len(counts) == 0:
            # fallback: any user
            user_idx = int(um["user_idx"].sample(1, random_state=args.seed).iloc[0])
        else:
            user_idx = int(counts.sample(1, random_state=args.seed)["user_idx"].iloc[0])
        userId = int(idx_to_userId[user_idx])
    else:
        userId = int(args.userId)
        if userId not in userId_to_idx:
            raise ValueError(f"userId {userId} not found in mapping.")
        user_idx = int(userId_to_idx[userId])

    # Get this user's liked movies (from train edges)
    user_movies = um.loc[um["user_idx"] == user_idx, "movie_idx"].values
    if len(user_movies) == 0:
        raise ValueError(f"userId={userId} has 0 train interactions in edges_user_movie_train.csv")

    # Sample at most max_movies for clean plot
    if len(user_movies) > args.max_movies:
        user_movies = np.random.choice(user_movies, size=args.max_movies, replace=False)

    movie_nodes = []
    genre_nodes = set()
    tag_nodes = set()

    # Build node labels
    for m_idx in user_movies:
        m_idx = int(m_idx)
        mId = int(idx_to_movieId[m_idx])
        title = movieId_to_title.get(mId, f"movieId={mId}")
        movie_nodes.append((m_idx, title))

        # Genres for this movie
        g_idxs = mg.loc[mg["movie_idx"] == m_idx, "genre_idx"].tolist()
        for g in g_idxs:
            genre_nodes.add(int(g))

        # Tags for this movie (optional)
        if mt is not None:
            t_idxs = mt.loc[mt["movie_idx"] == m_idx, "tag_idx"].tolist()
            # limit tags per movie to keep it clean
            if len(t_idxs) > args.max_tags_per_movie:
                t_idxs = t_idxs[: args.max_tags_per_movie]
            for t in t_idxs:
                tag_nodes.add(int(t))

    # Create NetworkX graph
    G = nx.Graph()

    user_node = f"user:{userId}"
    G.add_node(user_node, ntype="user", label=f"user\n{userId}")

    # Add movie nodes + edges user->movie
    movie_node_ids = []
    for (m_idx, title) in movie_nodes:
        node_id = f"movie:{m_idx}"
        movie_node_ids.append(node_id)
        G.add_node(node_id, ntype="movie", label=shorten(title, 30))
        G.add_edge(user_node, node_id)

    # Add genre nodes + edges movie->genre
    genre_node_ids = []
    for g_idx in sorted(list(genre_nodes)):
        gname = genre_idx_to_name.get(g_idx, f"genre_{g_idx}")
        node_id = f"genre:{g_idx}"
        genre_node_ids.append(node_id)
        G.add_node(node_id, ntype="genre", label=f"Genre\n{gname}")

    for (m_idx, _) in movie_nodes:
        m_node = f"movie:{int(m_idx)}"
        g_idxs = mg.loc[mg["movie_idx"] == int(m_idx), "genre_idx"].tolist()
        for g in g_idxs:
            g_node = f"genre:{int(g)}"
            if G.has_node(g_node):
                G.add_edge(m_node, g_node)

    # Add tag nodes + edges movie->tag
    tag_node_ids = []
    if mt is not None and len(tag_nodes) > 0:
        for t_idx in sorted(list(tag_nodes)):
            tname = tag_idx_to_name.get(t_idx, f"tag_{t_idx}")
            node_id = f"tag:{t_idx}"
            tag_node_ids.append(node_id)
            G.add_node(node_id, ntype="tag", label=f"Tag\n{shorten(tname, 18)}")

        for (m_idx, _) in movie_nodes:
            m_node = f"movie:{int(m_idx)}"
            t_idxs = mt.loc[mt["movie_idx"] == int(m_idx), "tag_idx"].tolist()
            if len(t_idxs) > args.max_tags_per_movie:
                t_idxs = t_idxs[: args.max_tags_per_movie]
            for t in t_idxs:
                t_node = f"tag:{int(t)}"
                if G.has_node(t_node):
                    G.add_edge(m_node, t_node)

    # Layered layout positions
    outer_nodes = genre_node_ids + tag_node_ids
    pos = layered_positions(user_node, movie_node_ids, outer_nodes)

    # Draw
    plt.figure(figsize=(16, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.25)

    # Nodes by type
    nx.draw_networkx_nodes(G, pos, nodelist=[user_node], node_size=2600, node_shape="s")
    nx.draw_networkx_nodes(G, pos, nodelist=movie_node_ids, node_size=1600, node_shape="o")
    nx.draw_networkx_nodes(G, pos, nodelist=genre_node_ids, node_size=1100, node_shape="^")
    if len(tag_node_ids) > 0:
        nx.draw_networkx_nodes(G, pos, nodelist=tag_node_ids, node_size=1100, node_shape="D")

    # Labels
    labels = {n: G.nodes[n].get("label", n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.title(
        f"Heterogeneous Graph Sample (userId={userId})\nUser → liked movies → genres/tags",
        fontsize=14
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"\n✅ Saved pretty graph to: {out_png}")

    # Optional: write CSV with node list (helps in report / debugging)
    if out_csv:
        rows = []
        for n in G.nodes():
            rows.append({
                "node_id": n,
                "node_type": G.nodes[n].get("ntype"),
                "label": G.nodes[n].get("label")
            })
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"✅ Saved node list CSV to: {out_csv}")


if __name__ == "__main__":
    main()