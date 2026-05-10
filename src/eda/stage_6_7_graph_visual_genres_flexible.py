import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx


# =====================================================
# 🔹 CHANGE ONLY THIS LINE WHEN YOU WANT ANOTHER USER
# =====================================================
USER_ID = 1943      # <-- Change this anytime (MovieLens original userId)
TOP_MOVIES = 5
MAX_GENRES_PER_MOVIE = 3
# =====================================================


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(8):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def shorten(text, max_len=25):
    text = str(text)
    return text if len(text) <= max_len else text[:max_len - 3] + "..."


def main():
    print("============================================")
    print("✅ Flexible Graph Visualization")
    print("============================================")

    project_root = find_project_root(Path(__file__).parent)

    edges_dir = project_root / "data" / "processed" / "graph_edges"
    out_dir = project_root / "outputs" / "gnn" / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Files
    user_map = pd.read_csv(edges_dir / "user_id_mapping.csv")
    movie_map = pd.read_csv(edges_dir / "movie_id_mapping.csv")
    genre_map = pd.read_csv(edges_dir / "genre_mapping.csv")

    edges_um = pd.read_csv(edges_dir / "edges_user_movie_train.csv")
    edges_mg = pd.read_csv(edges_dir / "edges_movie_genre.csv")

    movies_df = pd.read_csv(project_root / "data" / "raw" / "movies.csv")

    # Map userId → user_idx
    row = user_map[user_map["userId"] == USER_ID]
    if len(row) == 0:
        raise ValueError(f"userId={USER_ID} not found.")
    user_idx = int(row["user_idx"].iloc[0])

    print(f"Using userId={USER_ID} → user_idx={user_idx}")

    # Build lookups
    movie_idx_to_movieId = dict(zip(movie_map["movie_idx"], movie_map["movieId"]))
    movieId_to_title = dict(zip(movies_df["movieId"], movies_df["title"]))
    genre_idx_to_name = dict(zip(genre_map["genre_idx"], genre_map["genre"]))

    # Get liked movies
    user_movies = edges_um[edges_um["user_idx"] == user_idx]["movie_idx"].tolist()

    if len(user_movies) == 0:
        raise ValueError("This user has no positive train interactions.")

    user_movies = user_movies[:TOP_MOVIES]

    # Build graph
    G = nx.Graph()

    user_node = f"user:{USER_ID}"
    G.add_node(user_node, ntype="user", label=f"user\n{USER_ID}")

    for m_idx in user_movies:
        m_idx = int(m_idx)
        movieId = movie_idx_to_movieId[m_idx]
        title = shorten(movieId_to_title[movieId])

        movie_node = f"movie:{m_idx}"
        G.add_node(movie_node, ntype="movie", label=title)
        G.add_edge(user_node, movie_node)

        # Genres
        genres = edges_mg[edges_mg["movie_idx"] == m_idx]["genre_idx"].tolist()
        genres = genres[:MAX_GENRES_PER_MOVIE]

        for g_idx in genres:
            g_name = genre_idx_to_name[g_idx]
            genre_node = f"genre:{g_idx}"
            G.add_node(genre_node, ntype="genre", label=g_name)
            G.add_edge(movie_node, genre_node)

    # Draw
    plt.figure(figsize=(16, 9))
    pos = nx.spring_layout(G, seed=42, k=0.9)

    nx.draw_networkx_edges(G, pos, alpha=0.3)

    user_nodes = [n for n, d in G.nodes(data=True) if d["ntype"] == "user"]
    movie_nodes = [n for n, d in G.nodes(data=True) if d["ntype"] == "movie"]
    genre_nodes = [n for n, d in G.nodes(data=True) if d["ntype"] == "genre"]

    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_shape="s", node_size=1500)
    nx.draw_networkx_nodes(G, pos, nodelist=movie_nodes, node_shape="o", node_size=1000)
    nx.draw_networkx_nodes(G, pos, nodelist=genre_nodes, node_shape="^", node_size=800)

    labels = {n: d["label"] for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.title(
        f"Heterogeneous Graph Sample (Genres Only)\nuserId={USER_ID} → liked movies → genres",
        fontsize=14
    )

    plt.axis("off")
    plt.tight_layout()

    output_path = out_dir / f"graph_user_{USER_ID}.png"
    plt.savefig(output_path, dpi=250)
    plt.close()

    print(f"✅ Saved to: {output_path}")


if __name__ == "__main__":
    main()