import os
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

RATING_THRESHOLD = 4.0
TAG_MIN_COUNT = 50

# Make it readable:
MAX_MOVIES = 5
MAX_TAGS_PER_MOVIE = 1   # keep very small for clarity


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def clean_tag(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def shorten(text: str, n: int = 28) -> str:
    text = str(text)
    return text if len(text) <= n else text[:n] + "…"


def main():
    ROOT = project_root()

    train_path = os.path.join(ROOT, "data", "processed", "train.csv")
    movies_path = os.path.join(ROOT, "data", "raw", "movies.csv")
    tags_path = os.path.join(ROOT, "data", "raw", "tags.csv")

    out_dir = os.path.join(ROOT, "outputs", "gnn", "graph")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "graph_sample_pretty.png")

    # Load
    train = pd.read_csv(train_path, usecols=["userId", "movieId", "rating"])
    movies = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
    tags = pd.read_csv(tags_path, usecols=["movieId", "tag"])

    train_pos = train[train["rating"] >= RATING_THRESHOLD].copy()

    # Choose a user with at least MAX_MOVIES positives
    user_counts = train_pos["userId"].value_counts()
    candidate_users = user_counts[user_counts >= MAX_MOVIES].index.tolist()
    user_id = random.choice(candidate_users)

    # User movies
    user_movies = (
        train_pos[train_pos["userId"] == user_id]["movieId"]
        .drop_duplicates()
        .head(MAX_MOVIES)
        .tolist()
    )

    movies_lookup = movies.set_index("movieId")
    user_movies = [m for m in user_movies if m in movies_lookup.index]

    # Tags cleaning + filtering
    tags["tag_clean"] = tags["tag"].apply(clean_tag)
    tags = tags[tags["tag_clean"] != ""]
    tag_counts = tags["tag_clean"].value_counts()
    kept_tags = set(tag_counts[tag_counts >= TAG_MIN_COUNT].index.tolist())
    tags = tags[tags["tag_clean"].isin(kept_tags)].copy()

    # Pick top tags per movie
    movie_to_top_tags = {}
    for mid in user_movies:
        subset = tags[tags["movieId"] == mid]["tag_clean"]
        if len(subset) == 0:
            movie_to_top_tags[mid] = []
        else:
            movie_to_top_tags[mid] = subset.value_counts().head(MAX_TAGS_PER_MOVIE).index.tolist()

    # Build graph
    G = nx.Graph()
    user_node = f"User\n{user_id}"
    G.add_node(user_node, ntype="user")

    movie_nodes = []
    genre_nodes = set()
    tag_nodes = set()

    for mid in user_movies:
        title = shorten(movies_lookup.loc[mid, "title"], 32)
        movie_node = f"Movie\n{title}"
        movie_nodes.append(movie_node)
        G.add_node(movie_node, ntype="movie")
        G.add_edge(user_node, movie_node)

        # genres
        genres_str = str(movies_lookup.loc[mid, "genres"])
        for g in genres_str.split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                gn = f"Genre\n{g}"
                genre_nodes.add(gn)
                G.add_node(gn, ntype="genre")
                G.add_edge(movie_node, gn)

        # tags
        for t in movie_to_top_tags.get(mid, []):
            tn = f"Tag\n{t}"
            tag_nodes.add(tn)
            G.add_node(tn, ntype="tag")
            G.add_edge(movie_node, tn)

    genre_nodes = list(genre_nodes)
    tag_nodes = list(tag_nodes)

    # Layout: place user center-ish and spread others
    pos = nx.spring_layout(G, seed=42, k=1.0)

    plt.figure(figsize=(16, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.25)

    # Draw with different shapes (no need to manually choose colors)
    nx.draw_networkx_nodes(G, pos, nodelist=[user_node], node_size=2200, node_shape="s")
    nx.draw_networkx_nodes(G, pos, nodelist=movie_nodes, node_size=1400, node_shape="o")
    nx.draw_networkx_nodes(G, pos, nodelist=genre_nodes, node_size=900, node_shape="^")
    nx.draw_networkx_nodes(G, pos, nodelist=tag_nodes, node_size=900, node_shape="D")

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=9)

    plt.title(
        f"Heterogeneous Graph Sample (userId={user_id})\n"
        f"User → liked movies → genres/tags",
        fontsize=14
    )

    # Simple legend (proxy artists)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='User', markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Movie', markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='^', color='w', label='Genre', markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='D', color='w', label='Tag', markerfacecolor='gray', markersize=12),
    ]
    plt.legend(handles=legend_elements, loc="lower left")

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=250)
    plt.close()

    print("✅ Saved prettier visualization:", out_png)
    print("User:", user_id)
    print("Movies shown:", len(user_movies))
    print("Genres shown:", len(genre_nodes))
    print("Tags shown:", len(tag_nodes))


if __name__ == "__main__":
    main()