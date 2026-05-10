import os
import re
import random
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

RATING_THRESHOLD = 4.0
TAG_MIN_COUNT = 50

# For a readable figure:
MAX_MOVIES = 6          # how many liked movies to show for the user
MAX_TAGS_PER_MOVIE = 2  # keep it small & readable


def project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def clean_tag(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip().lower()
    t = re.sub(r"\s+", " ", t)
    return t


def main():
    ROOT = project_root()

    # Paths
    train_path = os.path.join(ROOT, "data", "processed", "train.csv")
    movies_path = os.path.join(ROOT, "data", "raw", "movies.csv")
    tags_path = os.path.join(ROOT, "data", "raw", "tags.csv")

    out_dir = os.path.join(ROOT, "outputs", "gnn", "graph")
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "graph_sample.png")

    # Load data
    train = pd.read_csv(train_path, usecols=["userId", "movieId", "rating"])
    movies = pd.read_csv(movies_path, usecols=["movieId", "title", "genres"])
    tags = pd.read_csv(tags_path, usecols=["movieId", "tag"])

    # Keep positive train interactions
    train_pos = train[train["rating"] >= RATING_THRESHOLD].copy()

    # Pick a user with enough positives
    user_counts = train_pos["userId"].value_counts()
    candidate_users = user_counts[user_counts >= MAX_MOVIES].index.tolist()
    if not candidate_users:
        raise ValueError("No users found with enough positive interactions to sample.")

    user_id = random.choice(candidate_users)

    # Select user's liked movies (sample)
    user_movies = train_pos[train_pos["userId"] == user_id]["movieId"].drop_duplicates().tolist()
    user_movies = user_movies[:MAX_MOVIES]

    # Build movie lookup
    movies_lookup = movies.set_index("movieId")
    # If any movie missing in movies.csv, drop it
    user_movies = [m for m in user_movies if m in movies_lookup.index]

    # Prepare tag filtering
    tags["tag_clean"] = tags["tag"].apply(clean_tag)
    tags = tags[tags["tag_clean"] != ""]
    tag_counts = tags["tag_clean"].value_counts()
    kept_tags = set(tag_counts[tag_counts >= TAG_MIN_COUNT].index.tolist())
    tags = tags[tags["tag_clean"].isin(kept_tags)].copy()

    # For each movie, get top tags
    # (use frequency within that movie’s tags)
    movie_to_top_tags = {}
    for mid in user_movies:
        subset = tags[tags["movieId"] == mid]["tag_clean"]
        if len(subset) == 0:
            movie_to_top_tags[mid] = []
        else:
            top_tags = subset.value_counts().head(MAX_TAGS_PER_MOVIE).index.tolist()
            movie_to_top_tags[mid] = top_tags

    # Build graph
    G = nx.Graph()

    user_node = f"U:{user_id}"
    G.add_node(user_node, ntype="user")

    for mid in user_movies:
        title = str(movies_lookup.loc[mid, "title"])
        title_short = title if len(title) <= 28 else title[:28] + "…"
        movie_node = f"M:{mid}\n{title_short}"
        G.add_node(movie_node, ntype="movie")
        G.add_edge(user_node, movie_node, etype="rates")

        # Genres
        genres_str = str(movies_lookup.loc[mid, "genres"])
        for g in genres_str.split("|"):
            g = g.strip()
            if g and g != "(no genres listed)":
                genre_node = f"G:{g}"
                G.add_node(genre_node, ntype="genre")
                G.add_edge(movie_node, genre_node, etype="has_genre")

        # Tags (optional)
        for t in movie_to_top_tags.get(mid, []):
            tag_node = f"T:{t}"
            G.add_node(tag_node, ntype="tag")
            G.add_edge(movie_node, tag_node, etype="has_tag")

    # Layout (spring)
    pos = nx.spring_layout(G, seed=42, k=0.8)

    # Draw nodes by type (no custom colors requested, so use default styles)
    # We'll just vary node size for readability.
    user_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "user"]
    movie_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "movie"]
    genre_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "genre"]
    tag_nodes = [n for n, d in G.nodes(data=True) if d.get("ntype") == "tag"]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_edges(G, pos, alpha=0.35)

    nx.draw_networkx_nodes(G, pos, nodelist=user_nodes, node_size=1400)
    nx.draw_networkx_nodes(G, pos, nodelist=movie_nodes, node_size=900)
    nx.draw_networkx_nodes(G, pos, nodelist=genre_nodes, node_size=650)
    nx.draw_networkx_nodes(G, pos, nodelist=tag_nodes, node_size=650)

    nx.draw_networkx_labels(G, pos, font_size=8)

    plt.title(f"Graph Sample: 1 user → liked movies → genres/tags (userId={user_id})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print("✅ Saved graph sample visualization to:")
    print(out_png)
    print(f"User shown: {user_id}")
    print(f"Movies shown: {len(user_movies)}")


if __name__ == "__main__":
    main()