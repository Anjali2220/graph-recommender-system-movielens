import os
import json
import numpy as np
import pandas as pd

# =============================
# Stage 6.1 — Define Node Types
# =============================

def main():

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Paths
    MF_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")
    USER_IDS_PATH = os.path.join(MF_DIR, "user_ids.npy")
    ITEM_IDS_PATH = os.path.join(MF_DIR, "item_ids.npy")

    MOVIES_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv")
    TAGS_CSV = os.path.join(PROJECT_ROOT, "data", "raw", "tags.csv")

    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "gnn", "graph")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_JSON = os.path.join(OUT_DIR, "node_type_metadata.json")

    print("Project root:", PROJECT_ROOT)

    # --------------------------
    # 1️⃣ Users
    # --------------------------
    user_ids = np.load(USER_IDS_PATH)
    n_users = len(user_ids)

    # --------------------------
    # 2️⃣ Movies
    # --------------------------
    item_ids = np.load(ITEM_IDS_PATH)
    n_movies = len(item_ids)

    # --------------------------
    # 3️⃣ Genres
    # --------------------------
    movies = pd.read_csv(MOVIES_CSV)

    all_genres = set()
    for g in movies["genres"].dropna():
        for genre in g.split("|"):
            if genre != "(no genres listed)":
                all_genres.add(genre.strip())

    genre_list = sorted(list(all_genres))
    n_genres = len(genre_list)

    # --------------------------
    # 4️⃣ Tags (optional but recommended)
    # --------------------------
    tags = pd.read_csv(TAGS_CSV)

    # Basic cleaning
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()

    # Remove very rare tags (optional but recommended)
    tag_counts = tags["tag"].value_counts()
    filtered_tags = tag_counts[tag_counts >= 50].index.tolist()

    n_tags = len(filtered_tags)

    # --------------------------
    # Save Metadata
    # --------------------------
    metadata = {
        "node_types": {
            "user": n_users,
            "movie": n_movies,
            "genre": n_genres,
            "tag": n_tags
        },
        "description": "Node counts for heterogeneous graph construction."
    }

    with open(OUT_JSON, "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✅ Node Types Defined Successfully")
    print("Users  :", n_users)
    print("Movies :", n_movies)
    print("Genres :", n_genres)
    print("Tags   :", n_tags)

    print("\nMetadata saved to:")
    print(OUT_JSON)


if __name__ == "__main__":
    main()