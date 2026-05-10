import os
import json
import re
import pandas as pd


def project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def clean_tag(t: str) -> str:
    if not isinstance(t, str):
        return ""
    t = t.strip().lower()
    # optional: normalize multiple spaces
    t = re.sub(r"\s+", " ", t)
    return t


def main():
    ROOT = project_root()

    # ===== INPUTS =====
    TRAIN_PATH = os.path.join(ROOT, "data", "processed", "train.csv")
    MOVIES_PATH = os.path.join(ROOT, "data", "raw", "movies.csv")  # MovieLens file with movieId,title,genres
    TAGS_PATH = os.path.join(ROOT, "data", "raw", "tags.csv")      # MovieLens file with userId,movieId,tag,timestamp

    # ===== OUTPUT DIR =====
    OUT_DIR = os.path.join(ROOT, "data", "processed", "graph_edges")
    os.makedirs(OUT_DIR, exist_ok=True)

    # ===== OUTPUT FILES =====
    OUT_USER_MOVIE = os.path.join(OUT_DIR, "edges_user_movie_train.csv")
    OUT_MOVIE_GENRE = os.path.join(OUT_DIR, "edges_movie_genre.csv")
    OUT_MOVIE_TAG = os.path.join(OUT_DIR, "edges_movie_tag.csv")

    OUT_USER_MAP = os.path.join(OUT_DIR, "user_id_mapping.csv")   # userId -> user_idx
    OUT_MOVIE_MAP = os.path.join(OUT_DIR, "movie_id_mapping.csv") # movieId -> movie_idx
    OUT_GENRE_MAP = os.path.join(OUT_DIR, "genre_mapping.csv")    # genre -> genre_idx
    OUT_TAG_MAP = os.path.join(OUT_DIR, "tag_mapping.csv")        # tag_clean -> tag_idx

    META_JSON = os.path.join(OUT_DIR, "stage_6_4_edge_build_metadata.json")

    # ===== SETTINGS =====
    RATING_THRESHOLD = 4.0
    TAG_MIN_COUNT = 50

    print("============================================")
    print("🔷 Stage 6.4 — Build Edge Lists (Leakage-safe)")
    print("============================================")
    print("Train   :", TRAIN_PATH)
    print("Movies  :", MOVIES_PATH)
    print("Tags    :", TAGS_PATH)
    print("Out dir :", OUT_DIR)
    print("--------------------------------------------")
    print("Rating threshold:", RATING_THRESHOLD)
    print("Tag min count   :", TAG_MIN_COUNT)
    print("============================================\n")

    # ------------------------------------------------------------------
    # (A) Train interaction edges from train.csv ONLY (no leakage)
    # ------------------------------------------------------------------
    train = pd.read_csv(TRAIN_PATH)

    # Expect columns at least: userId, movieId, rating
    for col in ["userId", "movieId", "rating"]:
        if col not in train.columns:
            raise ValueError(f"train.csv missing required column: {col}")

    train_pos = train[train["rating"] >= RATING_THRESHOLD][["userId", "movieId"]].drop_duplicates()

    # Build mappings from TRAIN positives (best practice for train-graph)
    user_ids = train_pos["userId"].unique()
    movie_ids = train_pos["movieId"].unique()

    user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    movie_id_to_idx = {mid: i for i, mid in enumerate(movie_ids)}

    # Save mapping CSVs
    pd.DataFrame({"userId": user_ids, "user_idx": [user_id_to_idx[u] for u in user_ids]}).to_csv(OUT_USER_MAP, index=False)
    pd.DataFrame({"movieId": movie_ids, "movie_idx": [movie_id_to_idx[m] for m in movie_ids]}).to_csv(OUT_MOVIE_MAP, index=False)

    # Map train edges
    edges_user_movie = pd.DataFrame({
        "user_idx": train_pos["userId"].map(user_id_to_idx).astype("int64"),
        "movie_idx": train_pos["movieId"].map(movie_id_to_idx).astype("int64"),
    })
    edges_user_movie.to_csv(OUT_USER_MOVIE, index=False)

    print("✅ (A) Train user–movie edges saved:", OUT_USER_MOVIE)
    print("   Positives (unique edges):", len(edges_user_movie))
    print("   Users:", len(user_ids), "| Movies:", len(movie_ids), "\n")

    # ------------------------------------------------------------------
    # (B) Movie–Genre edges from movies.csv (metadata, safe)
    # ------------------------------------------------------------------
    movies = pd.read_csv(MOVIES_PATH)
    for col in ["movieId", "genres"]:
        if col not in movies.columns:
            raise ValueError(f"movies.csv missing required column: {col}")

    # Keep only movies that exist in the TRAIN mapping (important alignment)
    movies = movies[movies["movieId"].isin(movie_id_to_idx.keys())].copy()

    # Build genre vocabulary
    all_genres = set()
    movie_genre_pairs = []

    for _, row in movies.iterrows():
        mid = row["movieId"]
        gstr = row["genres"]
        if not isinstance(gstr, str):
            continue
        parts = gstr.split("|")
        for g in parts:
            g = g.strip()
            if g == "":
                continue
            all_genres.add(g)

    genre_list = sorted(list(all_genres))
    genre_to_idx = {g: i for i, g in enumerate(genre_list)}

    # Save genre mapping
    pd.DataFrame({"genre": genre_list, "genre_idx": [genre_to_idx[g] for g in genre_list]}).to_csv(OUT_GENRE_MAP, index=False)

    # Build edges
    for _, row in movies.iterrows():
        mid = row["movieId"]
        gstr = row["genres"]
        if not isinstance(gstr, str):
            continue
        m_idx = movie_id_to_idx[mid]
        for g in gstr.split("|"):
            g = g.strip()
            if g == "":
                continue
            movie_genre_pairs.append((m_idx, genre_to_idx[g]))

    edges_movie_genre = pd.DataFrame(movie_genre_pairs, columns=["movie_idx", "genre_idx"]).drop_duplicates()
    edges_movie_genre.to_csv(OUT_MOVIE_GENRE, index=False)

    print("✅ (B) Movie–genre edges saved:", OUT_MOVIE_GENRE)
    print("   Genres:", len(genre_list), "| Edges:", len(edges_movie_genre), "\n")

    # ------------------------------------------------------------------
    # (C) Movie–Tag edges from tags.csv (metadata, safe) with cleaning + filtering
    # ------------------------------------------------------------------
    tags = pd.read_csv(TAGS_PATH)
    for col in ["movieId", "tag"]:
        if col not in tags.columns:
            raise ValueError(f"tags.csv missing required column: {col}")

    # Keep only movies that exist in our TRAIN movie mapping
    tags = tags[tags["movieId"].isin(movie_id_to_idx.keys())].copy()
    tags["tag_clean"] = tags["tag"].apply(clean_tag)
    tags = tags[tags["tag_clean"] != ""]

    # Count tags and filter rare ones
    tag_counts = tags["tag_clean"].value_counts()
    kept_tags = tag_counts[tag_counts >= TAG_MIN_COUNT].index.tolist()

    tags = tags[tags["tag_clean"].isin(kept_tags)].copy()

    tag_list = sorted(tags["tag_clean"].unique().tolist())
    tag_to_idx = {t: i for i, t in enumerate(tag_list)}

    # Save tag mapping
    pd.DataFrame({"tag": tag_list, "tag_idx": [tag_to_idx[t] for t in tag_list]}).to_csv(OUT_TAG_MAP, index=False)

    # Build edges (movie_idx, tag_idx)
    edges_movie_tag = pd.DataFrame({
        "movie_idx": tags["movieId"].map(movie_id_to_idx).astype("int64"),
        "tag_idx": tags["tag_clean"].map(tag_to_idx).astype("int64"),
    }).drop_duplicates()

    edges_movie_tag.to_csv(OUT_MOVIE_TAG, index=False)

    print("✅ (C) Movie–tag edges saved:", OUT_MOVIE_TAG)
    print("   Kept tags:", len(tag_list), "| Edges:", len(edges_movie_tag), "\n")

    # ------------------------------------------------------------------
    # Save metadata summary
    # ------------------------------------------------------------------
    meta = {
        "rating_threshold": RATING_THRESHOLD,
        "tag_min_count": TAG_MIN_COUNT,
        "node_counts": {
            "user": int(len(user_ids)),
            "movie": int(len(movie_ids)),
            "genre": int(len(genre_list)),
            "tag": int(len(tag_list)),
        },
        "edge_counts": {
            "user__rates__movie": int(len(edges_user_movie)),
            "movie__has_genre__genre": int(len(edges_movie_genre)),
            "movie__has_tag__tag": int(len(edges_movie_tag)),
        },
        "outputs": {
            "edges_user_movie_train": os.path.basename(OUT_USER_MOVIE),
            "edges_movie_genre": os.path.basename(OUT_MOVIE_GENRE),
            "edges_movie_tag": os.path.basename(OUT_MOVIE_TAG),
            "user_map": os.path.basename(OUT_USER_MAP),
            "movie_map": os.path.basename(OUT_MOVIE_MAP),
            "genre_map": os.path.basename(OUT_GENRE_MAP),
            "tag_map": os.path.basename(OUT_TAG_MAP),
        },
        "notes": "User–movie edges are built from train.csv only (no leakage). Genre/tag edges are metadata; tag vocabulary is cleaned + filtered.",
    }

    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=4)

    print("✅ Metadata saved:", META_JSON)
    print("\nDone ✅")


if __name__ == "__main__":
    main()