import os
import json
import numpy as np
import pandas as pd


# ==========================
# SETTINGS
# ==========================
RATING_THRESHOLD = 4.0
TAG_MIN_COUNT = 50  # must match Stage 6.1
# ==========================


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Reuse the SAME user/movie mapping as MF (critical for consistency)
    MF_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")
    USER_IDS_PATH = os.path.join(MF_DIR, "user_ids.npy")   # original userId list
    ITEM_IDS_PATH = os.path.join(MF_DIR, "item_ids.npy")   # original movieId list

    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv")
    TAGS_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "tags.csv")

    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "gnn", "graph")
    ensure_dir(OUT_DIR)

    # Output edge files
    UM_EDGE = os.path.join(OUT_DIR, "edges_user_movie_train.csv")
    MU_EDGE = os.path.join(OUT_DIR, "edges_movie_user_train_rev.csv")

    MG_EDGE = os.path.join(OUT_DIR, "edges_movie_genre.csv")
    GM_EDGE = os.path.join(OUT_DIR, "edges_genre_movie_rev.csv")

    MT_EDGE = os.path.join(OUT_DIR, "edges_movie_tag.csv")
    TM_EDGE = os.path.join(OUT_DIR, "edges_tag_movie_rev.csv")

    META_JSON = os.path.join(OUT_DIR, "edge_type_metadata.json")

    print("Project root:", PROJECT_ROOT)
    print("Output dir  :", OUT_DIR)

    # --------------------------
    # Load mappings
    # --------------------------
    user_ids = np.load(USER_IDS_PATH)
    item_ids = np.load(ITEM_IDS_PATH)

    n_users = len(user_ids)
    n_movies = len(item_ids)

    user2idx = {int(u): i for i, u in enumerate(user_ids)}
    movie2idx = {int(m): i for i, m in enumerate(item_ids)}

    print(f"Users : {n_users}")
    print(f"Movies: {n_movies}")

    # ============================================================
    # 1) user -> movie edges (TRAIN ONLY, rating >= threshold)
    # ============================================================
    train = pd.read_csv(TRAIN_PATH, usecols=["userId", "movieId", "rating"])
    train = train[train["rating"] >= RATING_THRESHOLD].copy()

    train["user_idx"] = train["userId"].map(user2idx)
    train["movie_idx"] = train["movieId"].map(movie2idx)
    train = train.dropna(subset=["user_idx", "movie_idx"]).copy()

    train["user_idx"] = train["user_idx"].astype(np.int32)
    train["movie_idx"] = train["movie_idx"].astype(np.int32)

    # remove duplicates (multiple ratings same user-movie in ML20M can exist)
    train = train.drop_duplicates(subset=["user_idx", "movie_idx"], keep="last")

    edges_um = train[["user_idx", "movie_idx"]].copy()
    edges_um.to_csv(UM_EDGE, index=False)

    # reverse: movie -> user
    edges_mu = edges_um.rename(columns={"user_idx": "dst_user_idx", "movie_idx": "src_movie_idx"})
    edges_mu = edges_mu[["src_movie_idx", "dst_user_idx"]]
    edges_mu.to_csv(MU_EDGE, index=False)

    print("\n✅ Built user↔movie edges (train only)")
    print("user->movie edges:", len(edges_um))

    # ============================================================
    # 2) movie -> genre edges (from movies.csv parsing)
    # ============================================================
    movies = pd.read_csv(MOVIES_PATH, usecols=["movieId", "genres"])
    movies["movie_idx"] = movies["movieId"].map(movie2idx)
    movies = movies.dropna(subset=["movie_idx"]).copy()
    movies["movie_idx"] = movies["movie_idx"].astype(np.int32)

    # build genre vocabulary
    genre_set = set()
    for g in movies["genres"].fillna(""):
        for ge in str(g).split("|"):
            ge = ge.strip()
            if not ge or ge == "(no genres listed)":
                continue
            genre_set.add(ge)

    genre_list = sorted(list(genre_set))
    genre2idx = {g: i for i, g in enumerate(genre_list)}
    n_genres = len(genre_list)

    mg_rows = []
    for row in movies.itertuples(index=False):
        m_idx = int(row.movie_idx)
        g_str = str(row.genres)
        for ge in g_str.split("|"):
            ge = ge.strip()
            if not ge or ge == "(no genres listed)":
                continue
            mg_rows.append((m_idx, genre2idx[ge]))

    edges_mg = pd.DataFrame(mg_rows, columns=["movie_idx", "genre_idx"]).drop_duplicates()
    edges_mg.to_csv(MG_EDGE, index=False)

    # reverse: genre -> movie
    edges_gm = edges_mg.rename(columns={"movie_idx": "dst_movie_idx", "genre_idx": "src_genre_idx"})
    edges_gm = edges_gm[["src_genre_idx", "dst_movie_idx"]]
    edges_gm.to_csv(GM_EDGE, index=False)

    print("\n✅ Built movie↔genre edges")
    print("Genres:", n_genres)
    print("movie->genre edges:", len(edges_mg))

    # ============================================================
    # 3) movie -> tag edges (from tags.csv, filtered)
    # ============================================================
    tags = pd.read_csv(TAGS_PATH, usecols=["movieId", "tag"])
    tags["tag"] = tags["tag"].astype(str).str.lower().str.strip()

    # filter rare tags
    tag_counts = tags["tag"].value_counts()
    kept_tags = tag_counts[tag_counts >= TAG_MIN_COUNT].index.tolist()
    kept_tag_set = set(kept_tags)

    tags = tags[tags["tag"].isin(kept_tag_set)].copy()
    tags["movie_idx"] = tags["movieId"].map(movie2idx)
    tags = tags.dropna(subset=["movie_idx"]).copy()
    tags["movie_idx"] = tags["movie_idx"].astype(np.int32)

    tag_list = sorted(list(kept_tag_set))
    tag2idx = {t: i for i, t in enumerate(tag_list)}
    n_tags = len(tag_list)

    tags["tag_idx"] = tags["tag"].map(tag2idx)
    tags = tags.dropna(subset=["tag_idx"]).copy()
    tags["tag_idx"] = tags["tag_idx"].astype(np.int32)

    edges_mt = tags[["movie_idx", "tag_idx"]].drop_duplicates()
    edges_mt.to_csv(MT_EDGE, index=False)

    # reverse: tag -> movie
    edges_tm = edges_mt.rename(columns={"movie_idx": "dst_movie_idx", "tag_idx": "src_tag_idx"})
    edges_tm = edges_tm[["src_tag_idx", "dst_movie_idx"]]
    edges_tm.to_csv(TM_EDGE, index=False)

    print("\n✅ Built movie↔tag edges")
    print("Tags kept (min_count={}): {}".format(TAG_MIN_COUNT, n_tags))
    print("movie->tag edges:", len(edges_mt))

    # ============================================================
    # Save metadata
    # ============================================================
    meta = {
        "rating_threshold": RATING_THRESHOLD,
        "tag_min_count": TAG_MIN_COUNT,
        "node_counts": {
            "user": n_users,
            "movie": n_movies,
            "genre": n_genres,
            "tag": n_tags
        },
        "edge_counts": {
            "user__rates__movie": int(len(edges_um)),
            "movie__rev_rates__user": int(len(edges_mu)),
            "movie__has_genre__genre": int(len(edges_mg)),
            "genre__rev_has_genre__movie": int(len(edges_gm)),
            "movie__has_tag__tag": int(len(edges_mt)),
            "tag__rev_has_tag__movie": int(len(edges_tm)),
        },
        "edge_files": {
            "user_movie_train": os.path.basename(UM_EDGE),
            "movie_user_train_rev": os.path.basename(MU_EDGE),
            "movie_genre": os.path.basename(MG_EDGE),
            "genre_movie_rev": os.path.basename(GM_EDGE),
            "movie_tag": os.path.basename(MT_EDGE),
            "tag_movie_rev": os.path.basename(TM_EDGE),
        },
        "notes": "User–movie edges are from train.csv only (no leakage). Genre/tag edges are metadata and included globally."
    }

    with open(META_JSON, "w") as f:
        json.dump(meta, f, indent=4)

    print("\n✅ Saved edge metadata:")
    print(META_JSON)

    print("\nAll edge files saved in:")
    print(OUT_DIR)


if __name__ == "__main__":
    main()