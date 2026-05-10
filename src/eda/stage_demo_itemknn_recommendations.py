import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# Utils
# =========================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def find_first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def load_movies(project_root: Path) -> pd.DataFrame:
    candidates = [
        project_root / "data" / "raw" / "movies.csv",
        project_root / "data" / "raw" / "ml-20m" / "movies.csv",
    ]
    path = find_first_existing(candidates)
    if path is None:
        raise FileNotFoundError("movies.csv not found")
    return pd.read_csv(path)[["movieId", "title", "genres"]]


def load_tags(project_root: Path) -> pd.DataFrame:
    candidates = [
        project_root / "data" / "raw" / "tags.csv",
        project_root / "data" / "raw" / "ml-20m" / "tags.csv",
    ]
    path = find_first_existing(candidates)
    if path is None:
        return pd.DataFrame(columns=["movieId", "tags"])

    df = pd.read_csv(path)
    if not {"movieId", "tag"}.issubset(df.columns):
        return pd.DataFrame(columns=["movieId", "tags"])

    df["tag"] = df["tag"].astype(str).str.strip()
    df = df[df["tag"] != ""].copy()

    tag_df = (
        df.groupby("movieId")["tag"]
        .apply(lambda x: ", ".join(x.value_counts().head(5).index.tolist()))
        .reset_index()
    )
    tag_df.columns = ["movieId", "tags"]
    return tag_df


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_ids", type=int, nargs="+", default=[3054, 23349, 28364, 87505, 116229])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--rating_threshold", type=float, default=4.0)

    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve().parent)

    train_csv = project_root / "data" / "processed" / "train.csv"
    out_dir = project_root / "outputs" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "itemknn_user_recommendations.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv: {train_csv}")

    print("Loading data...")
    df = pd.read_csv(train_csv)
    df = df[df["rating"] >= args.rating_threshold].copy()

    movies = load_movies(project_root)
    tags = load_tags(project_root)

    # build user-item matrix
    unique_users = sorted(df["userId"].unique())
    unique_movies = sorted(df["movieId"].unique())

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}

    rows = df["userId"].map(user_to_idx).values
    cols = df["movieId"].map(movie_to_idx).values
    vals = np.ones(len(df), dtype=np.float32)

    X = csr_matrix((vals, (rows, cols)), shape=(len(unique_users), len(unique_movies)))

    print("Computing item-item cosine similarity...")
    item_sim = cosine_similarity(X.T, dense_output=False)

    results = []

    for user_id in args.user_ids:
        if user_id not in user_to_idx:
            print(f"userId {user_id} not found, skipping.")
            continue

        uidx = user_to_idx[user_id]
        user_profile = X[uidx]

        scores = user_profile @ item_sim
        scores = np.asarray(scores.todense()).ravel()

        seen_idx = user_profile.indices
        scores[seen_idx] = -1e9

        top_idx = np.argsort(-scores)[: args.top_k]

        seen_movie_ids = [idx_to_movie[i] for i in seen_idx[:5]]
        watched_title = "a previously liked movie"
        if len(seen_movie_ids) > 0:
            sample_row = movies[movies["movieId"] == seen_movie_ids[0]]
            if len(sample_row) > 0:
                watched_title = sample_row.iloc[0]["title"]

        print("\n----------------------------------")
        print(f"User {user_id}")
        print(f"Because you watched: {watched_title}")
        print("We recommend:")

        for rank, midx in enumerate(top_idx, start=1):
            movie_id = idx_to_movie[midx]

            movie_row = movies[movies["movieId"] == movie_id]
            title = movie_row.iloc[0]["title"] if len(movie_row) > 0 else ""
            genres = movie_row.iloc[0]["genres"] if len(movie_row) > 0 else ""

            tag_row = tags[tags["movieId"] == movie_id]
            tag_text = tag_row.iloc[0]["tags"] if len(tag_row) > 0 else ""

            print(f"{rank}. {title}")

            results.append(
                {
                    "userId": user_id,
                    "rank": rank,
                    "movieId": movie_id,
                    "title": title,
                    "genres": genres,
                    "tags": tag_text,
                    "score": float(scores[midx]),
                }
            )

    out_df = pd.DataFrame(results)
    out_df.to_csv(out_csv, index=False)

    print("\nSaved to:")
    print(out_csv)


if __name__ == "__main__":
    main()