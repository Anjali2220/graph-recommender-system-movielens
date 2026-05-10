import os
import json
import numpy as np
import pandas as pd


# ============================================
# SETTINGS
# ============================================
RATING_THRESHOLD = 4.0
# ============================================


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv")

    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Cpopularity_model")
    os.makedirs(OUT_DIR, exist_ok=True)

    COUNTS_CSV = os.path.join(OUT_DIR, "popularity_counts_with_titles.csv")
    RANKING_NPY = os.path.join(OUT_DIR, "popularity_ranking.npy")
    INFO_JSON = os.path.join(OUT_DIR, "popularity_model_info.json")

    print("============================================")
    print("📊 Building Popularity Baseline")
    print("============================================")
    print("Train file :", TRAIN_PATH)
    print("Threshold  :", RATING_THRESHOLD)
    print("Output dir :", OUT_DIR)
    print("============================================\n")

    # --------------------------------------------
    # Load train data
    # --------------------------------------------
    df = pd.read_csv(TRAIN_PATH)
    movies = pd.read_csv(MOVIES_PATH)

    print("Total train rows:", len(df))

    # --------------------------------------------
    # Convert to implicit positives
    # --------------------------------------------
    df_pos = df[df["rating"] >= RATING_THRESHOLD]
    print("Positive interactions (rating ≥ threshold):", len(df_pos))

    # --------------------------------------------
    # Count movie popularity
    # --------------------------------------------
    movie_counts = (
        df_pos
        .groupby("movieId")
        .size()
        .reset_index(name="like_count")
        .sort_values("like_count", ascending=False)
        .reset_index(drop=True)
    )

    # --------------------------------------------
    # Merge with movie metadata
    # --------------------------------------------
    movie_counts = movie_counts.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId",
        how="left"
    )

    # Reorder columns nicely
    movie_counts = movie_counts[
        ["movieId", "title", "genres", "like_count"]
    ]

    print("\nTop 5 most popular movies:")
    print(movie_counts.head())

    # --------------------------------------------
    # Save ranking (ordered movieIds only)
    # --------------------------------------------
    popularity_ranking = movie_counts["movieId"].values
    np.save(RANKING_NPY, popularity_ranking)

    # Save full CSV with titles
    movie_counts.to_csv(COUNTS_CSV, index=False)

    # Save model info
    info = {
        "model": "Popularity",
        "rating_threshold": RATING_THRESHOLD,
        "total_train_rows": int(len(df)),
        "positive_interactions": int(len(df_pos)),
        "unique_movies_ranked": int(len(popularity_ranking))
    }

    with open(INFO_JSON, "w") as f:
        json.dump(info, f, indent=4)

    print("\n✅ Popularity baseline built successfully.")
    print("Files saved:")
    print(" -", COUNTS_CSV)
    print(" -", RANKING_NPY)
    print(" -", INFO_JSON)


if __name__ == "__main__":
    main()
