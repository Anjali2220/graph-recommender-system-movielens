import os
import numpy as np
import pandas as pd


def load_model(model_dir: str):
    item_ids = np.load(os.path.join(model_dir, "item_ids.npy"))
    knn_indices = np.load(os.path.join(model_dir, "knn_indices.npy"))
    knn_sims = np.load(os.path.join(model_dir, "knn_sims.npy"))
    movieId_to_itemIndex = {int(m): i for i, m in enumerate(item_ids)}
    return item_ids, movieId_to_itemIndex, knn_indices, knn_sims


def score_user_all_items(user_history: np.ndarray,
                         knn_indices: np.ndarray,
                         knn_sims: np.ndarray,
                         topk_neighbors: int,
                         n_items: int):
    scores = np.zeros(n_items, dtype=np.float32)
    for h in user_history:
        neigh = knn_indices[h, :topk_neighbors]
        sims = knn_sims[h, :topk_neighbors]
        scores[neigh] += sims
    return scores


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model")

    OUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model",
                            "demo_user_likes_vs_recs.csv")

    # Settings
    RATING_THRESHOLD = 4.0
    TOPK_NEIGHBORS = 50
    TOPN_RECS = 10
    RECENT_LIKES = 5
    NUM_USERS_DEMO = 10
    SEED = 42

    print("Loading movies metadata...")
    movies = pd.read_csv(MOVIES_PATH, usecols=["movieId", "title", "genres"])

    print("Loading ItemKNN model...")
    item_ids, movieId_to_itemIndex, knn_indices, knn_sims = load_model(MODEL_DIR)
    n_items = len(item_ids)

    print("Loading train interactions...")
    train = pd.read_csv(TRAIN_PATH, usecols=["userId", "movieId", "rating", "timestamp"])

    # Keep only positives
    train = train[train["rating"] >= RATING_THRESHOLD].copy()

    # Merge title/genre for the "recent likes"
    train = train.merge(movies, on="movieId", how="left")

    # Map movieId -> itemIndex (so we can score)
    train["itemIndex"] = train["movieId"].map(movieId_to_itemIndex)
    train = train.dropna(subset=["itemIndex"]).copy()
    train["itemIndex"] = train["itemIndex"].astype(np.int32)

    # Build user history arrays for scoring
    # (use unique items per user)
    train = train.sort_values("timestamp").drop_duplicates(["userId", "itemIndex"], keep="last")

    history = train.groupby("userId")["itemIndex"].apply(lambda x: x.to_numpy()).to_dict()

    users = list(history.keys())
    rng = np.random.default_rng(SEED)
    demo_users = rng.choice(users, size=NUM_USERS_DEMO, replace=False).tolist()

    print("Demo users:", demo_users)

    rows = []

    for user in demo_users:
        user_hist = history[user]

        # Recent liked movies (most recent)
        recent = train[train["userId"] == user].sort_values("timestamp", ascending=False).head(RECENT_LIKES)

        recent_likes_str = " | ".join(
            [f"{r.title} ({r.genres})" for r in recent.itertuples(index=False)]
        )

        # Get recommendations
        scores = score_user_all_items(user_hist, knn_indices, knn_sims, TOPK_NEIGHBORS, n_items)

        # Remove already seen
        scores[user_hist] = -np.inf

        top_item_indices = np.argsort(-scores)[:TOPN_RECS]
        rec_movie_ids = [int(item_ids[idx]) for idx in top_item_indices]
        rec_scores = [float(scores[idx]) for idx in top_item_indices]

        recs_meta = movies[movies["movieId"].isin(rec_movie_ids)].copy()
        recs_meta["rank"] = recs_meta["movieId"].apply(lambda x: rec_movie_ids.index(int(x)) + 1)

        # Ensure correct order by rank
        recs_meta = recs_meta.sort_values("rank")

        recs_str = " | ".join(
            [f"{r.rank}. {r.title} ({r.genres})" for r in recs_meta.itertuples(index=False)]
        )

        rows.append({
            "userId": int(user),
            "recent_liked_movies_train": recent_likes_str,
            "top10_recommendations_itemknn": recs_str
        })

    demo_df = pd.DataFrame(rows)
    demo_df.to_csv(OUT_PATH, index=False)

    print("\n✅ Saved professor-friendly demo table to:")
    print(OUT_PATH)
    print("\nPreview:")
    print(demo_df.head(10))


if __name__ == "__main__":
    main()
