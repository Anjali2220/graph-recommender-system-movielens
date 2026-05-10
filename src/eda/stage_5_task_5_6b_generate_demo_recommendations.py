import os
import numpy as np
import pandas as pd
import json


# ----------------------------
# Load trained model
# ----------------------------
def load_model(model_dir):
    item_ids = np.load(os.path.join(model_dir, "item_ids.npy"))
    knn_indices = np.load(os.path.join(model_dir, "knn_indices.npy"))
    knn_sims = np.load(os.path.join(model_dir, "knn_sims.npy"))

    movieId_to_itemIndex = {int(m): i for i, m in enumerate(item_ids)}

    return item_ids, movieId_to_itemIndex, knn_indices, knn_sims


# ----------------------------
# Build user train history (positives only)
# ----------------------------
def build_train_history(train_path, movieId_to_itemIndex, rating_threshold=4.0):
    df = pd.read_csv(train_path)
    df = df[df["rating"] >= rating_threshold]

    df["itemIndex"] = df["movieId"].map(movieId_to_itemIndex)
    df = df.dropna(subset=["itemIndex"])
    df["itemIndex"] = df["itemIndex"].astype(np.int32)

    df = df.sort_values("timestamp").drop_duplicates(["userId", "itemIndex"], keep="last")

    history = df.groupby("userId")["itemIndex"].apply(lambda x: x.to_numpy()).to_dict()
    return history


# ----------------------------
# Score ALL items for a user
# ----------------------------
def score_user_all_items(user_history, knn_indices, knn_sims, topk_neighbors, n_items):

    scores = np.zeros(n_items, dtype=np.float32)

    for h in user_history:
        neigh = knn_indices[h, :topk_neighbors]
        sims = knn_sims[h, :topk_neighbors]
        scores[neigh] += sims

    return scores


# ----------------------------
# Main
# ----------------------------
def main():

    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv")
    MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model")
    OUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model",
                            "topN_recommendations_itemknn_with_titles.csv")

    TOPK_NEIGHBORS = 50
    TOPN = 10
    NUM_USERS_DEMO = 10

    print("Loading model...")
    item_ids, movieId_to_itemIndex, knn_indices, knn_sims = load_model(MODEL_DIR)
    n_items = len(item_ids)

    print("Building user history...")
    train_history = build_train_history(TRAIN_PATH, movieId_to_itemIndex)

    print("Selecting demo users...")
    users = list(train_history.keys())
    demo_users = users[:NUM_USERS_DEMO]

    all_recommendations = []

    for user in demo_users:

        user_history = train_history[user]
        scores = score_user_all_items(
            user_history,
            knn_indices,
            knn_sims,
            TOPK_NEIGHBORS,
            n_items
        )

        # Remove already seen items
        scores[user_history] = -np.inf

        top_indices = np.argsort(-scores)[:TOPN]

        for rank, item_index in enumerate(top_indices, start=1):
            movie_id = int(item_ids[item_index])
            score = float(scores[item_index])

            all_recommendations.append({
                "userId": user,
                "rank": rank,
                "movieId": movie_id,
                "score": score
            })

    recs_df = pd.DataFrame(all_recommendations)

    print("Merging with movie titles...")
    movies_df = pd.read_csv(MOVIES_PATH)[["movieId", "title", "genres"]]
    recs_df = recs_df.merge(movies_df, on="movieId", how="left")

    recs_df.to_csv(OUT_PATH, index=False)

    print("\n✅ Demo recommendations saved to:")
    print(OUT_PATH)
    print("\nPreview:")
    print(recs_df.head(20))


if __name__ == "__main__":
    main()
