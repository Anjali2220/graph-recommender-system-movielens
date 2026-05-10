import os
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


# ==========================================================
# 1️⃣ Convert Explicit Ratings → Implicit (rating >= threshold)
# ==========================================================
def build_implicit_interactions(
    df: pd.DataFrame,
    rating_threshold: float = 4.0,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    time_col: str = "timestamp",
):
    """
    Keep only positive interactions (rating >= threshold).
    Remove duplicate (user,item) keeping latest timestamp.
    """
    pos = df[df[rating_col] >= rating_threshold].copy()

    if time_col in pos.columns:
        pos = pos.sort_values(time_col)
        pos = pos.drop_duplicates(subset=[user_col, item_col], keep="last")
    else:
        pos = pos.drop_duplicates(subset=[user_col, item_col], keep="last")

    return pos


# ==========================================================
# 2️⃣ Create ID Mappings
# ==========================================================
def create_mappings(df, user_col="userId", item_col="movieId"):
    user_ids = df[user_col].unique()
    item_ids = df[item_col].unique()

    user_ids = np.array(user_ids, dtype=np.int64)
    item_ids = np.array(item_ids, dtype=np.int64)

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}

    return user_ids, item_ids, user2idx, item2idx


# ==========================================================
# 3️⃣ Build Sparse User-Item Matrix
# ==========================================================
def build_sparse_matrix(df, user2idx, item2idx):
    rows = df["userId"].map(user2idx).astype(np.int32).to_numpy()
    cols = df["movieId"].map(item2idx).astype(np.int32).to_numpy()
    data = np.ones(len(df), dtype=np.float32)

    n_users = len(user2idx)
    n_items = len(item2idx)

    R = csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    return R


# ==========================================================
# 4️⃣ Train ItemKNN
# ==========================================================
def train_itemknn(R, topk=100):
    """
    Compute top-K item neighbors using cosine similarity.
    """
    print("Computing item-item cosine similarity (topK =", topk, ")...")

    # Item vectors (items x users)
    item_user_matrix = R.T.tocsr()

    knn = NearestNeighbors(
        n_neighbors=topk + 1,  # +1 because first neighbor is itself
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )

    knn.fit(item_user_matrix)

    distances, indices = knn.kneighbors(item_user_matrix)

    similarities = 1.0 - distances

    # Remove self-neighbor
    indices = indices[:, 1:]
    similarities = similarities[:, 1:]

    return indices.astype(np.int32), similarities.astype(np.float32)


# ==========================================================
# 5️⃣ Save Model Artifacts
# ==========================================================
def save_model(
    output_dir,
    user_ids,
    item_ids,
    knn_indices,
    knn_sims,
    config,
):
    os.makedirs(output_dir, exist_ok=True)

    np.save(os.path.join(output_dir, "user_ids.npy"), user_ids)
    np.save(os.path.join(output_dir, "item_ids.npy"), item_ids)
    np.save(os.path.join(output_dir, "knn_indices.npy"), knn_indices)
    np.save(os.path.join(output_dir, "knn_sims.npy"), knn_sims)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    print("\n✅ Model saved successfully at:")
    print(output_dir)


# ==========================================================
# 6️⃣ Main
# ==========================================================
def main():

    # ----------------------
    # Paths
    # ----------------------
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")
    OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model")

    # ----------------------
    # Hyperparameters
    # ----------------------
    RATING_THRESHOLD = 4.0
    TOPK = 100

    print("Project root:", PROJECT_ROOT)
    print("Train file:", TRAIN_PATH)
    print("Output dir:", OUTPUT_DIR)
    print("Threshold:", RATING_THRESHOLD, "| TopK:", TOPK)

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError("train.csv not found!")

    # ----------------------
    # Load Train
    # ----------------------
    train_df = pd.read_csv(TRAIN_PATH)
    print("Train rows:", len(train_df))

    # ----------------------
    # Convert to Implicit
    # ----------------------
    pos_train = build_implicit_interactions(
        train_df,
        rating_threshold=RATING_THRESHOLD,
    )
    print("Positive interactions:", len(pos_train))

    # ----------------------
    # Build Matrix
    # ----------------------
    user_ids, item_ids, user2idx, item2idx = create_mappings(pos_train)
    R = build_sparse_matrix(pos_train, user2idx, item2idx)

    print("Matrix shape:", R.shape)
    print("Non-zero interactions:", R.nnz)

    # ----------------------
    # Train ItemKNN
    # ----------------------
    knn_indices, knn_sims = train_itemknn(R, topk=TOPK)

    print("KNN shape:", knn_indices.shape)

    # ----------------------
    # Save Model
    # ----------------------
    config = {
        "model": "ItemKNN",
        "rating_threshold": RATING_THRESHOLD,
        "topK": TOPK,
        "similarity": "cosine",
        "matrix_type": "implicit_binary",
        "n_users": int(len(user_ids)),
        "n_items": int(len(item_ids)),
        "positive_interactions": int(len(pos_train)),
    }

    save_model(
        OUTPUT_DIR,
        user_ids,
        item_ids,
        knn_indices,
        knn_sims,
        config,
    )

    # ----------------------
    # Show Example
    # ----------------------
    print("\nExample model inspection:")
    print("First item ID:", item_ids[0])
    print("Top-10 similar item IDs:", item_ids[knn_indices[0][:10]])
    print("Top-10 similarity scores:", knn_sims[0][:10])


if __name__ == "__main__":
    main()
