"""
STAGE 5 — Baseline Recommender System
TASK 5.5 — Train Baseline Model (ItemKNN) + Save Model Artifact

What this script does:
- Loads train_per_user.csv and test_per_user.csv (already created in TASK 5.3)
- Prints the effective train/test % (based on row counts)
- Builds sparse user–item matrix from TRAIN
- Fits ItemKNN with cosine similarity (Top-K neighbors per item)
- Saves a reusable "model artifact" folder:
    outputs/baseline/itemknn_model/
        - item_neighbors.npy   (num_items x K)
        - item_sims.npy        (num_items x K)
        - item_id_mapping.csv  (movieId <-> item_col)
        - user_id_mapping.csv  (userId <-> user_row)
        - config.json          (K, flags, shapes)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# =====================================================
# 1. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

TRAIN_PATH = PROJECT_DIR / "data" / "processed" / "train_per_user.csv"
TEST_PATH = PROJECT_DIR / "data" / "processed" / "test_per_user.csv"

MODEL_DIR = PROJECT_DIR / "outputs" / "baseline" / "itemknn_model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NEIGHBORS_PATH = MODEL_DIR / "item_neighbors.npy"
SIMS_PATH = MODEL_DIR / "item_sims.npy"
ITEM_MAP_PATH = MODEL_DIR / "item_id_mapping.csv"
USER_MAP_PATH = MODEL_DIR / "user_id_mapping.csv"
CONFIG_PATH = MODEL_DIR / "config.json"

# =====================================================
# 2. Hyperparameters
# =====================================================

K = 100  # number of nearest neighbors per item
USE_RATINGS_AS_VALUES = True  # True: explicit ratings as values; False: implicit 1.0
METRIC = "cosine"

# =====================================================
# 3. Load train/test and report effective split %
# =====================================================

print("📥 Loading train/test split files...\n")

train_df = pd.read_csv(TRAIN_PATH, usecols=["userId", "movieId", "rating"])
test_df = pd.read_csv(TEST_PATH, usecols=["userId", "movieId", "rating"])

train_rows = len(train_df)
test_rows = len(test_df)
total_rows = train_rows + test_rows

train_pct = 100.0 * train_rows / total_rows
test_pct = 100.0 * test_rows / total_rows

print("✅ Train rows:", train_rows)
print("✅ Test rows :", test_rows)
print(f"📊 Effective split (by interactions): Train {train_pct:.2f}% / Test {test_pct:.2f}%")
print("\nℹ️ Note: This is a per-user leave-last-1-out split, so test has 1 interaction per user.")

# =====================================================
# 4. Build user/movie index mappings (from TRAIN only)
# =====================================================

print("\n🔄 Building index mappings from TRAIN data...")

unique_users = train_df["userId"].unique()
unique_movies = train_df["movieId"].unique()

user_to_row = {u: i for i, u in enumerate(unique_users)}
movie_to_col = {m: j for j, m in enumerate(unique_movies)}

num_users = len(unique_users)
num_items = len(unique_movies)

print("Users in TRAIN:", num_users)
print("Items in TRAIN:", num_items)

# Save mapping files
pd.DataFrame({"userId": unique_users, "user_row": np.arange(num_users, dtype=np.int32)}).to_csv(USER_MAP_PATH, index=False)
pd.DataFrame({"movieId": unique_movies, "item_col": np.arange(num_items, dtype=np.int32)}).to_csv(ITEM_MAP_PATH, index=False)

print("💾 Saved user mapping:", USER_MAP_PATH)
print("💾 Saved item mapping:", ITEM_MAP_PATH)

# =====================================================
# 5. Build sparse user–item matrix (CSR)
# =====================================================

print("\n🧩 Building sparse user–item matrix (CSR)...")

rows = train_df["userId"].map(user_to_row).to_numpy(dtype=np.int32)
cols = train_df["movieId"].map(movie_to_col).to_numpy(dtype=np.int32)

if USE_RATINGS_AS_VALUES:
    vals = train_df["rating"].to_numpy(dtype=np.float32)
else:
    vals = np.ones(train_rows, dtype=np.float32)

R = csr_matrix((vals, (rows, cols)), shape=(num_users, num_items))

print("Matrix shape:", R.shape)
print("Non-zeros:", R.nnz)

# =====================================================
# 6. Fit ItemKNN (Top-K neighbors per item)
# =====================================================

print(f"\n🤖 Training ItemKNN (metric={METRIC}) with Top-{K} neighbors per item...")
print("Using item vectors (R.T) so we do NOT build a full item×item similarity matrix.")

X_items = R.T.tocsr()  # (num_items, num_users)

knn = NearestNeighbors(
    n_neighbors=K + 1,   # +1 because nearest neighbor is the item itself
    metric=METRIC,
    algorithm="brute",
    n_jobs=-1
)
knn.fit(X_items)

distances, indices = knn.kneighbors(X_items, return_distance=True)

# Remove self-neighbor
indices = indices[:, 1:].astype(np.int32)        # (num_items, K)
distances = distances[:, 1:].astype(np.float32)  # cosine distance

# Convert cosine distance to cosine similarity
sims = (1.0 - distances).astype(np.float32)

# Save neighbors + sims
np.save(NEIGHBORS_PATH, indices)
np.save(SIMS_PATH, sims)

print("\n💾 Saved ItemKNN model artifacts:")
print(" -", NEIGHBORS_PATH)
print(" -", SIMS_PATH)

# =====================================================
# 7. Save config metadata (so you can reload model later)
# =====================================================

config = {
    "model_type": "ItemKNN",
    "metric": METRIC,
    "K": K,
    "use_ratings_as_values": USE_RATINGS_AS_VALUES,
    "num_users_train": num_users,
    "num_items_train": num_items,
    "train_rows": int(train_rows),
    "test_rows": int(test_rows),
    "train_pct_by_rows": float(train_pct),
    "test_pct_by_rows": float(test_pct),
    "artifacts": {
        "item_neighbors": str(NEIGHBORS_PATH),
        "item_sims": str(SIMS_PATH),
        "user_mapping": str(USER_MAP_PATH),
        "item_mapping": str(ITEM_MAP_PATH),
    }
}

with open(CONFIG_PATH, "w") as f:
    json.dump(config, f, indent=2)

print("💾 Saved config:", CONFIG_PATH)

print("\n✅ TASK 5.5 COMPLETED SUCCESSFULLY")
print("ItemKNN baseline trained and saved as a reusable model artifact folder:")
print("📁", MODEL_DIR)
