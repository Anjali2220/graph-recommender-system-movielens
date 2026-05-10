"""
STAGE 4 — Data Cleaning & Preprocessing
TASK 4.5 — Encode User and Movie IDs

Goal:
- Map userId -> user_idx (0 ... num_users-1)
- Map movieId -> movie_idx (0 ... num_movies-1)
- Prepare data for graph construction and ML models
"""

import pandas as pd
from pathlib import Path

# =====================================================
# 1. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent       # src/eda
PROJECT_DIR = BASE_DIR.parent.parent             # project root

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ratings_input_path = OUTPUT_DIR / "ratings_clean_no_missing.csv"

ratings_encoded_path = OUTPUT_DIR / "ratings_encoded.csv"
user_mapping_path = OUTPUT_DIR / "user_id_mapping.csv"
movie_mapping_path = OUTPUT_DIR / "movie_id_mapping.csv"


# =====================================================
# 2. Load clean ratings
# =====================================================

print("📥 Loading cleaned ratings...")
ratings = pd.read_csv(ratings_input_path)

print("✅ Ratings shape:", ratings.shape)
print("Columns:", ratings.columns.tolist())


# =====================================================
# 3. Create USER ID mapping
# =====================================================

print("\n🔧 Encoding user IDs...")

unique_users = ratings["userId"].unique()
user_id_to_idx = {uid: idx for idx, uid in enumerate(unique_users)}

ratings["user_idx"] = ratings["userId"].map(user_id_to_idx)

print("Total users:", len(unique_users))


# =====================================================
# 4. Create MOVIE ID mapping
# =====================================================

print("\n🔧 Encoding movie IDs...")

unique_movies = ratings["movieId"].unique()
movie_id_to_idx = {mid: idx for idx, mid in enumerate(unique_movies)}

ratings["movie_idx"] = ratings["movieId"].map(movie_id_to_idx)

print("Total movies:", len(unique_movies))


# =====================================================
# 5. Final encoded interaction table
# =====================================================

ratings_encoded = ratings[["user_idx", "movie_idx", "rating"]]

print("\n📊 Encoded ratings preview:")
print(ratings_encoded.head())


# =====================================================
# 6. Save outputs
# =====================================================

ratings_encoded.to_csv(ratings_encoded_path, index=False)

user_mapping_df = pd.DataFrame({
    "userId": list(user_id_to_idx.keys()),
    "user_idx": list(user_id_to_idx.values())
})
user_mapping_df.to_csv(user_mapping_path, index=False)

movie_mapping_df = pd.DataFrame({
    "movieId": list(movie_id_to_idx.keys()),
    "movie_idx": list(movie_id_to_idx.values())
})
movie_mapping_df.to_csv(movie_mapping_path, index=False)

print("\n💾 Saved files:")
print(" -", ratings_encoded_path)
print(" -", user_mapping_path)
print(" -", movie_mapping_path)

print("\n✅ TASK 4.5 COMPLETED SUCCESSFULLY")
print("User and movie IDs encoded into continuous indices.")
