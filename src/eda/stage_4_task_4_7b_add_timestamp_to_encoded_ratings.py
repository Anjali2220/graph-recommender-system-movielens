"""
STAGE 4 — Data Cleaning & Preprocessing
TASK 4.7 (Optional Extension)
Create ratings_encoded_with_timestamp.csv safely

Goal:
- Combine encoded indices with original timestamps
- Avoid incorrect joins
"""

import pandas as pd
from pathlib import Path

# =====================================================
# 1. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent       # src/eda
PROJECT_DIR = BASE_DIR.parent.parent             # project root
OUTPUT_DIR = PROJECT_DIR / "outputs"

ratings_clean_path = OUTPUT_DIR / "ratings_clean_no_missing.csv"
user_mapping_path = OUTPUT_DIR / "user_id_mapping.csv"
movie_mapping_path = OUTPUT_DIR / "movie_id_mapping.csv"

ratings_encoded_with_ts_path = OUTPUT_DIR / "ratings_encoded_with_timestamp.csv"

# =====================================================
# 2. Load datasets
# =====================================================

print("📥 Loading datasets...")

ratings = pd.read_csv(
    ratings_clean_path,
    usecols=["userId", "movieId", "rating", "timestamp"]
)

user_mapping = pd.read_csv(user_mapping_path)    # userId, user_idx
movie_mapping = pd.read_csv(movie_mapping_path)  # movieId, movie_idx

print("Ratings:", ratings.shape)
print("User mapping:", user_mapping.shape)
print("Movie mapping:", movie_mapping.shape)

# =====================================================
# 3. Attach user_idx and movie_idx (SAFE joins)
# =====================================================

print("\n🔗 Encoding ratings with timestamp...")

ratings = ratings.merge(user_mapping, on="userId", how="inner")
ratings = ratings.merge(movie_mapping, on="movieId", how="inner")

print("After encoding:", ratings.shape)

# =====================================================
# 4. Final encoded table with timestamp
# =====================================================

ratings_encoded_with_ts = ratings[
    ["user_idx", "movie_idx", "rating", "timestamp"]
]

print("\n📊 Preview:")
print(ratings_encoded_with_ts.head())

# =====================================================
# 5. Save output
# =====================================================

ratings_encoded_with_ts.to_csv(ratings_encoded_with_ts_path, index=False)

print("\n💾 Saved:", ratings_encoded_with_ts_path)
print("✅ ratings_encoded_with_timestamp.csv created safely")
