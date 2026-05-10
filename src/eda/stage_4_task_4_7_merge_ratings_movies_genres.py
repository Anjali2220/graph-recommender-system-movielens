"""
STAGE 4 — Data Cleaning & Preprocessing
TASK 4.7 — Merge Ratings with Movies (and Genres)

Outputs:
1) ratings_with_movies.csv
   - userId, movieId, rating, timestamp, title, genres

2) ratings_encoded_with_movies.csv
   - user_idx, movie_idx, rating, timestamp, title, genres

3) user_movie_edges.csv
   - user_idx, movie_idx, rating  (+ optional timestamp)

(4) movie_genre_edges.csv already exists from TASK 4.6
    - movie_idx, genre_idx

(5) genre_mapping.csv already exists from TASK 4.6
    - genre, genre_idx
"""

import pandas as pd
from pathlib import Path

# =====================================================
# 1. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent       # src/eda
PROJECT_DIR = BASE_DIR.parent.parent             # project root
OUTPUT_DIR = PROJECT_DIR / "outputs"

# Inputs
ratings_clean_path = OUTPUT_DIR / "ratings_clean_no_missing.csv"
ratings_encoded_path = OUTPUT_DIR / "ratings_encoded.csv"
movies_clean_path = OUTPUT_DIR / "movies_clean_no_missing.csv"
movie_mapping_path = OUTPUT_DIR / "movie_id_mapping.csv"
movie_genre_edges_path = OUTPUT_DIR / "movie_genre_edges.csv"   # from task 4.6
genre_mapping_path = OUTPUT_DIR / "genre_mapping.csv"           # from task 4.6

# Outputs
ratings_with_movies_path = OUTPUT_DIR / "ratings_with_movies.csv"
ratings_encoded_with_movies_path = OUTPUT_DIR / "ratings_encoded_with_movies.csv"
user_movie_edges_path = OUTPUT_DIR / "user_movie_edges.csv"

# =====================================================
# 2. Load data (use only necessary columns to save memory)
# =====================================================

print("📥 Loading inputs...")

ratings_clean = pd.read_csv(
    ratings_clean_path,
    usecols=["userId", "movieId", "rating", "timestamp"]
)

movies_clean = pd.read_csv(
    movies_clean_path,
    usecols=["movieId", "title", "genres"]
)

movie_mapping = pd.read_csv(movie_mapping_path)  # movieId, movie_idx

ratings_encoded = pd.read_csv(ratings_encoded_path)  # user_idx, movie_idx, rating

print("✅ ratings_clean:", ratings_clean.shape)
print("✅ movies_clean :", movies_clean.shape)
print("✅ movie_mapping:", movie_mapping.shape)
print("✅ ratings_encoded:", ratings_encoded.shape)

# =====================================================
# 3. Merge: Ratings + Movies (baseline-ready table)
# =====================================================

print("\n🔗 Merging ratings with movies (baseline table)...")

ratings_with_movies = ratings_clean.merge(movies_clean, on="movieId", how="left")

missing_titles = ratings_with_movies["title"].isna().sum()
print("Missing movie titles after merge:", int(missing_titles))

ratings_with_movies.to_csv(ratings_with_movies_path, index=False)
print("💾 Saved:", ratings_with_movies_path)
print("✅ ratings_with_movies:", ratings_with_movies.shape)

# =====================================================
# 4. Merge: Encoded ratings + Movie metadata (GNN-friendly + interpretable)
# =====================================================

print("\n🔗 Preparing encoded ratings + movie metadata...")

# Add movieId to encoded ratings by joining movie_mapping (movie_idx -> movieId)
movie_idx_to_movieId = movie_mapping[["movieId", "movie_idx"]]

ratings_encoded_plus_ids = ratings_encoded.merge(movie_idx_to_movieId, on="movie_idx", how="left")

# Add timestamp by joining with ratings_clean (userId, movieId)
# First, attach userId too (from user mapping) is optional — we can merge on movieId only for metadata
# Here we add movie metadata using movieId:
ratings_encoded_with_movies = ratings_encoded_plus_ids.merge(movies_clean, on="movieId", how="left")

# Timestamp isn't in ratings_encoded.csv, so optionally add it by merging via movieId + rating + ... is risky.
# Best practice: keep timestamp only in ratings_with_movies.
# We'll keep encoded_with_movies without timestamp to avoid incorrect joins.

ratings_encoded_with_movies = ratings_encoded_with_movies[["user_idx", "movie_idx", "rating", "movieId", "title", "genres"]]

ratings_encoded_with_movies.to_csv(ratings_encoded_with_movies_path, index=False)
print("💾 Saved:", ratings_encoded_with_movies_path)
print("✅ ratings_encoded_with_movies:", ratings_encoded_with_movies.shape)

# =====================================================
# 5. Save user–movie edges for graph use
# =====================================================

print("\n🧩 Creating user–movie edge table...")

user_movie_edges = ratings_encoded[["user_idx", "movie_idx", "rating"]]
user_movie_edges.to_csv(user_movie_edges_path, index=False)

print("💾 Saved:", user_movie_edges_path)
print("✅ user_movie_edges:", user_movie_edges.shape)

# =====================================================
# 6. Confirm genre files exist (from Task 4.6)
# =====================================================

print("\n📌 Checking genre outputs...")

if movie_genre_edges_path.exists():
    mg = pd.read_csv(movie_genre_edges_path)
    print("✅ movie_genre_edges exists:", mg.shape)
else:
    print("⚠️ movie_genre_edges.csv not found. Run TASK 4.6 first.")

if genre_mapping_path.exists():
    gm = pd.read_csv(genre_mapping_path)
    print("✅ genre_mapping exists:", gm.shape)
else:
    print("⚠️ genre_mapping.csv not found. Run TASK 4.6 first.")

print("\n✅ TASK 4.7 COMPLETED SUCCESSFULLY")
print("You now have:")
print(" - Baseline table: ratings_with_movies.csv")
print(" - Encoded+metadata: ratings_encoded_with_movies.csv")
print(" - Graph edges: user_movie_edges.csv and movie_genre_edges.csv")
