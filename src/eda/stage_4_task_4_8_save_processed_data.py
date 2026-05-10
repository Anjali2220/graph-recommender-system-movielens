"""
STAGE 4 — Data Cleaning & Preprocessing
TASK 4.8 — Save Processed Data

Goal:
Persist final, reusable datasets in data/processed/
"""

import pandas as pd
from pathlib import Path
import shutil

# =====================================================
# 1. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent       # src/eda
PROJECT_DIR = BASE_DIR.parent.parent             # project root

OUTPUT_DIR = PROJECT_DIR / "outputs"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Source files (from outputs/)
ratings_clean_src = OUTPUT_DIR / "ratings_with_movies.csv"
interactions_encoded_src = OUTPUT_DIR / "ratings_encoded.csv"
user_movie_edges_src = OUTPUT_DIR / "user_movie_edges.csv"
movie_genre_edges_src = OUTPUT_DIR / "movie_genre_edges.csv"
genre_mapping_src = OUTPUT_DIR / "genre_mapping.csv"

# Destination files (data/processed/)
ratings_clean_dst = PROCESSED_DIR / "ratings_clean.csv"
interactions_encoded_dst = PROCESSED_DIR / "interactions_encoded.csv"
user_movie_edges_dst = PROCESSED_DIR / "user_movie_edges.csv"
movie_genre_edges_dst = PROCESSED_DIR / "movie_genre_edges.csv"
genre_mapping_dst = PROCESSED_DIR / "genre_mapping.csv"

# =====================================================
# 2. Save processed datasets
# =====================================================

print("💾 Saving processed datasets...\n")

shutil.copy(ratings_clean_src, ratings_clean_dst)
print("✅ ratings_clean.csv saved")

shutil.copy(interactions_encoded_src, interactions_encoded_dst)
print("✅ interactions_encoded.csv saved")

shutil.copy(user_movie_edges_src, user_movie_edges_dst)
print("✅ user_movie_edges.csv saved")

shutil.copy(movie_genre_edges_src, movie_genre_edges_dst)
print("✅ movie_genre_edges.csv saved")

shutil.copy(genre_mapping_src, genre_mapping_dst)
print("✅ genre_mapping.csv saved")

print("\n📁 All processed data saved in:", PROCESSED_DIR)
print("✅ TASK 4.8 COMPLETED SUCCESSFULLY")
