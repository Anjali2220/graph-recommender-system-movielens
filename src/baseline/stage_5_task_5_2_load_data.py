"""
STAGE 5 — Baseline Recommender System Development
TASK 5.2 — Load Processed Data

Goal:
- Load data/processed/ratings_clean.csv
- Check shape, columns, dtypes
- Count unique users and movies
- Confirm required columns exist (for ItemKNN + PyTorch MF)
"""

import pandas as pd
from pathlib import Path

# =====================================================
# 1. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent        # .../src/baseline
PROJECT_DIR = BASE_DIR.parent.parent              # .../project_root

DATA_PATH = PROJECT_DIR / "data" / "processed" / "ratings_clean.csv"


# =====================================================
# 2. Load dataset
# =====================================================

print("📥 Loading processed dataset for baseline models...\n")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"❌ ratings_clean.csv not found at: {DATA_PATH}")

ratings = pd.read_csv(DATA_PATH)

print("✅ Dataset loaded successfully")
print("File:", DATA_PATH)


# =====================================================
# 3. Basic sanity checks
# =====================================================

print("\n🔍 BASIC CHECKS")
print("-" * 60)

print("Shape (rows, cols):", ratings.shape)

print("\nColumns:")
print(ratings.columns.tolist())

print("\nDtypes:")
print(ratings.dtypes)

# Unique counts (important for both baselines)
num_users = ratings["userId"].nunique()
num_movies = ratings["movieId"].nunique()

print("\nUnique users:", num_users)
print("Unique movies:", num_movies)

print("\nPreview (first 5 rows):")
print(ratings.head())


# =====================================================
# 4. Required column validation
# =====================================================

required_cols = {"userId", "movieId", "rating"}

missing = required_cols - set(ratings.columns)
if missing:
    raise ValueError(f"❌ Missing required columns: {missing}")

# timestamp is recommended (for time split), but not strictly required to load
if "timestamp" not in ratings.columns:
    print("\n⚠️ Warning: 'timestamp' column not found. Time-based split will not be possible.")
else:
    print("\n✅ 'timestamp' column found (good for time-based split).")

# title/genres are helpful for analysis but not required for training
if "title" not in ratings.columns or "genres" not in ratings.columns:
    print("ℹ️ Note: title/genres not found. This is okay for training, but less interpretable.")
else:
    print("✅ 'title' and 'genres' found (good for analysis & reporting).")


# =====================================================
# 5. Final confirmation
# =====================================================

print("\n✅ TASK 5.2 COMPLETED SUCCESSFULLY")
print("Dataset is ready for baseline training:")
print("- ItemKNN: userId/movieId/rating are present")
print("- PyTorch MF: userId/movieId can be re-indexed for embeddings")
