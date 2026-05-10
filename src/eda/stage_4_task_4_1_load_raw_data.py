"""
STAGE 4 — Data Cleaning & Preprocessing
TASK 4.1 — Load Raw Data

Goal:
- Load raw CSV files correctly (ratings.csv, movies.csv, optional tags.csv)
- Perform basic sanity checks (shape, columns, dtypes)
- Validate required columns exist
"""

import pandas as pd
from pathlib import Path


# =====================================================
# 1. Define project paths (FIXED for src/eda structure)
# =====================================================

BASE_DIR = Path(__file__).resolve().parent       # .../src/eda
PROJECT_DIR = BASE_DIR.parent.parent             # .../project_root (go up 2 levels)

DATA_DIR = PROJECT_DIR / "data" / "raw"

ratings_path = DATA_DIR / "ratings.csv"
movies_path = DATA_DIR / "movies.csv"
tags_path = DATA_DIR / "tags.csv"   # optional


# =====================================================
# 2. Debug prints (to confirm paths are correct)
# =====================================================

print("📌 Path Debug")
print("BASE_DIR    :", BASE_DIR)
print("PROJECT_DIR :", PROJECT_DIR)
print("DATA_DIR    :", DATA_DIR)
print("ratings_path:", ratings_path)
print("movies_path :", movies_path)
print("tags_path   :", tags_path)
print("-" * 50)


# =====================================================
# 3. Load raw datasets
# =====================================================

print("📥 Loading raw datasets...\n")

# --- Ratings ---
if not ratings_path.exists():
    raise FileNotFoundError(f"ratings.csv not found at: {ratings_path}")
ratings = pd.read_csv(ratings_path)

# --- Movies ---
if not movies_path.exists():
    raise FileNotFoundError(f"movies.csv not found at: {movies_path}")
movies = pd.read_csv(movies_path)

# --- Tags (optional) ---
if tags_path.exists():
    tags = pd.read_csv(tags_path)
    print("✅ tags.csv loaded")
else:
    tags = None
    print("ℹ️ tags.csv not found (skipping for now)")

print("\n✅ ratings.csv and movies.csv loaded successfully")


# =====================================================
# 4. Basic sanity checks
# =====================================================

print("\n🔍 BASIC SANITY CHECKS")
print("-" * 50)

# --- Shapes ---
print("Ratings shape:", ratings.shape)
print("Movies shape :", movies.shape)
if tags is not None:
    print("Tags shape   :", tags.shape)

# --- Columns ---
print("\nRatings columns:", ratings.columns.tolist())
print("Movies columns :", movies.columns.tolist())
if tags is not None:
    print("Tags columns   :", tags.columns.tolist())

# --- Data types ---
print("\nRatings dtypes:")
print(ratings.dtypes)

print("\nMovies dtypes:")
print(movies.dtypes)

if tags is not None:
    print("\nTags dtypes:")
    print(tags.dtypes)


# =====================================================
# 5. Final validation (required columns check)
# =====================================================

required_ratings_cols = {"userId", "movieId", "rating"}
required_movies_cols = {"movieId", "title"}

missing_ratings_cols = required_ratings_cols - set(ratings.columns)
missing_movies_cols = required_movies_cols - set(movies.columns)

if missing_ratings_cols:
    raise ValueError(f"❌ ratings.csv is missing required columns: {missing_ratings_cols}")

if missing_movies_cols:
    raise ValueError(f"❌ movies.csv is missing required columns: {missing_movies_cols}")

print("\n✅ TASK 4.1 COMPLETED SUCCESSFULLY")
print("Raw data loaded and validated.).")
