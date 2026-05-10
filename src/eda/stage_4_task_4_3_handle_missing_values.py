"""
STAGE 4 — Data Cleaning & Preprocessing
TASK 4.3 — Handle Missing Values

Goal:
- RATINGS: drop rows missing critical fields (userId, movieId, rating)
- MOVIES: replace missing/empty genres with "Unknown"
- TAGS (optional): drop rows missing (userId, movieId, tag)
Output:
- outputs/ratings_clean_no_missing.csv
- outputs/movies_clean_no_missing.csv
- outputs/tags_clean_no_missing.csv (if tags.csv exists)
"""

import pandas as pd
from pathlib import Path

# =====================================================
# 1. Paths (FIXED for src/eda structure)
# =====================================================

BASE_DIR = Path(__file__).resolve().parent       # .../src/eda
PROJECT_DIR = BASE_DIR.parent.parent             # .../project_root

DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Inputs
ratings_input_path = OUTPUT_DIR / "ratings_clean_no_dupes.csv"  # from TASK 4.2
movies_input_path = DATA_DIR / "movies.csv"
tags_input_path = DATA_DIR / "tags.csv"  # optional

# Outputs
ratings_output_path = OUTPUT_DIR / "ratings_clean_no_missing.csv"
movies_output_path = OUTPUT_DIR / "movies_clean_no_missing.csv"
tags_output_path = OUTPUT_DIR / "tags_clean_no_missing.csv"


# =====================================================
# 2. Load datasets
# =====================================================

print("📥 Loading datasets...\n")

if not ratings_input_path.exists():
    raise FileNotFoundError(
        f"ratings_clean_no_dupes.csv not found at: {ratings_input_path}\n"
        "Run TASK 4.2 first."
    )

ratings = pd.read_csv(ratings_input_path)
movies = pd.read_csv(movies_input_path)

print("✅ Ratings shape (before):", ratings.shape)
print("✅ Movies shape  (before):", movies.shape)


# =====================================================
# 3. Handle missing values in RATINGS
# =====================================================

print("\n🔍 RATINGS: Missing values BEFORE")
print(ratings.isna().sum())

# Drop rows with missing critical fields
ratings_clean = ratings.dropna(subset=["userId", "movieId", "rating"])

removed_ratings = len(ratings) - len(ratings_clean)
print(f"\n🧹 RATINGS: Removed {removed_ratings} rows with missing critical values")
print("✅ Ratings shape (after):", ratings_clean.shape)


# =====================================================
# 4. Handle missing values in MOVIES
# =====================================================

print("\n🔍 MOVIES: Missing values BEFORE")
print(movies.isna().sum())

movies_clean = movies.copy()

# Replace missing/empty genres with "Unknown"
movies_clean["genres"] = (
    movies_clean["genres"]
    .fillna("Unknown")
    .replace("", "Unknown")
)

# Some datasets may store "(no genres listed)" — treat it as unknown if you want consistency
movies_clean["genres"] = movies_clean["genres"].replace("(no genres listed)", "Unknown")

print("\n🧹 MOVIES: Replaced missing/empty/no-genres with 'Unknown'")
print("✅ Movies shape (after):", movies_clean.shape)


# =====================================================
# 5. Handle missing values in TAGS (optional)
# =====================================================

if tags_input_path.exists():
    print("\n📥 Loading tags.csv...")
    tags = pd.read_csv(tags_input_path)
    print("✅ Tags shape (before):", tags.shape)

    print("\n🔍 TAGS: Missing values BEFORE")
    print(tags.isna().sum())

    # Drop rows with missing critical fields in tags
    tags_clean = tags.dropna(subset=["userId", "movieId", "tag"])

    removed_tags = len(tags) - len(tags_clean)
    print(f"\n🧹 TAGS: Removed {removed_tags} rows with missing critical values")
    print("✅ Tags shape (after):", tags_clean.shape)

else:
    tags = None
    tags_clean = None
    print("\nℹ️ tags.csv not found — skipping tags cleaning")


# =====================================================
# 6. Final validation
# =====================================================

print("\n✅ FINAL VALIDATION")

# Ratings validation
missing_ratings_critical = ratings_clean[["userId", "movieId", "rating"]].isna().sum().sum()
print("Missing critical in ratings:", int(missing_ratings_critical))
if missing_ratings_critical != 0:
    raise ValueError("❌ Ratings still contain missing critical values after cleaning!")

# Movies validation
missing_movies_genres = movies_clean["genres"].isna().sum()
print("Missing in movies.genres:", int(missing_movies_genres))
if missing_movies_genres != 0:
    raise ValueError("❌ Movies still contain missing genres after cleaning!")

# Tags validation (if present)
if tags_clean is not None:
    missing_tags_critical = tags_clean[["userId", "movieId", "tag"]].isna().sum().sum()
    print("Missing critical in tags:", int(missing_tags_critical))
    if missing_tags_critical != 0:
        raise ValueError("❌ Tags still contain missing critical values after cleaning!")


# =====================================================
# 7. Save cleaned datasets
# =====================================================

ratings_clean.to_csv(ratings_output_path, index=False)
movies_clean.to_csv(movies_output_path, index=False)

print("\n💾 Saved cleaned files:")
print(" -", ratings_output_path)
print(" -", movies_output_path)

if tags_clean is not None:
    tags_clean.to_csv(tags_output_path, index=False)
    print(" -", tags_output_path)

print("\n✅ TASK 4.3 COMPLETED SUCCESSFULLY")
print("No missing values remain in essential columns.")
