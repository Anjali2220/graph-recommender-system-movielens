"""
STAGE 4 — Data Cleaning & Preprocessing
TASK 4.2 — Remove Duplicate Records

Goal:
- Remove duplicate user–movie interactions in ratings
- Keep only the latest rating based on timestamp
"""

import pandas as pd
from pathlib import Path

# =====================================================
# 1. Paths (same fixed structure: src/eda -> project root)
# =====================================================

BASE_DIR = Path(__file__).resolve().parent       # .../src/eda
PROJECT_DIR = BASE_DIR.parent.parent             # .../project_root
DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

ratings_path = DATA_DIR / "ratings.csv"
clean_ratings_path = OUTPUT_DIR / "ratings_clean_no_dupes.csv"

# =====================================================
# 2. Load ratings
# =====================================================

print("📥 Loading ratings.csv...")
ratings = pd.read_csv(ratings_path)
print("✅ Loaded ratings:", ratings.shape)

# =====================================================
# 3. Check duplicates on (userId, movieId)
# =====================================================

dup_mask = ratings.duplicated(subset=["userId", "movieId"], keep=False)
num_dup_rows = int(dup_mask.sum())

print("\n🔍 Duplicate check")
print("Rows involved in duplicates (including originals):", num_dup_rows)

if num_dup_rows == 0:
    print("✅ No duplicate user–movie pairs found. Nothing to remove.")
    ratings_clean = ratings.copy()
else:
    # =====================================================
    # 4. Remove duplicates (keep latest by timestamp)
    # =====================================================
    print("\n🧹 Removing duplicates (keeping latest by timestamp)...")

    # Sort so the latest timestamp is last for each user-movie
    ratings_sorted = ratings.sort_values(by=["userId", "movieId", "timestamp"])

    # Keep last = latest timestamp
    ratings_clean = ratings_sorted.drop_duplicates(subset=["userId", "movieId"], keep="last")

    removed = len(ratings) - len(ratings_clean)
    print("✅ Removed duplicate rows:", removed)

# =====================================================
# 5. Validate duplicates are gone
# =====================================================

remaining_dupes = ratings_clean.duplicated(subset=["userId", "movieId"]).sum()
print("\n✅ Validation")
print("Remaining duplicate (userId, movieId) pairs:", int(remaining_dupes))

if remaining_dupes != 0:
    raise ValueError("❌ Duplicate pairs still exist after cleaning. Something went wrong.")

# =====================================================
# 6. Save clean ratings
# =====================================================

ratings_clean.to_csv(clean_ratings_path, index=False)
print("\n💾 Saved:", clean_ratings_path)
print("✅ TASK 4.2 COMPLETED SUCCESSFULLY")
print("Clean ratings data has unique (userId, movieId) interactions.")
