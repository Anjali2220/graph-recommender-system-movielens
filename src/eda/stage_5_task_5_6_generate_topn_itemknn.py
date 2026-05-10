"""
STAGE 5 — Baseline Recommender System
TASK 5.6b — Make Human-Readable Recommendations (Top-N with titles + genres)

Inputs:
- outputs/baseline/itemknn_model/topN_recommendations_itemknn.csv
- data/processed/ratings_clean.csv  (for movieId -> title, genres)

Outputs:
- outputs/baseline/itemknn_model/top5_itemknn_with_titles_genres.csv
- outputs/baseline/itemknn_model/top5_demo_users.csv (optional small sample)
"""

import pandas as pd
from pathlib import Path

# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

RECS_PATH = PROJECT_DIR / "outputs" / "baseline" / "itemknn_model" / "topN_recommendations_itemknn.csv"
RATINGS_CLEAN_PATH = PROJECT_DIR / "data" / "processed" / "ratings_clean.csv"

OUT_ALL_PATH = PROJECT_DIR / "outputs" / "baseline" / "itemknn_model" / "top5_itemknn_with_titles_genres.csv"
OUT_DEMO_PATH = PROJECT_DIR / "outputs" / "baseline" / "itemknn_model" / "top5_demo_users.csv"

# =====================================================
# Load recommendations
# =====================================================
print("📥 Loading recommendations...")
recs = pd.read_csv(RECS_PATH)
print("Recs shape:", recs.shape)
print("Columns:", recs.columns.tolist())

# =====================================================
# Load movie metadata (movieId -> title, genres)
# =====================================================
print("\n📥 Loading movie metadata from ratings_clean.csv...")
# ratings_clean has repeated movie rows; we keep unique movie metadata
movies = pd.read_csv(RATINGS_CLEAN_PATH, usecols=["movieId", "title", "genres"]).drop_duplicates("movieId")
print("Unique movies:", movies.shape)

# =====================================================
# Merge to make human-readable
# =====================================================
print("\n🔗 Merging recommendations with movie titles & genres...")
recs_hr = recs.merge(movies, on="movieId", how="left")

missing = recs_hr["title"].isna().sum()
print("Missing titles after merge:", int(missing))

# Reorder columns nicely
recs_hr = recs_hr[["userId", "rank", "movieId", "title", "genres", "score"]]

# =====================================================
# Save full human-readable file
# =====================================================
recs_hr.to_csv(OUT_ALL_PATH, index=False)
print("\n💾 Saved (all users):", OUT_ALL_PATH)

# =====================================================
# Optional: Save a small demo sample (easier to show professor)
# =====================================================
demo_users = recs_hr["userId"].drop_duplicates().head(10)  # first 10 users as demo
demo_df = recs_hr[recs_hr["userId"].isin(demo_users)].copy()

demo_df.to_csv(OUT_DEMO_PATH, index=False)
print("💾 Saved (demo users):", OUT_DEMO_PATH)

print("\n✅ Human-readable recommendations created successfully.")
