"""
STAGE 4 — Data Cleaning & Preprocessing
TASK 4.6 — Process Genres for Graph Use

Goal:
- Create genre nodes with genre_idx
- Create movie–genre edges for graph construction
"""

import pandas as pd
from pathlib import Path

# =====================================================
# 1. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent       # src/eda
PROJECT_DIR = BASE_DIR.parent.parent             # project root

DATA_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

movies_path = DATA_DIR / "movies.csv"
movie_mapping_path = OUTPUT_DIR / "movie_id_mapping.csv"

genre_mapping_path = OUTPUT_DIR / "genre_mapping.csv"
movie_genre_edges_path = OUTPUT_DIR / "movie_genre_edges.csv"


# =====================================================
# 2. Load datasets
# =====================================================

print("📥 Loading movies and movie_id_mapping...")

movies = pd.read_csv(movies_path)
movie_mapping = pd.read_csv(movie_mapping_path)

print("Movies shape:", movies.shape)
print("Movie mapping shape:", movie_mapping.shape)


# =====================================================
# 3. Merge movie_idx into movies
# =====================================================

movies = movies.merge(movie_mapping, on="movieId", how="inner")

print("Movies with movie_idx shape:", movies.shape)


# =====================================================
# 4. Split genres
# =====================================================

print("\n🔧 Processing genres...")

# Ensure no missing genres
movies["genres"] = movies["genres"].fillna("Unknown")

# Split genres into list
movies["genre_list"] = movies["genres"].str.split("|")

# Explode into one genre per row
movie_genres = movies[["movie_idx", "genre_list"]].explode("genre_list")
movie_genres.rename(columns={"genre_list": "genre"}, inplace=True)

# Remove 'Unknown' if you don’t want it as a node (optional)
movie_genres = movie_genres[movie_genres["genre"] != "Unknown"]

print("Movie–genre rows:", movie_genres.shape)


# =====================================================
# 5. Create genre mapping
# =====================================================

unique_genres = sorted(movie_genres["genre"].unique())
genre_to_idx = {g: i for i, g in enumerate(unique_genres)}

genre_mapping = pd.DataFrame({
    "genre": list(genre_to_idx.keys()),
    "genre_idx": list(genre_to_idx.values())
})

print("Total genres:", len(genre_mapping))


# =====================================================
# 6. Create movie–genre edge table
# =====================================================

movie_genres["genre_idx"] = movie_genres["genre"].map(genre_to_idx)

movie_genre_edges = movie_genres[["movie_idx", "genre_idx"]]

print("\n📊 Movie–Genre edges preview:")
print(movie_genre_edges.head())


# =====================================================
# 7. Save outputs
# =====================================================

genre_mapping.to_csv(genre_mapping_path, index=False)
movie_genre_edges.to_csv(movie_genre_edges_path, index=False)

print("\n💾 Saved files:")
print(" -", genre_mapping_path)
print(" -", movie_genre_edges_path)

print("\n✅ TASK 4.6 COMPLETED SUCCESSFULLY")
print("Genres processed as graph nodes.")
