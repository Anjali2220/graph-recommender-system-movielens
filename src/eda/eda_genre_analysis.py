import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("../../data/raw")
OUTPUT_DIR = Path("../../outputs/figures")

movies_path = DATA_DIR / "movies.csv"
genre_plot_path = OUTPUT_DIR / "genre_distribution.png"

# -----------------------------
# Load data
# -----------------------------
print("Loading movies data...")
movies = pd.read_csv(movies_path)
print("Movies data loaded.\n")

# -----------------------------
# Genre processing
# -----------------------------
print("Processing genres...")

# Split genres by '|'
movies["genres"] = movies["genres"].str.split("|")

# Explode so each genre gets its own row
genre_exploded = movies.explode("genres")

# Remove '(no genres listed)'
genre_exploded = genre_exploded[genre_exploded["genres"] != "(no genres listed)"]

# Count movies per genre
genre_counts = genre_exploded["genres"].value_counts()

print("\nTop genres:")
print(genre_counts.head(10))

# -----------------------------
# Plot genre distribution
# -----------------------------
plt.figure(figsize=(12, 6))
genre_counts.plot(kind="bar")
plt.xlabel("Genre")
plt.ylabel("Number of Movies")
plt.title("Genre Distribution in MovieLens 20M Dataset")
plt.xticks(rotation=45, ha="right")
plt.grid(axis="y", linestyle="--", alpha=0.7)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(genre_plot_path)
plt.close()

print(f"\nGenre distribution plot saved to: {genre_plot_path}")
