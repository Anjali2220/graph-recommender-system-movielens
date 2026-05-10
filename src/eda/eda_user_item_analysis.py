import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("../../data/raw")
OUTPUT_DIR = Path("../../outputs/figures")

ratings_path = DATA_DIR / "ratings.csv"

user_plot_path = OUTPUT_DIR / "user_activity.png"
movie_plot_path = OUTPUT_DIR / "movie_popularity.png"

# -----------------------------
# Load data
# -----------------------------
print("Loading ratings data...")
ratings = pd.read_csv(ratings_path)
print("Ratings loaded.\n")

# -----------------------------
# A) Ratings per user (User activity)
# -----------------------------
print("Calculating ratings per user...")
user_counts = ratings.groupby("userId").size()

print("\nUser Activity (Ratings per user):")
print(user_counts.describe())

plt.figure(figsize=(10, 6))
plt.hist(user_counts, bins=50)
plt.xlabel("Number of Ratings by a User")
plt.ylabel("Number of Users")
plt.title("User Activity Distribution (Ratings per User)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(user_plot_path)
plt.close()

print(f"\nUser activity plot saved to: {user_plot_path}")

# -----------------------------
# B) Ratings per movie (Movie popularity)
# -----------------------------
print("\nCalculating ratings per movie...")
movie_counts = ratings.groupby("movieId").size()

print("\nMovie Popularity (Ratings per movie):")
print(movie_counts.describe())

plt.figure(figsize=(10, 6))
plt.hist(movie_counts, bins=50)
plt.xlabel("Number of Ratings for a Movie")
plt.ylabel("Number of Movies")
plt.title("Movie Popularity Distribution (Ratings per Movie)")
plt.grid(axis="y", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig(movie_plot_path)
plt.close()

print(f"\nMovie popularity plot saved to: {movie_plot_path}")
