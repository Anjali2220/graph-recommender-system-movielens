import pandas as pd
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("../../data/raw")
ratings_path = DATA_DIR / "ratings.csv"

# -----------------------------
# Load data
# -----------------------------
print("Loading ratings data...")
ratings = pd.read_csv(ratings_path)
print("Ratings loaded.\n")

# -----------------------------
# Sparsity calculation
# -----------------------------
num_users = ratings["userId"].nunique()
num_movies = ratings["movieId"].nunique()
num_ratings = len(ratings)

total_possible_interactions = num_users * num_movies
sparsity = 1 - (num_ratings / total_possible_interactions)

# -----------------------------
# Print results
# -----------------------------
print("Sparsity Analysis Results")
print("==========================")
print(f"Number of users            : {num_users}")
print(f"Number of movies           : {num_movies}")
print(f"Number of ratings          : {num_ratings}")
print(f"Total possible interactions: {total_possible_interactions}")
print(f"Sparsity                   : {sparsity:.6f}")
print(f"Sparsity percentage        : {sparsity * 100:.2f}%")

