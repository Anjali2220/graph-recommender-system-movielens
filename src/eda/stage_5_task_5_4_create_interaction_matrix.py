"""
STAGE 5 — Baseline Recommender System
TASK 5.4 — Create User–Item Interaction Matrix

Goal:
- Load train_per_user.csv
- Map userId and movieId to matrix indices
- Create sparse user-item matrix (CSR format)
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from pathlib import Path

# =====================================================
# 1. Paths
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

TRAIN_PATH = PROJECT_DIR / "data" / "processed" / "train_per_user.csv"

print("📥 Loading training data...\n")
train_df = pd.read_csv(TRAIN_PATH, usecols=["userId", "movieId", "rating"])

print("Train shape:", train_df.shape)

# =====================================================
# 2. Map userId and movieId to continuous indices
# =====================================================

print("\n🔄 Mapping userId and movieId to matrix indices...")

unique_users = train_df["userId"].unique()
unique_movies = train_df["movieId"].unique()

user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}

train_df["user_idx"] = train_df["userId"].map(user_to_idx)
train_df["movie_idx"] = train_df["movieId"].map(movie_to_idx)

num_users = len(unique_users)
num_movies = len(unique_movies)

print("Number of users:", num_users)
print("Number of movies:", num_movies)

# =====================================================
# 3. Create sparse matrix
# =====================================================

print("\n🧩 Creating sparse user–item matrix...")

rows = train_df["user_idx"].values
cols = train_df["movie_idx"].values
values = train_df["rating"].values  # explicit ratings

train_matrix = csr_matrix(
    (values, (rows, cols)),
    shape=(num_users, num_movies)
)

print("Matrix shape:", train_matrix.shape)
print("Number of non-zero entries:", train_matrix.nnz)


print("Sparse train_matrix ready for baseline models.")
