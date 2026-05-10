"""
STAGE 5 — Baseline Recommender System
TASK 5.3 — Train/Test Split (Per-User Time Split)

Goal:
- For each user, sort interactions by timestamp
- Put the last interaction into test, the rest into train
- Ensures every test user exists in train
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent        # src/baseline
PROJECT_DIR = BASE_DIR.parent.parent              # project root

DATA_PATH = PROJECT_DIR / "data" / "processed" / "ratings_clean.csv"

TRAIN_PATH = PROJECT_DIR / "data" / "processed" / "train_per_user.csv"
TEST_PATH  = PROJECT_DIR / "data" / "processed" / "test_per_user.csv"

print("📥 Loading ratings data...")
df = pd.read_csv(DATA_PATH, usecols=["userId", "movieId", "rating", "timestamp", "title", "genres"])
print("Original shape:", df.shape)

print("\n⏳ Sorting by userId + timestamp...")
df = df.sort_values(["userId", "timestamp"])

# Mark the last interaction per user as test
print("\n🧪 Creating per-user leave-last-1-out split...")
is_last = df.groupby("userId").cumcount(ascending=True) == (df.groupby("userId")["userId"].transform("size") - 1)

test_df = df[is_last].copy()
train_df = df[~is_last].copy()

# Optional safety: remove users with only 1 interaction (they’d have empty train)
train_counts = train_df.groupby("userId").size()
valid_users = set(train_counts[train_counts >= 1].index)

test_df = test_df[test_df["userId"].isin(valid_users)]
train_df = train_df[train_df["userId"].isin(valid_users)]

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)
print("Train users:", train_df["userId"].nunique())
print("Test users :", test_df["userId"].nunique())

train_df.to_csv(TRAIN_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)

print("\n💾 Saved:")
print(" -", TRAIN_PATH)
print(" -", TEST_PATH)
print("\n✅ Per-user time split ready for evaluation.")
