# ==========================================
# Task 3.7: Tag Usage Analysis (MovieLens 20M)
# ==========================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# 0. Robust project-root path setup
# -------------------------------
# This makes the script work no matter where you run it from.
# File location: src/eda/eda_tags_analysis.py
# Project root:  graph_based_recommender_system/
BASE_DIR = Path(__file__).resolve().parents[2]

# -------------------------------
# 1. Load tags.csv
# -------------------------------
tags_path = BASE_DIR / "data" / "raw" / "tags.csv"

if not tags_path.exists():
    raise FileNotFoundError(
        f"Could not find tags.csv at: {tags_path}\n"
        "Make sure your file is located at: data/raw/tags.csv"
    )

tags_df = pd.read_csv(tags_path)

print("=== TAG DATA LOADED SUCCESSFULLY ===")
print(f"Loaded from: {tags_path}")
print(tags_df.head())
print("\n")

# -------------------------------
# 2. Basic Tag Statistics
# -------------------------------
total_tags = len(tags_df)
unique_tags = tags_df["tag"].nunique()
unique_users = tags_df["userId"].nunique()
unique_movies = tags_df["movieId"].nunique()

print("=== BASIC TAG STATISTICS ===")
print(f"Total tag records      : {total_tags}")
print(f"Unique tags            : {unique_tags}")
print(f"Users who used tags    : {unique_users}")
print(f"Movies with tags       : {unique_movies}")
print("\n")

# -------------------------------
# 3. Top 20 Most Frequent Tags
# -------------------------------
top_tags = tags_df["tag"].value_counts().head(20)

print("=== TOP 20 MOST FREQUENT TAGS ===")
print(top_tags)
print("\n")

# Plot top 20 tags
plt.figure(figsize=(10, 6))
top_tags.plot(kind="bar")
plt.title("Top 20 Most Frequent Tags")
plt.xlabel("Tag")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

# -------------------------------
# 4. Movies with the Most Tags
# -------------------------------
tags_per_movie = tags_df.groupby("movieId").size().sort_values(ascending=False)

print("=== TOP 10 MOVIES WITH MOST TAGS ===")
print(tags_per_movie.head(10))
print("\n")

# -------------------------------
# 5. Tag Frequency Distribution
# -------------------------------
tag_freq = tags_df["tag"].value_counts()

plt.figure(figsize=(8, 5))
plt.hist(tag_freq.values, bins=50, log=True)
plt.title("Distribution of Tag Frequencies (Log Scale)")
plt.xlabel("Tag Frequency")
plt.ylabel("Number of Tags")
plt.tight_layout()
plt.show()

# -------------------------------
# 6. Summary Insight (For Report)
# -------------------------------
print("=== INSIGHT SUMMARY ===")
print(
    "Tags provide additional semantic information about movies.\n"
    "However, tag usage is highly skewed: a small number of tags are very frequent,\n"
    "while many tags appear only a few times.\n"
    "This indicates that tags can be useful but may introduce noise.\n"
    "Therefore, tag-based relationships are considered optional in the current scope."
)
