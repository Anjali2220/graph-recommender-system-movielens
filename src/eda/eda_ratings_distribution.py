import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# -----------------------------
# Paths
# -----------------------------
DATA_DIR = Path("../../data/raw")
OUTPUT_DIR = Path("../../outputs/figures")

ratings_path = DATA_DIR / "ratings.csv"
output_path = OUTPUT_DIR / "rating_distribution.png"

# -----------------------------
# Load ratings data
# -----------------------------
print("Loading ratings data...")
ratings = pd.read_csv(ratings_path)
print("Ratings data loaded.\n")

# -----------------------------
# Rating distribution
# -----------------------------
rating_counts = ratings["rating"].value_counts().sort_index()

print("Rating Distribution:")
print(rating_counts)

# -----------------------------
# Plot rating distribution
# -----------------------------
plt.figure(figsize=(10, 6))
rating_counts.plot(kind="bar")
plt.xlabel("Rating Value")
plt.ylabel("Number of Ratings")
plt.title("Rating Distribution in MovieLens 20M Dataset")
plt.xticks(rotation=0)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# -----------------------------
# Save plot
# -----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(output_path)
plt.close()

print(f"Rating distribution plot saved to: {output_path}")
