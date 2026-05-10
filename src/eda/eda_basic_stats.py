import pandas as pd
from pathlib import Path
# -----------------------------
# Paths (adjust if needed)
# -----------------------------
DATA_DIR = Path("../../data/raw")
OUTPUT_DIR = Path("../../outputs")
ratings_path = DATA_DIR / "ratings.csv"
movies_path = DATA_DIR / "movies.csv"
summary_path = OUTPUT_DIR / "eda_summary.txt"
# -----------------------------
# Load datasets
# -----------------------------
print("Loading datasets...")
ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)
print("Datasets loaded successfully.\n")
# -----------------------------
# Basic statistics
# -----------------------------
total_users = ratings["userId"].nunique()
total_movies_rated = ratings["movieId"].nunique()
total_movies = movies["movieId"].nunique()
total_ratings = len(ratings)
# Convert timestamps to datetime
ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s")
start_date = ratings["datetime"].min().date()
end_date = ratings["datetime"].max().date()
# -----------------------------
# Create summary table
# -----------------------------
summary_data = [
   ("Total Users", total_users),
   ("Total Movies (in movies.csv)", total_movies),
   ("Total Movies Rated", total_movies_rated),
   ("Total Ratings", total_ratings),
   ("Ratings Start Date", start_date),
   ("Ratings End Date", end_date),
]
summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])
# -----------------------------
# Print summary to console
# -----------------------------
print("Basic Dataset Statistics:")
print(summary_df.to_string(index=False))
# -----------------------------
# Save summary to file
# -----------------------------
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
with open(summary_path, "w") as f:
   f.write("MovieLens 20M Dataset - Basic Statistics\n")
   f.write("=" * 45 + "\n\n")
   f.write(summary_df.to_string(index=False))
print(f"\nSummary saved to: {summary_path}")