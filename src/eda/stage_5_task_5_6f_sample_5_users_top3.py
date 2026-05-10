import pandas as pd
from pathlib import Path

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

TRAIN_PATH = PROJECT_DIR / "data" / "processed" / "train_per_user.csv"
RECS_PATH  = PROJECT_DIR / "outputs" / "baseline" / "itemknn_model" / "top5_itemknn_with_titles_genres.csv"
OUT_PATH   = PROJECT_DIR / "outputs" / "baseline" / "itemknn_model" / "sample_5_users_top3_recs.csv"

# =========================
# Load
# =========================
print("📥 Loading train history...")
train = pd.read_csv(TRAIN_PATH, usecols=["userId", "movieId", "rating"])

print("📥 Loading recommendations...")
recs = pd.read_csv(RECS_PATH)

# =========================
# Pick 5 users (simple method: smallest 5 userIds)
# You can change this to .sample(5, random_state=42) if you want random users
# =========================
users = sorted(recs["userId"].unique())[:5]
print("✅ Selected users:", users)

# =========================
# Build interest summary from TRAIN (liked movies = rating >= 4)
# We'll summarize: top genres + average rating
# =========================
def top_genres(genres_series, top_k=3):
    all_g = []
    for g in genres_series.dropna().astype(str):
        all_g += [x.strip() for x in g.split("|") if x.strip()]
    if not all_g:
        return "N/A"
    counts = pd.Series(all_g).value_counts()
    return ", ".join(counts.head(top_k).index.tolist())

# We need genres for train history: use recs file as movie metadata source
movie_meta = recs[["movieId", "genres", "title"]].drop_duplicates("movieId")

# Merge train history with metadata
train_m = train.merge(movie_meta, on="movieId", how="left")

rows = []
for u in users:
    u_hist = train_m[train_m["userId"] == u].copy()
    liked = u_hist[u_hist["rating"] >= 4.0]

    interest_genres = top_genres(liked["genres"], top_k=3)
    avg_rating = round(float(u_hist["rating"].mean()), 3)
    n_ratings = int(len(u_hist))

    # Top-3 recommendations for this user
    u_recs = recs[(recs["userId"] == u) & (recs["rank"] <= 3)].sort_values("rank")

    for _, r in u_recs.iterrows():
        rows.append({
            "userId": u,
            "num_train_ratings": n_ratings,
            "avg_train_rating": avg_rating,
            "user_top_genres_from_liked": interest_genres,
            "rec_rank": int(r["rank"]),
            "rec_title": r["title"],
            "rec_genres": r["genres"],
            "rec_score": float(r["score"])
        })

out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_PATH, index=False)

print("💾 Saved:", OUT_PATH)
print("✅ Done. This CSV is a simple, professor-friendly example output.")
