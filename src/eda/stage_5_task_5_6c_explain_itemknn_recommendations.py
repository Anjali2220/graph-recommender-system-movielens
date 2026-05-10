"""
STAGE 5 — Baseline Recommender System
TASK 5.6c — Explain ItemKNN Recommendations (Qualitative Case Study)

What it does:
- Picks a random user (optionally only users with >= MIN_HISTORY ratings in train)
- Shows user's top-rated movies from TRAIN
- Loads Top-N recommendations (human readable) produced earlier
- Explains each recommendation using Top contributing items:
    contribution = similarity * user's rating
- Summarizes genre overlap:
    "User likes: ..."
    "Recommendations are mostly: ..."

Outputs:
- outputs/baseline/itemknn_model/user_case_study_user_<userId>.csv
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

# =====================================================
# 1) Paths
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent

TRAIN_PATH = PROJECT_DIR / "data" / "processed" / "train_per_user.csv"

MODEL_DIR = PROJECT_DIR / "outputs" / "baseline" / "itemknn_model"
CONFIG_PATH = MODEL_DIR / "config.json"
NEIGHBORS_PATH = MODEL_DIR / "item_neighbors.npy"
SIMS_PATH = MODEL_DIR / "item_sims.npy"
USER_MAP_PATH = MODEL_DIR / "user_id_mapping.csv"
ITEM_MAP_PATH = MODEL_DIR / "item_id_mapping.csv"

# Human-readable recommendations with title+genres (the file you already created)
RECS_HR_PATH = MODEL_DIR / "top5_itemknn_with_titles_genres.csv"

# Movie metadata (title+genres) from processed file
MOVIES_META_PATH = PROJECT_DIR / "data" / "processed" / "ratings_clean.csv"

# Output folder
OUT_DIR = MODEL_DIR / "case_studies"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 2) Settings
# =====================================================
TOP_N = 5                # explain top-5 recs
TOP_CONTRIB = 3          # show top 3 "reasons" per recommended movie
MIN_HISTORY = 30         # only pick a random user with at least this many train ratings (better demo)
RANDOM_SEED = 42         # reproducible randomness
TOP_GENRES_SHOW = 5      # show top 5 genres

# =====================================================
# 3) Load artifacts
# =====================================================
print("📥 Loading model artifacts...")

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

neighbors = np.load(NEIGHBORS_PATH)  # (num_items, K)
sims = np.load(SIMS_PATH)            # (num_items, K)

user_map = pd.read_csv(USER_MAP_PATH)
item_map = pd.read_csv(ITEM_MAP_PATH)

userId_to_row = dict(zip(user_map["userId"].values, user_map["user_row"].values))
movieId_to_col = dict(zip(item_map["movieId"].values, item_map["item_col"].values))
col_to_movieId = item_map.sort_values("item_col")["movieId"].values

print("✅ num_users_train:", config["num_users_train"])
print("✅ num_items_train:", config["num_items_train"])
print("✅ K:", config["K"])

# =====================================================
# 4) Load train history + metadata
# =====================================================
print("\n📥 Loading train history...")
train = pd.read_csv(TRAIN_PATH, usecols=["userId", "movieId", "rating"])
print("Train rows:", train.shape)

print("\n📥 Loading movie metadata (title, genres)...")
movies_meta = pd.read_csv(MOVIES_META_PATH, usecols=["movieId", "title", "genres"]).drop_duplicates("movieId")
movies_meta = movies_meta.set_index("movieId")
print("Unique movies metadata:", movies_meta.shape)

# =====================================================
# 5) Pick a random user (with enough history)
# =====================================================
print("\n🎲 Selecting a random user (with enough history)...")
np.random.seed(RANDOM_SEED)

user_counts = train.groupby("userId").size()
eligible_users = user_counts[user_counts >= MIN_HISTORY].index.to_numpy()

if len(eligible_users) == 0:
    raise ValueError(f"No users found with >= {MIN_HISTORY} interactions in train. Reduce MIN_HISTORY.")

userId = int(np.random.choice(eligible_users))
print("✅ Selected userId:", userId, "| train interactions:", int(user_counts[userId]))

# User train history
u_hist = train[train["userId"] == userId].copy()

# Add title/genres to history
u_hist["title"] = u_hist["movieId"].map(lambda x: movies_meta.at[x, "title"] if x in movies_meta.index else None)
u_hist["genres"] = u_hist["movieId"].map(lambda x: movies_meta.at[x, "genres"] if x in movies_meta.index else None)

# =====================================================
# 6) Load user recommendations (Top-N) from human-readable file
# =====================================================
print("\n📥 Loading human-readable recommendations...")
recs = pd.read_csv(RECS_HR_PATH)

u_recs = recs[recs["userId"] == userId].copy()
u_recs = u_recs.sort_values("rank").head(TOP_N)

if u_recs.empty:
    raise ValueError(f"No recommendations found for userId={userId} in {RECS_HR_PATH}. "
                     f"Check that your file contains this user or regenerate Top-N.")

print("✅ Found recommendations:", len(u_recs))

# =====================================================
# 7) Genre summary (Upgrade B)
# =====================================================
def top_genres_from_rows(df, top_k=5):
    all_genres = []
    for g in df["genres"].dropna().astype(str).tolist():
        all_genres.extend([x.strip() for x in g.split("|") if x.strip()])
    if not all_genres:
        return []
    return [g for g, _ in Counter(all_genres).most_common(top_k)]

# Define "liked" history as rating >= 4.0 (common threshold)
liked_hist = u_hist[u_hist["rating"] >= 4.0].copy()
user_top_genres = top_genres_from_rows(liked_hist, top_k=TOP_GENRES_SHOW)

recs_top_genres = top_genres_from_rows(u_recs, top_k=TOP_GENRES_SHOW)

# =====================================================
# 8) Explain each recommendation by top contributing items (Upgrade A)
# =====================================================

# Precompute user history mapped to item_col (only those present in item mapping)
u_hist["item_col"] = u_hist["movieId"].map(movieId_to_col)
u_hist = u_hist.dropna(subset=["item_col"]).copy()
u_hist["item_col"] = u_hist["item_col"].astype(np.int32)

# Build a dict: item_col -> user rating for fast lookup
user_item_rating = dict(zip(u_hist["item_col"].values, u_hist["rating"].values))

# We'll use liked history first for explanations; if too small, use full history.
explain_pool = liked_hist.copy()
explain_pool["item_col"] = explain_pool["movieId"].map(movieId_to_col)
explain_pool = explain_pool.dropna(subset=["item_col"]).copy()
explain_pool["item_col"] = explain_pool["item_col"].astype(np.int32)

if len(explain_pool) < 5:
    explain_pool = u_hist.copy()

explain_items = explain_pool["item_col"].values
explain_ratings = explain_pool["rating"].values

rows_out = []

# Header section row (easy to read in CSV)
rows_out.append({
    "section": "USER_GENRE_PROFILE",
    "userId": userId,
    "user_likes_genres": ", ".join(user_top_genres) if user_top_genres else "N/A",
    "recs_genres": ", ".join(recs_top_genres) if recs_top_genres else "N/A"
})

# Show top-rated movies in train
top_rated = u_hist.sort_values("rating", ascending=False).head(10)
for i, r in top_rated.iterrows():
    rows_out.append({
        "section": "TOP_RATED_HISTORY",
        "userId": userId,
        "history_rank": int(top_rated.index.get_loc(i) + 1),
        "history_movieId": int(r["movieId"]),
        "history_title": r["title"],
        "history_genres": r["genres"],
        "history_rating": float(r["rating"])
    })

# Explain recommendations
for _, rec_row in u_recs.iterrows():
    rec_movieId = int(rec_row["movieId"])
    rec_title = rec_row["title"]
    rec_genres = rec_row["genres"]
    rec_score = float(rec_row["score"])

    rec_col = movieId_to_col.get(rec_movieId, None)
    if rec_col is None:
        # if movie isn't in training mapping (rare), can't explain via neighbors
        rows_out.append({
            "section": "RECOMMENDATION",
            "userId": userId,
            "rank": int(rec_row["rank"]),
            "rec_movieId": rec_movieId,
            "rec_title": rec_title,
            "rec_genres": rec_genres,
            "rec_score": rec_score,
            "note": "Movie not in training mapping; cannot compute neighbor-based explanation."
        })
        continue

    # For each history item, check if rec_col is in its neighbor list
    contribs = []
    for hist_col, hist_rating in zip(explain_items, explain_ratings):
        neigh = neighbors[hist_col]  # (K,)
        simv = sims[hist_col]        # (K,)

        # find rec_col in neigh
        # (K is 100, so linear search is fine)
        matches = np.where(neigh == rec_col)[0]
        if matches.size > 0:
            sim = float(simv[matches[0]])
            contribution = sim * float(hist_rating)
            hist_movieId = int(col_to_movieId[hist_col])
            hist_title = movies_meta.at[hist_movieId, "title"] if hist_movieId in movies_meta.index else None
            hist_genres = movies_meta.at[hist_movieId, "genres"] if hist_movieId in movies_meta.index else None

            contribs.append((contribution, sim, float(hist_rating), hist_movieId, hist_title, hist_genres))

    # sort by contribution desc
    contribs.sort(key=lambda x: x[0], reverse=True)
    top_contribs = contribs[:TOP_CONTRIB]

    # Add the recommendation row
    rows_out.append({
        "section": "RECOMMENDATION",
        "userId": userId,
        "rank": int(rec_row["rank"]),
        "rec_movieId": rec_movieId,
        "rec_title": rec_title,
        "rec_genres": rec_genres,
        "rec_score": rec_score
    })

    # Add explanation rows
    if not top_contribs:
        rows_out.append({
            "section": "EXPLANATION",
            "userId": userId,
            "rank": int(rec_row["rank"]),
            "explain": "No direct neighbor contribution found from the selected history pool (try increasing K or using full history)."
        })
    else:
        for j, (contribution, sim, hist_rating, h_mid, h_title, h_genres) in enumerate(top_contribs, start=1):
            rows_out.append({
                "section": "EXPLANATION",
                "userId": userId,
                "rank": int(rec_row["rank"]),
                "reason_rank": j,
                "because_movieId": h_mid,
                "because_title": h_title,
                "because_genres": h_genres,
                "similarity": sim,
                "user_rating": hist_rating,
                "contribution": contribution
            })

# =====================================================
# 9) Save case study CSV
# =====================================================
out_path = OUT_DIR / f"user_case_study_user_{userId}.csv"
pd.DataFrame(rows_out).to_csv(out_path, index=False)

print("\n💾 Saved case study file:")
print(" -", out_path)

print("\n✅ Done. Open this CSV and you’ll see:")
print("1) User genre profile")
print("2) Top-rated history")
print("3) Top-5 recommendations + top-3 reasons each")
