"""
STAGE 5 — Baseline Recommender System
TASK 5.6d — Convert messy case study CSV into a CLEAN professor-friendly table

Fix:
- Automatically finds the most recent file:
  outputs/baseline/itemknn_model/case_studies/user_case_study_user_*.csv
- Creates a CLEAN output file with one row per recommendation
"""

import pandas as pd
from pathlib import Path

# =====================================================
# Paths
# =====================================================
BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent.parent  # project root

CASE_DIR = PROJECT_DIR / "outputs" / "baseline" / "itemknn_model" / "case_studies"
CASE_DIR.mkdir(parents=True, exist_ok=True)

# =====================================================
# 1) Find the latest case study file
# =====================================================
case_files = sorted(CASE_DIR.glob("user_case_study_user_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)

if len(case_files) == 0:
    raise FileNotFoundError(
        f"❌ No case study CSV found in: {CASE_DIR}\n"
        f"Make sure you ran TASK 5.6c first (explain recommendations script)."
    )

INPUT_PATH = case_files[0]  # most recent file
print("✅ Using latest case study file:\n -", INPUT_PATH)

# Output name
OUT_PATH = CASE_DIR / (INPUT_PATH.stem + "_CLEAN.csv")

# =====================================================
# 2) Load
# =====================================================
df = pd.read_csv(INPUT_PATH)

# =====================================================
# 3) Extract profile row
# =====================================================
profile = df[df["section"] == "USER_GENRE_PROFILE"].head(1)
if len(profile) == 0:
    raise ValueError("❌ No USER_GENRE_PROFILE row found in the input file.")

userId = int(profile["userId"].values[0])
user_likes_genres = profile.get("user_likes_genres", pd.Series(["N/A"])).values[0]
recs_genres = profile.get("recs_genres", pd.Series(["N/A"])).values[0]

# =====================================================
# 4) Extract recommendations and explanations
# =====================================================
recs = df[df["section"] == "RECOMMENDATION"].copy()
recs = recs.sort_values("rank")

exps = df[df["section"] == "EXPLANATION"].copy()

# =====================================================
# 5) Build clean output table
# =====================================================
rows = []

for _, r in recs.iterrows():
    rank = int(r["rank"])

    rec_title = r.get("rec_title", r.get("title", None))
    rec_genres = r.get("rec_genres", r.get("genres", None))
    rec_score = r.get("rec_score", r.get("score", None))

    # Explanations for this recommendation rank
    e = exps[exps["rank"] == rank].sort_values("reason_rank").head(3)

    # Fill up to 3 reasons
    reasons = []
    for i in range(1, 4):
        if len(e) >= i:
            ei = e.iloc[i - 1]
            reasons.extend([
                ei.get("because_title", None),
                ei.get("similarity", None),
                ei.get("user_rating", None),
                ei.get("contribution", None),
            ])
        else:
            reasons.extend([None, None, None, None])

    rows.append({
        "userId": userId,
        "user_likes_genres": user_likes_genres,
        "recs_genres_summary": recs_genres,
        "rank": rank,
        "recommended_title": rec_title,
        "recommended_genres": rec_genres,
        "recommended_score": rec_score,

        "reason1_title": reasons[0],
        "sim1": reasons[1],
        "rating1": reasons[2],
        "contribution1": reasons[3],

        "reason2_title": reasons[4],
        "sim2": reasons[5],
        "rating2": reasons[6],
        "contribution2": reasons[7],

        "reason3_title": reasons[8],
        "sim3": reasons[9],
        "rating3": reasons[10],
        "contribution3": reasons[11],
    })

clean_df = pd.DataFrame(rows)

# =====================================================
# 6) Save
# =====================================================
clean_df.to_csv(OUT_PATH, index=False)

print("\n✅ CLEAN case study saved successfully:")
print(" -", OUT_PATH)
print("\nOpen this CLEAN CSV — it will be easy to read and show your professor.")
