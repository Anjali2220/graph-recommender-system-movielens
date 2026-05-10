import os
import json
import numpy as np
import pandas as pd


def main():
    # ----------------------------
    # Paths
    # ----------------------------
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "train.csv")

    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_PAIRS = os.path.join(OUT_DIR, "train_pairs.npy")         # shape (N, 2)
    OUT_USER_IDS = os.path.join(OUT_DIR, "user_ids.npy")         # index -> userId
    OUT_ITEM_IDS = os.path.join(OUT_DIR, "item_ids.npy")         # index -> movieId
    OUT_CONFIG = os.path.join(OUT_DIR, "config_mf_data.json")

    # ----------------------------
    # Settings
    # ----------------------------
    RATING_THRESHOLD = 4.0

    print("Project root :", PROJECT_ROOT)
    print("Train file   :", TRAIN_PATH)
    print("Output dir   :", OUT_DIR)
    print("Threshold    :", RATING_THRESHOLD)

    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"train.csv not found at: {TRAIN_PATH}")

    # ----------------------------
    # Load train
    # ----------------------------
    df = pd.read_csv(TRAIN_PATH, usecols=["userId", "movieId", "rating", "timestamp"])
    print("\nLoaded train.csv rows:", len(df))

    # ----------------------------
    # Convert explicit -> implicit positives
    # ----------------------------
    df = df[df["rating"] >= RATING_THRESHOLD].copy()
    print("Positive rows (rating>=threshold):", len(df))

    # Keep latest per (user,movie) to avoid duplicates
    df = df.sort_values("timestamp").drop_duplicates(["userId", "movieId"], keep="last")
    print("After dedupe (userId,movieId):", len(df))

    # ----------------------------
    # Create mappings
    # ----------------------------
    user_ids = df["userId"].unique()
    item_ids = df["movieId"].unique()

    # Make them numpy arrays (stable ordering as they appear)
    user_ids = np.array(user_ids, dtype=np.int64)
    item_ids = np.array(item_ids, dtype=np.int64)

    user2idx = {int(u): i for i, u in enumerate(user_ids)}
    item2idx = {int(m): i for i, m in enumerate(item_ids)}

    n_users = len(user_ids)
    n_items = len(item_ids)

    print("\nMF data summary:")
    print("Users:", n_users)
    print("Items:", n_items)

    # ----------------------------
    # Build training pairs (user_idx, item_idx)
    # ----------------------------
    df["user_idx"] = df["userId"].map(user2idx).astype(np.int32)
    df["item_idx"] = df["movieId"].map(item2idx).astype(np.int32)

    pairs = df[["user_idx", "item_idx"]].to_numpy(dtype=np.int32)
    print("Training pairs shape:", pairs.shape)

    # ----------------------------
    # Save outputs
    # ----------------------------
    np.save(OUT_PAIRS, pairs)
    np.save(OUT_USER_IDS, user_ids)
    np.save(OUT_ITEM_IDS, item_ids)

    config = {
        "stage": "MF-1 Data Preparation",
        "source": "train.csv only (time-based leave-two-out split)",
        "rating_threshold": RATING_THRESHOLD,
        "implicit": True,
        "n_users": int(n_users),
        "n_items": int(n_items),
        "n_pairs": int(pairs.shape[0]),
        "saved_files": {
            "train_pairs": os.path.basename(OUT_PAIRS),
            "user_ids": os.path.basename(OUT_USER_IDS),
            "item_ids": os.path.basename(OUT_ITEM_IDS),
        }
    }

    with open(OUT_CONFIG, "w") as f:
        json.dump(config, f, indent=2)

    print("\n✅ Saved MF-1 prepared data:")
    print(" -", OUT_PAIRS)
    print(" -", OUT_USER_IDS)
    print(" -", OUT_ITEM_IDS)
    print(" -", OUT_CONFIG)

    # ----------------------------
    # Quick preview
    # ----------------------------
    print("\nPreview (first 10 training pairs):")
    print(pairs[:10])


if __name__ == "__main__":
    main()
