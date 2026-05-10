import os
import pandas as pd


def leave_two_out_time_split(
    df: pd.DataFrame,
    user_col: str = "userId",
    item_col: str = "movieId",
    rating_col: str = "rating",
    time_col: str = "timestamp",
    min_interactions: int = 3,
):
    """
    Per-user time-based leave-two-out split:
      - test: last interaction (most recent)
      - val : second last
      - train: everything before those

    Users with < min_interactions are dropped (need at least 1 train + 1 val + 1 test).
    """

    required = {user_col, item_col, rating_col, time_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input df: {missing}")

    data = df[[user_col, item_col, rating_col, time_col]].copy()

    # Ensure timestamp sortable
    if data[time_col].dtype == "object":
        # try numeric first; else datetime
        try:
            data[time_col] = pd.to_numeric(data[time_col], errors="raise")
        except Exception:
            data[time_col] = pd.to_datetime(data[time_col], errors="coerce")

    # Sort by user then time
    data = data.sort_values([user_col, time_col], kind="mergesort")

    # Filter users with too few interactions
    counts = data.groupby(user_col).size()
    keep_users = counts[counts >= min_interactions].index
    data = data[data[user_col].isin(keep_users)].copy()

    # Rank within each user
    data["_rank"] = data.groupby(user_col).cumcount() + 1
    data["_n"] = data.groupby(user_col)[user_col].transform("size")

    # Split
    test = data[data["_rank"] == data["_n"]].drop(columns=["_rank", "_n"])
    val = data[data["_rank"] == (data["_n"] - 1)].drop(columns=["_rank", "_n"])
    train = data[data["_rank"] <= (data["_n"] - 2)].drop(columns=["_rank", "_n"])

    return train, val, test


def sanity_checks(train_df, val_df, test_df, user_col="userId", item_col="movieId"):
    """
    Basic checks:
      - Each user has exactly 1 val and 1 test
      - No overlap between splits for (user,item)
    """
    print("\n--- Sanity checks ---")
    print("Unique users -> train:", train_df[user_col].nunique())
    print("Unique users -> val  :", val_df[user_col].nunique())
    print("Unique users -> test :", test_df[user_col].nunique())

    val_counts = val_df.groupby(user_col).size()
    test_counts = test_df.groupby(user_col).size()
    print("Val per-user min/max :", int(val_counts.min()), int(val_counts.max()))
    print("Test per-user min/max:", int(test_counts.min()), int(test_counts.max()))

    # Pair overlaps
    train_pairs = set(zip(train_df[user_col], train_df[item_col]))
    val_pairs = set(zip(val_df[user_col], val_df[item_col]))
    test_pairs = set(zip(test_df[user_col], test_df[item_col]))

    print("Overlap train-val pairs :", len(train_pairs.intersection(val_pairs)))
    print("Overlap train-test pairs:", len(train_pairs.intersection(test_pairs)))
    print("Overlap val-test pairs  :", len(val_pairs.intersection(test_pairs)))
    print("--- End checks ---\n")


def main():
    # =========================
    # Paths (matches your project)
    # =========================
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

    # Use your cleaned ratings file (choose the one you trust most)
    # From your screenshot, you have: ratings_clean.csv (and others)
    INPUT_PATH = os.path.join(PROCESSED_DIR, "ratings_clean.csv")

    # Output split files (keep them inside data/processed)
    OUT_TRAIN = os.path.join(PROCESSED_DIR, "train.csv")
    OUT_VAL = os.path.join(PROCESSED_DIR, "val.csv")
    OUT_TEST = os.path.join(PROCESSED_DIR, "test.csv")

    print("Project root  :", PROJECT_ROOT)
    print("Input file    :", INPUT_PATH)
    print("Output train  :", OUT_TRAIN)
    print("Output val    :", OUT_VAL)
    print("Output test   :", OUT_TEST)

    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}\n"
            "Check file name inside data/processed/ (ratings_clean.csv etc.)"
        )

    # =========================
    # Load data
    # =========================
    df = pd.read_csv(INPUT_PATH)

    # =========================
    # Split
    # =========================
    train_df, val_df, test_df = leave_two_out_time_split(
        df,
        user_col="userId",
        item_col="movieId",
        rating_col="rating",
        time_col="timestamp",
        min_interactions=3,
    )

    # =========================
    # Save
    # =========================
    train_df.to_csv(OUT_TRAIN, index=False)
    val_df.to_csv(OUT_VAL, index=False)
    test_df.to_csv(OUT_TEST, index=False)

    print("\n✅ Split saved successfully!")
    print("Train rows:", len(train_df))
    print("Val rows  :", len(val_df))
    print("Test rows :", len(test_df))

    # =========================
    # Sanity checks
    # =========================
    sanity_checks(train_df, val_df, test_df)


if __name__ == "__main__":
    main()