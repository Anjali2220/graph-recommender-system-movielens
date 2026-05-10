import os
import numpy as np
import pandas as pd
import torch

from src.eda.mf_model import MFModel


# =========================================================
# 🔧 CHANGE THESE VALUES ONLY
# =========================================================
MODEL_TYPE = "itemknn"   # "itemknn" or "mf"
USER_ID = 1           # change to any userId you want
TOP_N = 5
# =========================================================


def safe_movie_lookup(movies_df: pd.DataFrame, movie_id: int):
    row = movies_df.loc[movies_df["movieId"] == movie_id]
    if row.empty:
        return "Unknown", "Unknown"
    r = row.iloc[0]
    return str(r.get("title", "Unknown")), str(r.get("genres", "Unknown"))


def build_user_pos_sets(pairs: np.ndarray, n_users: int):
    user_pos = [set() for _ in range(n_users)]
    for u, i in pairs:
        user_pos[int(u)].add(int(i))
    return user_pos


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # ----------------------------
    # Paths
    # ----------------------------
    itemknn_dir = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model")
    mf_dir = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")
    tuning_dir = os.path.join(mf_dir, "tuning")

    pairs_path = os.path.join(mf_dir, "train_pairs.npy")
    user_ids_path = os.path.join(mf_dir, "user_ids.npy")
    item_ids_path = os.path.join(mf_dir, "item_ids.npy")

    knn_indices_path = os.path.join(itemknn_dir, "knn_indices.npy")
    knn_sims_path = os.path.join(itemknn_dir, "knn_sims.npy")

    mf_model_path = os.path.join(tuning_dir, "mf_dim64_lr0.001_ep5.pt")

    movies_path = os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv")

    # ----------------------------
    # Load shared data
    # ----------------------------
    pairs = np.load(pairs_path)
    user_ids = np.load(user_ids_path)
    item_ids = np.load(item_ids_path)
    movies = pd.read_csv(movies_path)

    n_users = len(user_ids)
    n_items = len(item_ids)

    user2idx = {int(u): i for i, u in enumerate(user_ids)}

    if USER_ID not in user2idx:
        raise ValueError(f"userId {USER_ID} not found.")

    u_idx = user2idx[USER_ID]
    user_pos_sets = build_user_pos_sets(pairs, n_users)
    seen = user_pos_sets[u_idx]

    print("\n====================================================")
    print(f"Model Selected : {MODEL_TYPE.upper()}")
    print(f"userId         : {USER_ID}")
    print(f"Top-N          : {TOP_N}")
    print("====================================================\n")

    # =====================================================
    # 🔵 MF MODEL
    # =====================================================
    if MODEL_TYPE == "mf":

        device = "cuda" if torch.cuda.is_available() else "cpu"

        model, _ = MFModel.load(
            mf_model_path,
            n_users=n_users,
            n_items=n_items,
            emb_dim=64,
            map_location=device
        )
        model = model.to(device).eval()

        with torch.no_grad():
            user_emb = model.user_emb.weight.detach().cpu().numpy()
            item_emb = model.item_emb.weight.detach().cpu().numpy()

        scores = item_emb @ user_emb[u_idx]

        if seen:
            scores[list(seen)] = -np.inf

        top_idx = np.argsort(-scores)[:TOP_N]

    # =====================================================
    # 🟢 ITEMKNN MODEL
    # =====================================================
    elif MODEL_TYPE == "itemknn":

        knn_indices = np.load(knn_indices_path).astype(np.int32)
        knn_sims = np.load(knn_sims_path).astype(np.float32)

        scores = np.zeros(n_items, dtype=np.float32)

        for it in seen:
            scores[knn_indices[it]] += knn_sims[it]

        if seen:
            scores[list(seen)] = -np.inf

        top_idx = np.argsort(-scores)[:TOP_N]

    else:
        raise ValueError("MODEL_TYPE must be 'itemknn' or 'mf'")

    # =====================================================
    # Print Recommendations
    # =====================================================
    print("Top Recommendations:\n")

    for rank, idx in enumerate(top_idx, start=1):
        movie_id = int(item_ids[idx])
        title, genres = safe_movie_lookup(movies, movie_id)
        print(f"{rank}. {title}")
        print(f"   Genres : {genres}")
        print(f"   movieId: {movie_id}")
        print("----------------------------------------------------")


if __name__ == "__main__":
    main()
