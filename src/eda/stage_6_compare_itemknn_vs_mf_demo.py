import os
import numpy as np
import pandas as pd
import torch

from src.eda.mf_model import MFModel


def safe_movie_lookup(movies_df: pd.DataFrame, movie_id: int):
    row = movies_df.loc[movies_df["movieId"] == movie_id]
    if row.empty:
        return "Unknown", "Unknown"
    r = row.iloc[0]
    return str(r.get("title", "Unknown")), str(r.get("genres", "Unknown"))


def load_item_ids_robust(path: str):
    """
    Try to load item_ids saved in different possible formats.
    Returns a 1D numpy array of ints, or raises.
    """
    # 1) Normal npy
    try:
        arr = np.load(path)
        return np.asarray(arr).reshape(-1)
    except Exception:
        pass

    # 2) NPY with pickle objects
    try:
        arr = np.load(path, allow_pickle=True)
        return np.asarray(arr).reshape(-1)
    except Exception:
        pass

    # 3) It's actually a text file but named .npy (common mistake)
    try:
        arr = np.loadtxt(path, dtype=np.int64, delimiter=",")
        return np.asarray(arr).reshape(-1)
    except Exception:
        pass

    # 4) Try reading as CSV (single column)
    try:
        df = pd.read_csv(path, header=None)
        return df.iloc[:, 0].astype(np.int64).values.reshape(-1)
    except Exception as e:
        raise ValueError(f"Could not load item_ids from {path} using any method. Last error: {e}")


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # ----------------------------
    # Paths
    # ----------------------------
    ITEMKNN_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model")
    MF_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")
    TUNE_DIR = os.path.join(MF_DIR, "tuning")

    # Best MF checkpoint
    MF_MODEL_PATH = os.path.join(TUNE_DIR, "mf_dim64_lr0.001_ep5.pt")

    # MF artifacts
    MF_PAIRS_PATH = os.path.join(MF_DIR, "train_pairs.npy")
    MF_USER_IDS_PATH = os.path.join(MF_DIR, "user_ids.npy")
    MF_ITEM_IDS_PATH = os.path.join(MF_DIR, "item_ids.npy")

    # ItemKNN artifacts (your filenames)
    KNN_INDICES_PATH = os.path.join(ITEMKNN_DIR, "knn_indices.npy")
    KNN_SIMS_PATH = os.path.join(ITEMKNN_DIR, "knn_sims.npy")
    KNN_ITEM_IDS_PATH = os.path.join(ITEMKNN_DIR, "item_ids.npy")  # problematic file

    MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv")

    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "comparison_demo")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_MF = os.path.join(OUT_DIR, "mf_top5_same_users.csv")
    OUT_KNN = os.path.join(OUT_DIR, "itemknn_top5_same_users.csv")

    # ----------------------------
    # Sanity checks
    # ----------------------------
    required = [
        MF_MODEL_PATH, MF_PAIRS_PATH, MF_USER_IDS_PATH, MF_ITEM_IDS_PATH,
        KNN_INDICES_PATH, KNN_SIMS_PATH, MOVIES_PATH
    ]
    for p in required:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file:\n{p}")

    # ----------------------------
    # Load metadata
    # ----------------------------
    movies = pd.read_csv(MOVIES_PATH)

    # ----------------------------
    # Load MF artifacts (authoritative indexing)
    # ----------------------------
    pairs = np.load(MF_PAIRS_PATH)
    user_ids = np.load(MF_USER_IDS_PATH)
    mf_item_ids = np.load(MF_ITEM_IDS_PATH)

    n_users = len(user_ids)
    n_items = len(mf_item_ids)

    # Build train history sets (exclude seen)
    user_pos_sets = [set() for _ in range(n_users)]
    for u, i in pairs:
        user_pos_sets[int(u)].add(int(i))

    # ----------------------------
    # Load MF embeddings
    # ----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = MFModel.load(
        MF_MODEL_PATH,
        n_users=n_users,
        n_items=n_items,
        emb_dim=64,
        map_location=device
    )
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        user_emb = model.user_emb.weight.detach().cpu().numpy()
        item_emb = model.item_emb.weight.detach().cpu().numpy()

    # ----------------------------
    # Load ItemKNN arrays
    # ----------------------------
    knn_indices = np.load(KNN_INDICES_PATH).astype(np.int32)
    knn_sims = np.load(KNN_SIMS_PATH).astype(np.float32)

    # Try to load ItemKNN item_ids, but if it fails, fallback to MF item_ids (if shape matches)
    knn_item_ids = None
    if os.path.exists(KNN_ITEM_IDS_PATH):
        try:
            knn_item_ids = load_item_ids_robust(KNN_ITEM_IDS_PATH)
            knn_item_ids = knn_item_ids.astype(np.int64)
            print("Loaded ItemKNN item_ids from:", KNN_ITEM_IDS_PATH)
        except Exception as e:
            print("⚠️ Could not load ItemKNN item_ids.npy (corrupt or non-npy).")
            print("Reason:", e)

    if knn_item_ids is None:
        # fallback ONLY if sizes match
        if knn_indices.shape[0] == n_items:
            knn_item_ids = mf_item_ids.astype(np.int64)
            print("✅ Fallback: using MF item_ids mapping for ItemKNN (same n_items).")
        else:
            raise RuntimeError(
                f"Cannot map ItemKNN item indices to movieIds.\n"
                f"ItemKNN n_items={knn_indices.shape[0]} but MF n_items={n_items}.\n"
                f"Please provide a valid item mapping file for ItemKNN."
            )

    # ----------------------------
    # Pick SAME 5 users (reproducible)
    # ----------------------------
    rng = np.random.default_rng(42)
    demo_user_indices = rng.choice(n_users, size=5, replace=False)

    mf_rows, knn_rows = [], []

    print("\nGenerating Top-5 recommendations for SAME 5 users...")
    for u_idx in demo_user_indices:
        print(f"user_idx={u_idx} -> userId={int(user_ids[u_idx])}")

    # ----------------------------
    # Generate recs
    # ----------------------------
    for u_idx in demo_user_indices:
        user_id = int(user_ids[u_idx])
        seen = user_pos_sets[u_idx]

        # ===== MF =====
        mf_scores = item_emb @ user_emb[u_idx]
        if seen:
            mf_scores[list(seen)] = -np.inf
        top5_mf_idx = np.argsort(-mf_scores)[:5]

        for rank, item_idx in enumerate(top5_mf_idx, start=1):
            movie_id = int(mf_item_ids[item_idx])
            title, genres = safe_movie_lookup(movies, movie_id)
            mf_rows.append({
                "userId": user_id,
                "rank": rank,
                "movieId": movie_id,
                "title": title,
                "genres": genres
            })

        # ===== ItemKNN =====
        itemknn_scores = np.zeros(n_items, dtype=np.float32)
        for it in seen:
            neigh = knn_indices[it]
            sims = knn_sims[it]
            itemknn_scores[neigh] += sims
        if seen:
            itemknn_scores[list(seen)] = -np.inf
        top5_knn_idx = np.argsort(-itemknn_scores)[:5]

        for rank, item_idx in enumerate(top5_knn_idx, start=1):
            movie_id = int(knn_item_ids[item_idx])
            title, genres = safe_movie_lookup(movies, movie_id)
            knn_rows.append({
                "userId": user_id,
                "rank": rank,
                "movieId": movie_id,
                "title": title,
                "genres": genres
            })

    # ----------------------------
    # Save
    # ----------------------------
    mf_df = pd.DataFrame(mf_rows).sort_values(["userId", "rank"])
    knn_df = pd.DataFrame(knn_rows).sort_values(["userId", "rank"])

    mf_df.to_csv(OUT_MF, index=False)
    knn_df.to_csv(OUT_KNN, index=False)

    print("\n✅ Saved outputs:")
    print("MF      :", OUT_MF)
    print("ItemKNN :", OUT_KNN)

    print("\nPreview MF:")
    print(mf_df.head(10))

    print("\nPreview ItemKNN:")
    print(knn_df.head(10))


if __name__ == "__main__":
    main()
