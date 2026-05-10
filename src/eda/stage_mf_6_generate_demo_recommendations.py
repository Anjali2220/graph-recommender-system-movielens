import os
import numpy as np
import pandas as pd
import torch

from src.eda.mf_model import MFModel


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # ----------------------------
    # Paths
    # ----------------------------
    MF_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "Bmf_model")
    TUNE_DIR = os.path.join(MF_DIR, "tuning")

    MODEL_PATH = os.path.join(TUNE_DIR, "mf_dim64_lr0.001_ep5.pt")

    PAIRS_PATH = os.path.join(MF_DIR, "train_pairs.npy")
    USER_IDS_PATH = os.path.join(MF_DIR, "user_ids.npy")
    ITEM_IDS_PATH = os.path.join(MF_DIR, "item_ids.npy")

    MOVIES_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "movies.csv")

    OUT_DIR = os.path.join(MF_DIR, "demo_output")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_PATH = os.path.join(OUT_DIR, "top10_recommendations_mf_with_titles.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading MF model...")

    # ----------------------------
    # Load MF artifacts
    # ----------------------------
    pairs = np.load(PAIRS_PATH)
    user_ids = np.load(USER_IDS_PATH)
    item_ids = np.load(ITEM_IDS_PATH)

    n_users = len(user_ids)
    n_items = len(item_ids)

    model, _ = MFModel.load(
        MODEL_PATH,
        n_users=n_users,
        n_items=n_items,
        emb_dim=64,
        map_location=device,
    )
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        user_emb = model.user_emb.weight.detach().cpu().numpy()
        item_emb = model.item_emb.weight.detach().cpu().numpy()

    # ----------------------------
    # Build train history (exclude seen items)
    # ----------------------------
    user_pos_sets = [set() for _ in range(n_users)]
    for u, i in pairs:
        user_pos_sets[int(u)].add(int(i))

    # ----------------------------
    # Load movie titles
    # ----------------------------
    movies = pd.read_csv(MOVIES_PATH)

    # ----------------------------
    # Select demo users
    # ----------------------------
    demo_user_indices = np.random.choice(n_users, size=10, replace=False)

    rows = []

    print("Generating recommendations...")

    for u_idx in demo_user_indices:
        uvec = user_emb[u_idx]

        scores = item_emb @ uvec  # (n_items,)
        seen_items = user_pos_sets[u_idx]

        # remove seen
        scores[list(seen_items)] = -np.inf

        top10_idx = np.argsort(-scores)[:10]

        original_user_id = user_ids[u_idx]

        for rank, item_idx in enumerate(top10_idx, start=1):
            original_movie_id = item_ids[item_idx]

            movie_info = movies[movies["movieId"] == original_movie_id]

            if len(movie_info) > 0:
                title = movie_info.iloc[0]["title"]
                genres = movie_info.iloc[0]["genres"]
            else:
                title = "Unknown"
                genres = "Unknown"

            rows.append({
                "userId": original_user_id,
                "rank": rank,
                "movieId": original_movie_id,
                "title": title,
                "genres": genres,
                "score": scores[item_idx],
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_PATH, index=False)

    print("\n✅ Saved MF Top-10 recommendations to:")
    print(OUT_PATH)

    print("\nPreview:")
    print(df_out.head(20))


if __name__ == "__main__":
    main()
