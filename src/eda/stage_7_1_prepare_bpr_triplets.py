import os
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def main():
    project_root = find_project_root(Path(__file__).parent)

    train_csv = project_root / "data" / "processed" / "train.csv"
    edges_dir = project_root / "data" / "processed" / "graph_edges"
    out_dir = project_root / "outputs" / "gnn" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    user_map_path = edges_dir / "user_id_mapping.csv"
    movie_map_path = edges_dir / "movie_id_mapping.csv"

    print("============================================")
    print("✅ TASK 7.1 — Prepare BPR Triplets (GNN)")
    print("============================================")

    # Load train
    df = pd.read_csv(train_csv)

    # Load mappings
    user_map = pd.read_csv(user_map_path)
    movie_map = pd.read_csv(movie_map_path)

    # Merge to get user_idx and movie_idx
    df = df.merge(user_map, on="userId", how="inner")
    df = df.merge(movie_map, on="movieId", how="inner")

    # Keep only positive interactions
    df_pos = df[df["rating"] >= 4.0][["user_idx", "movie_idx"]].copy()

    df_pos["user_idx"] = df_pos["user_idx"].astype(int)
    df_pos["movie_idx"] = df_pos["movie_idx"].astype(int)

    # Remove duplicates
    df_pos = df_pos.drop_duplicates(["user_idx", "movie_idx"])

    n_users = user_map["user_idx"].max() + 1
    n_items = movie_map["movie_idx"].max() + 1

    print(f"Users: {n_users}")
    print(f"Items: {n_items}")
    print(f"Positive interactions: {len(df_pos)}")

    # Build user -> positive list
    user_pos = [[] for _ in range(n_users)]
    for row in df_pos.itertuples():
        user_pos[row.user_idx].append(row.movie_idx)

    # Convert to CSR style arrays
    pos_counts = np.array([len(lst) for lst in user_pos])
    user_ptr = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(pos_counts, out=user_ptr[1:])

    total_pos = user_ptr[-1]
    pos_items_flat = np.zeros(total_pos, dtype=np.int32)

    idx = 0
    for u in range(n_users):
        items = user_pos[u]
        if items:
            k = len(items)
            pos_items_flat[idx:idx + k] = items
            idx += k

    # Save
    np.savez_compressed(
        out_dir / "train_pos_sets.npz",
        n_users=n_users,
        n_items=n_items,
        user_ptr=user_ptr,
        pos_items_flat=pos_items_flat,
        pos_counts=pos_counts,
    )

    metadata = {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "positive_interactions": int(len(df_pos)),
        "artifact": "train_pos_sets.npz"
    }

    with open(out_dir / "train_pos_sets_info.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("\n✅ Saved:")
    print(out_dir / "train_pos_sets.npz")
    print(out_dir / "train_pos_sets_info.json")
    print("============================================")
    print("TASK 7.1 COMPLETE ✅")


if __name__ == "__main__":
    main()