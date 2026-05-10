import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import importlib.util


# =========================
# Utils
# =========================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def load_mapping(path: Path, col1: str, col2: str) -> Dict[int, int]:
    df = pd.read_csv(path)
    return dict(zip(df[col1].astype(int), df[col2].astype(int)))


def invert_mapping(d: Dict[int, int]) -> Dict[int, int]:
    return {v: k for k, v in d.items()}


# =========================
# Metadata
# =========================
def load_movies(project_root: Path) -> pd.DataFrame:
    candidates = [
        project_root / "data" / "raw" / "movies.csv",
        project_root / "data" / "raw" / "ml-20m" / "movies.csv",
    ]
    path = find_first_existing(candidates)
    if path is None:
        raise FileNotFoundError("movies.csv not found")

    df = pd.read_csv(path)
    return df[["movieId", "title", "genres"]].copy()


def load_tags(project_root: Path) -> pd.DataFrame:
    candidates = [
        project_root / "data" / "raw" / "tags.csv",
        project_root / "data" / "raw" / "ml-20m" / "tags.csv",
    ]
    path = find_first_existing(candidates)
    if path is None:
        return pd.DataFrame(columns=["movieId", "tags"])

    df = pd.read_csv(path)
    if not {"movieId", "tag"}.issubset(df.columns):
        return pd.DataFrame(columns=["movieId", "tags"])

    df["tag"] = df["tag"].astype(str).str.strip()
    df = df[df["tag"] != ""].copy()

    tag_df = (
        df.groupby("movieId")["tag"]
        .apply(lambda x: ", ".join(x.value_counts().head(5).index.tolist()))
        .reset_index()
    )
    tag_df.columns = ["movieId", "tags"]
    return tag_df


# =========================
# Seen items
# =========================
def load_seen_items(npz_path: Path, user_idx: int) -> np.ndarray:
    data = np.load(npz_path)
    user_ptr = data["user_ptr"].astype(np.int64)
    pos_items_flat = data["pos_items_flat"].astype(np.int64)

    s, e = user_ptr[user_idx], user_ptr[user_idx + 1]
    return pos_items_flat[s:e]


def pick_watched_title(seen_movie_idxs: List[int], movie_idx_to_id: Dict[int, int], movies_df: pd.DataFrame) -> str:
    for movie_idx in seen_movie_idxs:
        movie_id = movie_idx_to_id.get(movie_idx)
        if movie_id is None:
            continue
        row = movies_df[movies_df["movieId"] == movie_id]
        if len(row) > 0:
            return str(row.iloc[0]["title"])
    return "a previously liked movie"


# =========================
# Import LightGCN
# =========================
def import_lightgcn(project_root: Path):
    candidates = [
        project_root / "src" / "eda" / "stage_l3_lightgcn_model.py",
        project_root / "src" / "gnn" / "lightgcn_model.py",
    ]

    errors = []
    for file_path in candidates:
        if file_path.exists():
            spec = importlib.util.spec_from_file_location("lightgcn_module", str(file_path))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "LightGCN") and hasattr(mod, "build_norm_adj"):
                    return mod.LightGCN, mod.build_norm_adj
                errors.append(f"{file_path} exists but missing LightGCN/build_norm_adj")
            except Exception as e:
                errors.append(f"Failed importing {file_path}: {e}")

    raise ImportError("Could not import LightGCN:\n" + "\n".join(errors))


def load_lightgcn_state_robust(model: torch.nn.Module, state: Dict[str, torch.Tensor], n_users: int, n_items: int):
    if "emb.weight" in state:
        W = state["emb.weight"]
        if W.size(0) != (n_users + n_items):
            raise ValueError(f"emb.weight rows {W.size(0)} != n_users+n_items {n_users+n_items}")

        with torch.no_grad():
            model.user_emb.weight.copy_(W[:n_users])
            model.item_emb.weight.copy_(W[n_users:n_users + n_items])
        return

    model.load_state_dict(state, strict=False)


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_ids", type=int, nargs="+", default=[3054, 23349, 28364, 87505, 116229])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--tag", type=str, default="lightgcn_strong_run")

    # must match your lightgcn training
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--K", type=int, default=3)

    args = parser.parse_args()

    project_root = find_project_root(Path(__file__).resolve().parent)

    edge_index_pt = project_root / "outputs" / "gnn" / "lightgcn" / "train_graph" / "edge_index.pt"
    edge_info_json = project_root / "outputs" / "gnn" / "lightgcn" / "train_graph" / "edge_index_info.json"
    pos_npz = project_root / "outputs" / "gnn" / "data" / "train_pos_sets.npz"

    run_dir = project_root / "outputs" / "gnn" / "lightgcn" / "runs" / args.tag
    checkpoint = run_dir / "model_final.pt"

    mapping_dir = project_root / "data" / "processed" / "graph_edges"
    out_dir = project_root / "outputs" / "results"
    ensure_dir(out_dir)
    out_csv = out_dir / "lightgcn_user_recommendations.csv"

    print("============================================")
    print("✅ LightGCN Demo Recommendations")
    print("============================================")
    print(f"edge_index_pt : {edge_index_pt}")
    print(f"edge_info     : {edge_info_json}")
    print(f"checkpoint    : {checkpoint}")
    print(f"out_csv       : {out_csv}")
    print("============================================")

    for p in [edge_index_pt, edge_info_json, pos_npz, checkpoint]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # mappings
    user_map = load_mapping(mapping_dir / "user_id_mapping.csv", "userId", "user_idx")
    movie_map = load_mapping(mapping_dir / "movie_id_mapping.csv", "movieId", "movie_idx")
    movie_idx_to_id = invert_mapping(movie_map)

    # metadata
    movies = load_movies(project_root)
    tags = load_tags(project_root)

    # load graph info
    edge_info = pd.read_json(edge_info_json, typ="series")
    n_users = int(edge_info["n_users"])
    n_items = int(edge_info["n_items"])
    num_nodes = int(edge_info["num_nodes"])

    # import model
    LightGCN, build_norm_adj = import_lightgcn(project_root)

    edge_index = safe_torch_load(edge_index_pt, map_location="cpu")
    A_hat = build_norm_adj(edge_index, num_nodes=num_nodes)

    model = LightGCN(n_users=n_users, n_items=n_items, emb_dim=args.emb_dim, K=args.K)
    state = torch.load(checkpoint, map_location="cpu")
    load_lightgcn_state_robust(model, state, n_users, n_items)
    model.eval()

    print("Running full-graph embedding inference (CPU)...")
    with torch.no_grad():
        z_user, z_movie = model.get_all_embeddings(A_hat)

    z_user = F.normalize(z_user, dim=1).cpu()
    z_movie = F.normalize(z_movie, dim=1).cpu()

    rows = []

    for user_id in args.user_ids:
        if user_id not in user_map:
            print(f"\nUser {user_id} not found in mapping. Skipping.")
            continue

        user_idx = user_map[user_id]
        seen = set(load_seen_items(pos_npz, user_idx).tolist())

        user_vec = z_user[user_idx]
        scores = torch.matmul(z_movie, user_vec)

        if len(seen) > 0:
            seen_tensor = torch.tensor(list(seen), dtype=torch.long)
            scores[seen_tensor] = -1e9

        top_scores, top_idx = torch.topk(scores, args.top_k)
        top_idx = top_idx.cpu().numpy().tolist()
        top_scores = top_scores.cpu().numpy().tolist()

        watched_title = pick_watched_title(list(seen), movie_idx_to_id, movies)

        print("\n----------------------------------")
        print(f"User {user_id}")
        print(f"Because you watched: {watched_title}")
        print("We recommend:")

        for rank, (movie_idx, score) in enumerate(zip(top_idx, top_scores), start=1):
            movie_id = movie_idx_to_id.get(movie_idx, None)

            title = ""
            genres = ""
            tag_text = ""

            if movie_id is not None:
                movie_row = movies[movies["movieId"] == movie_id]
                if len(movie_row) > 0:
                    title = str(movie_row.iloc[0]["title"])
                    genres = str(movie_row.iloc[0]["genres"])

                tag_row = tags[tags["movieId"] == movie_id]
                if len(tag_row) > 0:
                    tag_text = str(tag_row.iloc[0]["tags"])

            print(f"{rank}. {title}")

            rows.append(
                {
                    "userId": user_id,
                    "rank": rank,
                    "movieId": movie_id,
                    "title": title,
                    "genres": genres,
                    "tags": tag_text,
                    "score": float(score),
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)

    print("\nSaved to:")
    print(out_csv)
    print("============================================")


if __name__ == "__main__":
    main()