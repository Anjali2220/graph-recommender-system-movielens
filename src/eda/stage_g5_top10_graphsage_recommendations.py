import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =========================
# CONFIG
# =========================
USER_LIST = [3054, 23349, 28364, 87505, 116229]
TOP_K = 10

TAG = "graphsage_strong_run"

EMB_DIM = 128
HIDDEN_DIM = 128
NUM_LAYERS = 3
DROPOUT = 0.05


# =========================
# Utils
# =========================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "src").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def safe_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_mapping(path: Path, col1: str, col2: str) -> Dict[int, int]:
    df = pd.read_csv(path)
    return dict(zip(df[col1].astype(int), df[col2].astype(int)))


def invert_mapping(d: Dict[int, int]) -> Dict[int, int]:
    return {v: k for k, v in d.items()}


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


# =========================
# Load GraphSAGE
# =========================
def import_graphsage(project_root: Path):
    import importlib.util

    candidates = [
        project_root / "src" / "eda" / "stage_g2_graphsage_recommender.py",
        project_root / "src" / "gnn" / "graphsage_recommender.py",
    ]

    errors = []
    for file_path in candidates:
        if file_path.exists():
            spec = importlib.util.spec_from_file_location("graphsage_module", str(file_path))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "GraphSAGERecommender"):
                    return mod.GraphSAGERecommender
                errors.append(f"{file_path} exists but no GraphSAGERecommender found")
            except Exception as e:
                errors.append(f"Failed importing {file_path}: {e}")

    raise ImportError("Could not import GraphSAGERecommender:\n" + "\n".join(errors))


# =========================
# Load metadata
# =========================
def load_movies(project_root: Path) -> pd.DataFrame:
    candidates = [
        project_root / "data" / "raw" / "movies.csv",
        project_root / "data" / "raw" / "ml-20m" / "movies.csv",
    ]
    path = find_first_existing(candidates)
    if path is None:
        raise FileNotFoundError(
            "Could not find movies.csv in any of these locations:\n"
            + "\n".join(str(p) for p in candidates)
        )

    df = pd.read_csv(path)
    needed = {"movieId", "title", "genres"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {needed}. Found: {list(df.columns)}")
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
    ptr = data["user_ptr"].astype(np.int64)
    items = data["pos_items_flat"].astype(np.int64)

    s = ptr[user_idx]
    e = ptr[user_idx + 1]
    return items[s:e]


# =========================
# Helper: pick one watched title
# =========================
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
# Main
# =========================
def main():
    project_root = find_project_root(Path(__file__).resolve())

    graph_pt = project_root / "outputs" / "gnn" / "graph" / "hetero_graph_train.pt"
    pos_npz = project_root / "outputs" / "gnn" / "data" / "train_pos_sets.npz"

    run_dir = project_root / "outputs" / "gnn" / "graphsage" / "runs" / TAG
    checkpoint = run_dir / "model_final.pt"

    mapping_dir = project_root / "data" / "processed" / "graph_edges"

    out_csv = run_dir / "demo_user_recommendations.csv"
    ensure_dir(out_csv.parent)

    print("============================================")
    print("✅ TASK G5 — Demo GraphSAGE Recommendations")
    print("============================================")
    print(f"graph_pt      : {graph_pt}")
    print(f"checkpoint    : {checkpoint}")
    print(f"pos_npz       : {pos_npz}")
    print(f"mapping_dir   : {mapping_dir}")
    print(f"out_csv       : {out_csv}")
    print("============================================")

    for p in [graph_pt, checkpoint, pos_npz]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # mappings
    user_map_path = mapping_dir / "user_id_mapping.csv"
    movie_map_path = mapping_dir / "movie_id_mapping.csv"

    if not user_map_path.exists() or not movie_map_path.exists():
        raise FileNotFoundError(
            "Missing mapping files:\n"
            f"- {user_map_path}\n"
            f"- {movie_map_path}"
        )

    user_map = load_mapping(user_map_path, "userId", "user_idx")
    movie_map = load_mapping(movie_map_path, "movieId", "movie_idx")
    movie_idx_to_id = invert_mapping(movie_map)

    # metadata files
    movies = load_movies(project_root)
    tags = load_tags(project_root)

    # graph + model
    print("Loading graph...")
    data = safe_load(graph_pt)

    metadata = data.metadata()
    num_nodes_dict = {nt: int(data[nt].num_nodes) for nt in data.node_types}

    GraphSAGERecommender = import_graphsage(project_root)

    model = GraphSAGERecommender(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    )

    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()
    print("✅ Loaded GraphSAGE checkpoint")

    # full-graph embeddings
    print("Running full-graph embedding inference (CPU)...")
    with torch.no_grad():
        z = model(data)

    z_user = F.normalize(z["user"], dim=1).cpu()
    z_movie = F.normalize(z["movie"], dim=1).cpu()

    rows = []

    for user_id in USER_LIST:
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

        top_scores, top_idx = torch.topk(scores, TOP_K)
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

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    print("\n============================================")
    print("✅ Saved recommendations to:")
    print(out_csv)
    print("============================================")


if __name__ == "__main__":
    main()