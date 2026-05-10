import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


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


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pretty(obj) -> str:
    return json.dumps(obj, indent=4, ensure_ascii=False)


def safe_torch_load(path: Path, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_id_mapping(path: Path, id_col: str, idx_col: str) -> Dict[int, int]:
    df = pd.read_csv(path)
    if id_col not in df.columns or idx_col not in df.columns:
        raise ValueError(f"{path} must contain columns {id_col} and {idx_col}. Found: {list(df.columns)}")
    return dict(zip(df[id_col].astype(int).tolist(), df[idx_col].astype(int).tolist()))


# =========================
# Metrics
# =========================
def dcg_at_k(rel: np.ndarray, k: int) -> float:
    rel = rel[:k]
    if rel.size == 0:
        return 0.0
    denom = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum(rel / denom))


def ndcg_at_k(rel: np.ndarray, k: int) -> float:
    ideal = np.sort(rel)[::-1]
    denom = dcg_at_k(ideal, k)
    if denom <= 0:
        return 0.0
    return dcg_at_k(rel, k) / denom


def precision_at_k(rel: np.ndarray, k: int) -> float:
    return float(np.sum(rel[:k]) / k) if k > 0 else 0.0


def recall_at_k(rel: np.ndarray, k: int, num_pos: int) -> float:
    if num_pos <= 0:
        return 0.0
    return float(np.sum(rel[:k]) / num_pos)


def hitrate_at_k(rel: np.ndarray, k: int) -> float:
    return float(np.any(rel[:k] > 0))


# =========================
# Import GraphSAGE model
# =========================
def import_graphsage(project_root: Path):
    candidates = [
        project_root / "src" / "eda" / "stage_g2_graphsage_recommender.py",
        project_root / "src" / "gnn" / "graphsage_recommender.py",
    ]

    errors = []
    for file_path in candidates:
        if file_path.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("graphsage_module", str(file_path))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)  # type: ignore
                if hasattr(mod, "GraphSAGERecommender"):
                    return mod.GraphSAGERecommender, str(file_path)
                errors.append(f"{file_path} exists but no GraphSAGERecommender found")
            except Exception as e:
                errors.append(f"Failed importing {file_path}: {e}")

    raise ImportError("Could not import GraphSAGERecommender:\n" + "\n".join(errors))


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()

    # IMPORTANT: default tag must match your trained run
    parser.add_argument("--tag", type=str, default="graphsage_strong_run")

    parser.add_argument("--graph_pt", type=str, default="")
    parser.add_argument("--pos_npz", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--train_csv", type=str, default="")
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--mapping_dir", type=str, default="")
    parser.add_argument("--out_dir", type=str, default="")

    parser.add_argument("--rating_threshold", type=float, default=4.0)
    parser.add_argument("--negatives", type=int, default=499)
    parser.add_argument("--k_list", type=int, nargs="+", default=[10, 20])
    parser.add_argument("--max_users", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)

    # IMPORTANT: must match strong training
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)

    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    project_root = find_project_root(Path(__file__).resolve().parent)

    graph_pt = Path(args.graph_pt) if args.graph_pt else (
        project_root / "outputs" / "gnn" / "graph" / "hetero_graph_train.pt"
    )
    pos_npz = Path(args.pos_npz) if args.pos_npz else (
        project_root / "outputs" / "gnn" / "data" / "train_pos_sets.npz"
    )

    run_dir = project_root / "outputs" / "gnn" / "graphsage" / "runs" / args.tag
    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "model_final.pt"

    train_csv = Path(args.train_csv) if args.train_csv else project_root / "data" / "processed" / "train.csv"
    val_csv = Path(args.val_csv) if args.val_csv else project_root / "data" / "processed" / "val.csv"
    mapping_dir = Path(args.mapping_dir) if args.mapping_dir else project_root / "data" / "processed" / "graph_edges"

    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "val_report"
    ensure_dir(out_dir)

    print("============================================")
    print("✅ TASK G4 — GraphSAGE Validation Evaluation")
    print("============================================")
    print(f"graph_pt      : {graph_pt}")
    print(f"pos_npz       : {pos_npz}")
    print(f"train.csv     : {train_csv}")
    print(f"val.csv       : {val_csv}")
    print(f"mappings dir  : {mapping_dir}")
    print(f"checkpoint    : {checkpoint}")
    print(f"out_dir       : {out_dir}")
    print(f"negatives     : {args.negatives}")
    print(f"K values      : {args.k_list}")
    print("============================================")

    for p in [graph_pt, pos_npz, train_csv, val_csv, checkpoint]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    # load graph
    data = safe_torch_load(graph_pt, map_location="cpu")
    metadata = data.metadata()
    num_nodes_dict = {ntype: int(data[ntype].num_nodes) for ntype in data.node_types}

    n_users = num_nodes_dict["user"]
    n_items = num_nodes_dict["movie"]

    # mappings
    user_map_path = mapping_dir / "user_id_mapping.csv"
    movie_map_path = mapping_dir / "movie_id_mapping.csv"
    if not user_map_path.exists() or not movie_map_path.exists():
        raise FileNotFoundError(
            "Expected mapping files:\n"
            f"- {user_map_path}\n"
            f"- {movie_map_path}"
        )

    userId_to_idx = load_id_mapping(user_map_path, "userId", "user_idx")
    movieId_to_idx = load_id_mapping(movie_map_path, "movieId", "movie_idx")

    # train positives for filtering negatives
    npz = np.load(pos_npz)
    user_ptr = npz["user_ptr"].astype(np.int64)
    pos_items_flat = npz["pos_items_flat"].astype(np.int64)

    def get_train_pos(u: int) -> np.ndarray:
        s, e = user_ptr[u], user_ptr[u + 1]
        return pos_items_flat[s:e]

    # validation positives
    df_val = pd.read_csv(val_csv)
    df_val = df_val[df_val["rating"] >= args.rating_threshold][["userId", "movieId"]].copy()
    df_val["user_idx"] = df_val["userId"].map(userId_to_idx)
    df_val["movie_idx"] = df_val["movieId"].map(movieId_to_idx)
    df_val = df_val.dropna(subset=["user_idx", "movie_idx"])
    df_val["user_idx"] = df_val["user_idx"].astype(int)
    df_val["movie_idx"] = df_val["movie_idx"].astype(int)

    # one validation positive per user
    user_to_val_pos: Dict[int, int] = {}
    for u, grp in df_val.groupby("user_idx"):
        items = grp["movie_idx"].to_numpy(dtype=np.int64)
        user_to_val_pos[int(u)] = int(items[rng.integers(0, len(items))])

    users = np.array(sorted(user_to_val_pos.keys()), dtype=np.int64)
    if args.max_users and args.max_users > 0:
        users = users[: args.max_users]

    print(f"Users evaluated: {len(users)}")

    # import model
    GraphSAGERecommender, imported_from = import_graphsage(project_root)
    print(f"✅ Imported GraphSAGERecommender from: {imported_from}")

    model = GraphSAGERecommender(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )

    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print("✅ Loaded checkpoint")

    model.eval()

    print("Running full-graph embedding inference (CPU)...")
    with torch.no_grad():
        z_dict = model(data)

    # IMPORTANT: normalize exactly like training
    z_user = F.normalize(z_dict["user"], dim=1).cpu().numpy()
    z_movie = F.normalize(z_dict["movie"], dim=1).cpu().numpy()

    k_list = sorted(args.k_list)
    sums = {f"Precision@{k}": 0.0 for k in k_list}
    sums.update({f"Recall@{k}": 0.0 for k in k_list})
    sums.update({f"NDCG@{k}": 0.0 for k in k_list})
    sums.update({f"HitRate@{k}": 0.0 for k in k_list})

    n_eval = 0
    for i, u in enumerate(users, start=1):
        pos_i = user_to_val_pos[int(u)]
        train_pos = set(get_train_pos(int(u)).tolist())

        negs: List[int] = []
        while len(negs) < args.negatives:
            j = int(rng.integers(0, n_items))
            if j == pos_i:
                continue
            if j in train_pos:
                continue
            negs.append(j)

        cand = np.array([pos_i] + negs, dtype=np.int64)
        scores = z_movie[cand] @ z_user[u]
        ranked = cand[np.argsort(-scores)]
        rel = (ranked == pos_i).astype(np.float32)

        for k in k_list:
            sums[f"Precision@{k}"] += precision_at_k(rel, k)
            sums[f"Recall@{k}"] += recall_at_k(rel, k, num_pos=1)
            sums[f"NDCG@{k}"] += ndcg_at_k(rel, k)
            sums[f"HitRate@{k}"] += hitrate_at_k(rel, k)

        n_eval += 1
        if i % 20000 == 0:
            print(f"Evaluated {i} users...")

    results = {k: v / max(1, n_eval) for k, v in sums.items()}
    row = {
        "model": "GraphSAGE",
        "users_evaluated": int(n_eval),
        "num_negatives": int(args.negatives),
        "emb_dim": int(args.emb_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "checkpoint": checkpoint.name,
        "tag": args.tag,
        **results
    }

    print("\n✅ GraphSAGE Validation Results:")
    print(row)

    out_csv = out_dir / "graphsage_val_results.csv"
    pd.DataFrame([row]).to_csv(out_csv, index=False)
    (out_dir / "graphsage_val_results.json").write_text(pretty(row), encoding="utf-8")

    print("\n✅ Saved to:")
    print(out_csv)
    print(out_dir / "graphsage_val_results.json")
    print("============================================")


if __name__ == "__main__":
    main()