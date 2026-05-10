import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm


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


def load_mapping(path: Path):
    df = pd.read_csv(path)
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


def compute_metrics(rank, K):
    if rank <= K:
        precision = 1.0 / K
        recall = 1.0
        ndcg = 1.0 / np.log2(rank + 1)
        hit = 1.0
    else:
        precision = 0.0
        recall = 0.0
        ndcg = 0.0
        hit = 0.0
    return precision, recall, ndcg, hit


# =========================
# Main
# =========================
def main():
    project_root = find_project_root(Path(__file__).parent)

    graph_pt = project_root / "outputs/gnn/graph/hetero_graph_train.pt"
    train_csv = project_root / "data/processed/train.csv"
    val_csv = project_root / "data/processed/val.csv"

    mapping_dir = project_root / "data/processed/graph_edges"
    user_map = load_mapping(mapping_dir / "user_id_mapping.csv")
    movie_map = load_mapping(mapping_dir / "movie_id_mapping.csv")

    checkpoint = project_root / "outputs/gnn/hgt_runs/hgt_balanced_run/model_final.pt"

    print("============================================")
    print("✅ TASK 7.5 — HGT Validation (Sampled)")
    print("============================================")

    # Load graph
    data = torch.load(graph_pt, weights_only=False)
    metadata = data.metadata()
    num_nodes_dict = {ntype: int(data[ntype].num_nodes) for ntype in data.node_types}

    from src.eda.stage_7_3_hgt_recommender import HGTRecommender

    model = HGTRecommender(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        emb_dim=64,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
    )

    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    print("Running full-graph embedding inference...")
    with torch.no_grad():
        z_dict = model(data)

    user_emb = z_dict["user"]
    item_emb = z_dict["movie"]

    # Load train to filter positives
    train_df = pd.read_csv(train_csv)
    train_df = train_df[train_df["rating"] >= 4.0]

    user_train_items = {}
    for row in train_df.itertuples():
        if row.userId in user_map and row.movieId in movie_map:
            u = user_map[row.userId]
            i = movie_map[row.movieId]
            user_train_items.setdefault(u, set()).add(i)

    # Load validation
    val_df = pd.read_csv(val_csv)

    K_values = [10, 20]
    num_negatives = 499

    results = {k: {"P": [], "R": [], "NDCG": [], "HR": []} for k in K_values}

    users_evaluated = 0

    all_items = set(range(item_emb.shape[0]))

    for row in tqdm(val_df.itertuples(), total=len(val_df)):
        if row.userId not in user_map or row.movieId not in movie_map:
            continue

        u = user_map[row.userId]
        pos_item = movie_map[row.movieId]

        if u not in user_train_items:
            continue

        train_items = user_train_items[u]

        # sample negatives not in training set
        candidates = list(all_items - train_items - {pos_item})
        if len(candidates) < num_negatives:
            continue

        negatives = random.sample(candidates, num_negatives)

        items_to_rank = negatives + [pos_item]

        user_vec = user_emb[u]
        item_vecs = item_emb[items_to_rank]

        scores = torch.matmul(item_vecs, user_vec).numpy()

        ranked_idx = np.argsort(-scores)
        ranked_items = [items_to_rank[i] for i in ranked_idx]

        rank = ranked_items.index(pos_item) + 1

        for K in K_values:
            p, r, ndcg, hr = compute_metrics(rank, K)
            results[K]["P"].append(p)
            results[K]["R"].append(r)
            results[K]["NDCG"].append(ndcg)
            results[K]["HR"].append(hr)

        users_evaluated += 1

    print("\nUsers evaluated:", users_evaluated)

    final = {}
    for K in K_values:
        final[f"Precision@{K}"] = float(np.mean(results[K]["P"]))
        final[f"Recall@{K}"] = float(np.mean(results[K]["R"]))
        final[f"NDCG@{K}"] = float(np.mean(results[K]["NDCG"]))
        final[f"HitRate@{K}"] = float(np.mean(results[K]["HR"]))

    print("\n✅ HGT Validation Results (Sampled):")
    print(final)

    out_dir = checkpoint.parent / "val_report"
    out_dir.mkdir(exist_ok=True)

    pd.DataFrame([final]).to_csv(out_dir / "hgt_val_results_sampled.csv", index=False)

    print("\n✅ Saved to:", out_dir / "hgt_val_results_sampled.csv")
    print("============================================")


if __name__ == "__main__":
    main()