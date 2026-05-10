import random
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch_geometric.loader import LinkNeighborLoader


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


# =========================
# Training
# =========================
def main():

    # -------------------------
    # Strong but CPU-safe setup
    # -------------------------
    EPOCHS = 8
    MAX_TRAIN_EDGES = 1_500_000
    BATCH_SIZE = 256
    NUM_NEIGHBORS = [5, 3]
    EMB_DIM = 64
    NEG_RATIO = 2.0

    print("============================================")
    print("🚀 HGT Training (IMPROVED CPU MODE)")
    print("============================================")
    print(f"Epochs               : {EPOCHS}")
    print(f"Batch size           : {BATCH_SIZE}")
    print(f"Neighbors            : {NUM_NEIGHBORS}")
    print(f"Emb dim              : {EMB_DIM}")
    print(f"Neg ratio            : {NEG_RATIO}")
    print(f"max_train_edges      : {MAX_TRAIN_EDGES}")
    print("============================================")

    project_root = find_project_root(Path(__file__).parent)

    graph_pt = project_root / "outputs/gnn/graph/hetero_graph_train.pt"
    out_dir = project_root / "outputs/gnn/hgt_runs/hgt_improved_run"
    ensure_dir(out_dir)

    # Load graph
    data = torch.load(graph_pt, weights_only=False)

    metadata = data.metadata()
    num_nodes_dict = {ntype: int(data[ntype].num_nodes) for ntype in data.node_types}

    from src.eda.stage_7_3_hgt_recommender import HGTRecommender

    model = HGTRecommender(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        emb_dim=EMB_DIM,
        num_layers=2,
        num_heads=2,
        dropout=0.1,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    rel = ("user", "rates", "movie")
    full_edge_index = data[rel].edge_index

    # Subsample training edges
    total_edges = full_edge_index.size(1)
    perm = torch.randperm(total_edges)
    selected = perm[:MAX_TRAIN_EDGES]
    edge_index = full_edge_index[:, selected]

    print(f"Supervision edges: using {edge_index.size(1)} / {total_edges}")

    loader = LinkNeighborLoader(
        data=data,
        num_neighbors=NUM_NEIGHBORS,
        edge_label_index=(rel, edge_index),
        edge_label=torch.ones(edge_index.size(1)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        neg_sampling_ratio=NEG_RATIO,
        num_workers=0,
    )

    bce = torch.nn.BCEWithLogitsLoss()

    # =========================
    # Train loop
    # =========================
    model.train()
    log_rows = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        total_loss = 0
        batches = 0

        for batch in loader:

            optimizer.zero_grad()

            z_dict = model(batch)

            edge_label_index = batch[rel].edge_label_index
            edge_label = batch[rel].edge_label.float()

            src = edge_label_index[0]
            dst = edge_label_index[1]

            scores = (z_dict["user"][src] * z_dict["movie"][dst]).sum(dim=-1)
            loss = bce(scores, edge_label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        avg_loss = total_loss / batches
        epoch_time = time.time() - t0

        print(f"✅ Epoch {epoch} | loss={avg_loss:.5f} | batches={batches} | time={epoch_time:.1f}s")

        torch.save(model.state_dict(), out_dir / f"model_epoch_{epoch}.pt")

        log_rows.append({
            "epoch": epoch,
            "loss": avg_loss,
            "batches": batches,
            "time_sec": epoch_time
        })

    torch.save(model.state_dict(), out_dir / "model_final.pt")
    pd.DataFrame(log_rows).to_csv(out_dir / "train_log.csv", index=False)

    print("============================================")
    print("✅ Training finished (IMPROVED MODE)")
    print("Model saved to:", out_dir)
    print("============================================")


if __name__ == "__main__":
    main()