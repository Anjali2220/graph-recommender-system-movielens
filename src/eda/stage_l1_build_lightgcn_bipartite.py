import json
from pathlib import Path

import torch


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "src").exists() and (cur / "data").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pretty(obj) -> str:
    return json.dumps(obj, indent=4, ensure_ascii=False)


def safe_torch_load(path: Path, map_location="cpu"):
    # PyTorch 2.6+ uses weights_only=True by default; for HeteroData we need full object.
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def main():
    project_root = find_project_root(Path(__file__).resolve().parent)

    hetero_pt = project_root / "outputs" / "gnn" / "graph" / "hetero_graph_train.pt"
    out_dir = project_root / "outputs" / "gnn" / "lightgcn" / "graph"
    ensure_dir(out_dir)

    out_pt = out_dir / "lightgcn_bipartite_graph.pt"
    out_meta = out_dir / "lightgcn_bipartite_metadata.json"

    print("============================================")
    print("✅ TASK L1 — Build LightGCN Bipartite Graph")
    print("============================================")
    print(f"Input hetero graph : {hetero_pt}")
    print(f"Output graph       : {out_pt}")
    print(f"Output metadata    : {out_meta}")
    print("============================================")

    if not hetero_pt.exists():
        raise FileNotFoundError(f"Missing: {hetero_pt}")

    data = safe_torch_load(hetero_pt, map_location="cpu")

    rel = ("user", "rates", "movie")
    if rel not in data.edge_types:
        raise ValueError(f"Edge type {rel} not found. Available: {data.edge_types}")

    n_users = int(data["user"].num_nodes)
    n_items = int(data["movie"].num_nodes)

    # Extract user->movie edges (train positives only)
    um = data[rel].edge_index  # [2, E], user_idx -> movie_idx
    u = um[0].to(torch.long)
    m = um[1].to(torch.long)

    # Build bipartite node space:
    # users: [0 .. n_users-1]
    # movies: [n_users .. n_users+n_items-1]
    m_global = m + n_users

    # Undirected edges for LightGCN (both directions)
    src = torch.cat([u, m_global], dim=0)
    dst = torch.cat([m_global, u], dim=0)
    edge_index = torch.stack([src, dst], dim=0)  # [2, 2E]

    # Optional: coalesce (removes duplicates, sorts). Safe but can take time on huge graphs.
    # You can skip if you want speed. Keeping it ON is cleaner.
    edge_index = torch.unique(edge_index, dim=1)

    graph = {
        "edge_index": edge_index,   # [2, num_edges]
        "n_users": n_users,
        "n_items": n_items,
        "num_nodes": n_users + n_items,
        "notes": "LightGCN bipartite graph: users and movies in one node space; undirected edges."
    }

    torch.save(graph, out_pt)

    meta = {
        "n_users": n_users,
        "n_items": n_items,
        "num_nodes": n_users + n_items,
        "edge_index_shape": list(edge_index.shape),
        "num_edges": int(edge_index.size(1)),
        "source_relation": str(rel),
        "notes": "Built from hetero_graph_train.pt using only user<->movie training edges (no leakage)."
    }
    out_meta.write_text(pretty(meta), encoding="utf-8")

    print("✅ Done.")
    print(pretty(meta))


if __name__ == "__main__":
    main()