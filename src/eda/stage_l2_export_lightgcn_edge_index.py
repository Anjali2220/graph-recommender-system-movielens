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


def main():
    project_root = find_project_root(Path(__file__).resolve().parent)

    bip_graph_path = project_root / "outputs" / "gnn" / "lightgcn" / "graph" / "lightgcn_bipartite_graph.pt"
    meta_path = project_root / "outputs" / "gnn" / "lightgcn" / "graph" / "lightgcn_bipartite_metadata.json"

    out_dir = project_root / "outputs" / "gnn" / "lightgcn" / "train_graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_out = out_dir / "edge_index.pt"
    info_out = out_dir / "edge_index_info.json"

    print("============================================")
    print("✅ TASK L2 — Export LightGCN edge_index.pt")
    print("============================================")

    data = torch.load(bip_graph_path, map_location="cpu", weights_only=False)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if "edge_index" not in data:
        raise ValueError("edge_index key not found in saved graph dictionary.")

    edge_index = data["edge_index"]

    num_nodes = int(meta["num_nodes"])
    if int(edge_index.max()) >= num_nodes:
        raise ValueError("edge_index contains invalid node indices.")

    torch.save(edge_index, edge_out)

    info = {
        "num_nodes": num_nodes,
        "n_users": int(meta["n_users"]),
        "n_items": int(meta["n_items"]),
        "edge_index_shape": [int(edge_index.size(0)), int(edge_index.size(1))],
        "num_edges": int(edge_index.size(1)),
        "notes": "LightGCN training edges (user<->movie) forward+reverse."
    }

    info_out.write_text(json.dumps(info, indent=4), encoding="utf-8")

    print("✅ Exported successfully.")
    print(json.dumps(info, indent=4))


if __name__ == "__main__":
    main()