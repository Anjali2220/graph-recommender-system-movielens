import json
from pathlib import Path


def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(10):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def main():
    project_root = find_project_root(Path(__file__).parent)

    out_dir = project_root / "outputs" / "gnn" / "hgt"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "tuning_plan.json"

    # ---------------------------------------------------------
    # ✅ Coarse search (8 configs) — good coverage, not too many
    # ---------------------------------------------------------
    base_epochs = 25  # early stopping will stop earlier if not improving

    coarse_configs = [
        # dim=64
        {"tag": "d64_l2_h2_lr1e-3_do0.1_wd0",   "emb_dim": 64,  "layers": 2, "heads": 2, "lr": 1e-3,  "dropout": 0.1, "weight_decay": 0.0,  "max_epochs": base_epochs},
        {"tag": "d64_l2_h4_lr1e-3_do0.1_wd0",   "emb_dim": 64,  "layers": 2, "heads": 4, "lr": 1e-3,  "dropout": 0.1, "weight_decay": 0.0,  "max_epochs": base_epochs},
        {"tag": "d64_l3_h2_lr5e-4_do0.2_wd1e-5","emb_dim": 64,  "layers": 3, "heads": 2, "lr": 5e-4,  "dropout": 0.2, "weight_decay": 1e-5, "max_epochs": base_epochs},
        {"tag": "d64_l3_h4_lr5e-4_do0.2_wd1e-5","emb_dim": 64,  "layers": 3, "heads": 4, "lr": 5e-4,  "dropout": 0.2, "weight_decay": 1e-5, "max_epochs": base_epochs},

        # dim=128
        {"tag": "d128_l2_h2_lr1e-3_do0.1_wd0",  "emb_dim": 128, "layers": 2, "heads": 2, "lr": 1e-3,  "dropout": 0.1, "weight_decay": 0.0,  "max_epochs": base_epochs},
        {"tag": "d128_l2_h4_lr1e-3_do0.1_wd0",  "emb_dim": 128, "layers": 2, "heads": 4, "lr": 1e-3,  "dropout": 0.1, "weight_decay": 0.0,  "max_epochs": base_epochs},
        {"tag": "d128_l3_h2_lr5e-4_do0.2_wd1e-5","emb_dim": 128,"layers": 3, "heads": 2, "lr": 5e-4, "dropout": 0.2, "weight_decay": 1e-5, "max_epochs": base_epochs},
        {"tag": "d128_l3_h4_lr5e-4_do0.2_wd1e-5","emb_dim": 128,"layers": 3, "heads": 4, "lr": 5e-4, "dropout": 0.2, "weight_decay": 1e-5, "max_epochs": base_epochs},
    ]

    # ---------------------------------------------------------
    # ✅ Refinement configs (optional)
    # We'll run these after we know the top 1–2 winners from coarse search.
    # ---------------------------------------------------------
    refine_configs = [
        # lower dropout, slightly longer training window (still early-stopped)
        {"tag": "refine_do0.05", "dropout": 0.05, "max_epochs": 40},
        {"tag": "refine_do0.15", "dropout": 0.15, "max_epochs": 40},
        {"tag": "refine_lr7e-4", "lr": 7e-4, "max_epochs": 40},
        {"tag": "refine_lr3e-4", "lr": 3e-4, "max_epochs": 40},
    ]

    plan = {
        "goal": "Tune HGT for maximum ranking performance (select by NDCG@20 on validation).",
        "selection_metric": "NDCG@20",
        "early_stopping": {
            "monitor": "NDCG@20",
            "patience": 3,
            "min_delta": 0.0005
        },
        "coarse_search": coarse_configs,
        "refinement_notes": "After coarse search, take the best 1–2 configs and apply refine_configs one by one.",
        "refine_configs": refine_configs
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=4)

    print("✅ Saved tuning plan to:")
    print(out_path)
    print("\nCoarse configs:", len(coarse_configs))
    print("Refine templates:", len(refine_configs))


if __name__ == "__main__":
    main()