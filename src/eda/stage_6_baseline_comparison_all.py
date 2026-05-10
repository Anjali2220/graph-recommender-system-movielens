import os
import pandas as pd


def load_one_row(csv_path: str, model_name: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing file:\n{csv_path}")
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f"No rows found in:\n{csv_path}")
    row = df.iloc[[0]].copy()
    row.insert(0, "Model", model_name)
    return row


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # ----------------------------
    # Baseline report paths
    # ----------------------------
    POP_TEST = os.path.join(
        PROJECT_ROOT, "outputs", "baseline", "Cpopularity_model", "final_report", "popularity_test_results.csv"
    )

    ITEMKNN_TEST = os.path.join(
        PROJECT_ROOT, "outputs", "baseline", "Aitemknn_model", "final_report", "baseline_test_results.csv"
    )

    MF_TEST = os.path.join(
        PROJECT_ROOT, "outputs", "baseline", "Bmf_model", "final_report", "mf_test_results.csv"
    )

    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "comparison")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_CSV = os.path.join(OUT_DIR, "baseline_comparison_all.csv")

    # ----------------------------
    # Load one-row summaries
    # ----------------------------
    pop_df = load_one_row(POP_TEST, "Popularity")
    knn_df = load_one_row(ITEMKNN_TEST, "ItemKNN")
    mf_df = load_one_row(MF_TEST, "MF (BPR)")

    # Combine
    combined = pd.concat([pop_df, knn_df, mf_df], ignore_index=True)

    # Keep only the important columns (if they exist)
    preferred_cols = [
        "Model",
        "Precision@10", "Recall@10", "NDCG@10", "HitRate@10",
        "Precision@20", "Recall@20", "NDCG@20", "HitRate@20",
        "users_evaluated", "num_negatives",
        "topk_neighbors",  # only ItemKNN has this
        "best_tag", "emb_dim", "lr", "epochs"  # MF has these
    ]

    final_cols = [c for c in preferred_cols if c in combined.columns]
    combined = combined[final_cols]

    # Round for nice display
    for c in combined.columns:
        if c.startswith(("Precision@", "Recall@", "NDCG@", "HitRate@")):
            combined[c] = combined[c].astype(float).round(4)

    # Save
    combined.to_csv(OUT_CSV, index=False)

    print("\n✅ FINAL BASELINE COMPARISON TABLE (TEST)")
    print("=" * 80)
    print(combined.to_string(index=False))
    print("=" * 80)
    print("\n✅ Saved to:", OUT_CSV)




if __name__ == "__main__":
    main()
