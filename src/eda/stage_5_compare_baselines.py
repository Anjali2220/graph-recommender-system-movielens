import os
import pandas as pd


def main():
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Paths to final reports
    ITEMKNN_PATH = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "baseline",
        "Aitemknn_model",
        "final_report",
        "baseline_test_results.csv",
    )

    MF_PATH = os.path.join(
        PROJECT_ROOT,
        "outputs",
        "baseline",
        "Bmf_model",
        "final_report",
        "mf_test_results.csv",
    )

    OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "baseline", "comparison")
    os.makedirs(OUT_DIR, exist_ok=True)

    OUT_PATH = os.path.join(OUT_DIR, "baseline_comparison_table.csv")

    print("Loading ItemKNN results from:")
    print(ITEMKNN_PATH)

    print("\nLoading MF results from:")
    print(MF_PATH)

    # Load CSVs
    itemknn_df = pd.read_csv(ITEMKNN_PATH)
    mf_df = pd.read_csv(MF_PATH)

    # Select comparable columns
    columns = [
        "Precision@10",
        "Recall@10",
        "NDCG@10",
        "Precision@20",
        "Recall@20",
        "NDCG@20",
        "HitRate@20",
        "users_evaluated",
    ]

    item_row = itemknn_df.iloc[0][columns]
    mf_row = mf_df.iloc[0][columns]

    comparison_df = pd.DataFrame(
        [item_row.values, mf_row.values],
        columns=columns,
        index=["ItemKNN", "MF (BPR)"],
    )

    comparison_df.to_csv(OUT_PATH)

    print("\n✅ Baseline Comparison Table")
    print("=" * 60)
    print(comparison_df)
    print("=" * 60)

    print("\nSaved comparison table to:")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
