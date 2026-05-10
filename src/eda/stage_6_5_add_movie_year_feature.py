import re
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# IMPORTANT: PyG class for safe loading
from torch_geometric.data import HeteroData


YEAR_REGEX = re.compile(r"\((\d{4})\)")


def extract_year(title: str) -> int:
    """Extract year from 'Toy Story (1995)' -> 1995. If missing -> 0."""
    if not isinstance(title, str):
        return 0
    m = YEAR_REGEX.search(title)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 0
    return 0


def minmax_normalize_keep_zero(arr: np.ndarray) -> np.ndarray:
    """Normalize non-zero years to [0,1], keep 0 as 0 (missing)."""
    arr = arr.astype(np.float32)
    mask = arr > 0
    if mask.sum() == 0:
        return arr
    y = arr[mask]
    ymin, ymax = y.min(), y.max()
    if ymax == ymin:
        arr[mask] = 1.0
    else:
        arr[mask] = (y - ymin) / (ymax - ymin)
    return arr


def safe_load_heterodata(path: Path):
    """
    PyTorch 2.6+ uses weights_only=True by default.
    We allowlist HeteroData and load with weights_only=False.
    """
    # torch.serialization exists in modern PyTorch
    if hasattr(torch, "serialization") and hasattr(torch.serialization, "safe_globals"):
        with torch.serialization.safe_globals([HeteroData]):
            return torch.load(path, weights_only=False)
    else:
        # Fallback for older PyTorch versions
        return torch.load(path)


def main():
    project_root = Path(__file__).resolve().parents[2]

    # Correct paths for your project
    graph_dir = project_root / "outputs" / "gnn" / "graph"
    graph_path = graph_dir / "hetero_graph_train.pt"

    # Mapping location (try graph folder first, fallback to graph_edges folder)
    movie_map_path_1 = graph_dir / "movie_id_mapping.csv"
    movie_map_path_2 = project_root / "data" / "processed" / "graph_edges" / "movie_id_mapping.csv"

    if movie_map_path_1.exists():
        movie_map_path = movie_map_path_1
    elif movie_map_path_2.exists():
        movie_map_path = movie_map_path_2
    else:
        raise FileNotFoundError(
            "Could not find movie_id_mapping.csv in either:\n"
            f" - {movie_map_path_1}\n"
            f" - {movie_map_path_2}\n"
        )

    # movies.csv
    movies_csv_1 = project_root / "data" / "raw" / "movies.csv"
    movies_csv_2 = project_root / "data" / "processed" / "movies.csv"
    if movies_csv_1.exists():
        movies_csv = movies_csv_1
    elif movies_csv_2.exists():
        movies_csv = movies_csv_2
    else:
        raise FileNotFoundError(
            "Could not find movies.csv in either:\n"
            f" - {movies_csv_1}\n"
            f" - {movies_csv_2}\n"
        )

    out_graph_path = graph_dir / "hetero_graph_train_with_year.pt"
    out_report_path = graph_dir / "movie_year_feature_report.json"

    print("============================================")
    print("✅ Stage 6.5 — Add Movie Year Feature")
    print("============================================")
    print("Project root    :", project_root)
    print("Movies CSV      :", movies_csv)
    print("Movie mapping   :", movie_map_path)
    print("Input graph     :", graph_path)
    print("Output graph    :", out_graph_path)
    print("============================================")

    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found at: {graph_path}")

    # ---------------------------
    # Load graph safely (PyTorch 2.6 fix)
    # ---------------------------
    data = safe_load_heterodata(graph_path)

    # ---------------------------
    # Load movie mapping (movieId -> movie_idx)
    # ---------------------------
    movie_map = pd.read_csv(movie_map_path)
    if "movieId" not in movie_map.columns or "movie_idx" not in movie_map.columns:
        raise ValueError("movie_id_mapping.csv must contain columns: movieId, movie_idx")

    movieId_to_idx = dict(
        zip(movie_map["movieId"].astype(int).tolist(), movie_map["movie_idx"].astype(int).tolist())
    )

    # ---------------------------
    # Load movies.csv and extract years
    # ---------------------------
    movies = pd.read_csv(movies_csv)
    if "movieId" not in movies.columns or "title" not in movies.columns:
        raise ValueError("movies.csv must contain columns: movieId, title")

    movies["movieId"] = movies["movieId"].astype(int)

    n_movies = int(data["movie"].num_nodes)
    years = np.zeros(n_movies, dtype=np.int32)

    for row in movies.itertuples(index=False):
        mid = int(getattr(row, "movieId"))
        if mid not in movieId_to_idx:
            continue
        idx = movieId_to_idx[mid]
        title = getattr(row, "title")
        years[idx] = extract_year(title)

    years_norm = minmax_normalize_keep_zero(years)
    x_year = torch.tensor(years_norm, dtype=torch.float32).view(-1, 1)

    # Attach as movie feature
    data["movie"].x = x_year

    # Save updated graph
    graph_dir.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_graph_path)

    # Save feature report (good for thesis/report)
    report = {
        "feature": "movie_year_normalized",
        "shape": [int(x_year.shape[0]), int(x_year.shape[1])],
        "num_movies": int(n_movies),
        "year_missing_count": int((years == 0).sum()),
        "year_present_count": int((years > 0).sum()),
        "year_min_present": int(years[years > 0].min()) if (years > 0).any() else None,
        "year_max_present": int(years[years > 0].max()) if (years > 0).any() else None,
        "notes": "Year extracted from title '(YYYY)'. Missing years set to 0. Non-zero years min-max normalized to [0,1]."
    }

    with open(out_report_path, "w") as f:
        json.dump(report, f, indent=4)

    print("\n✅ Movie year feature added successfully!")
    print("movie.x shape:", tuple(data["movie"].x.shape))
    print("Saved updated graph:", out_graph_path)
    print("Saved feature report:", out_report_path)


if __name__ == "__main__":
    main()