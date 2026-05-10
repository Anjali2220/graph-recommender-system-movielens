from pathlib import Path
from typing import Dict, List, Optional
import importlib.util

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F

from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD


# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Graph-Based Recommender System", layout="wide")


# =========================================================
# ROOT PATH DETECTION
# =========================================================
def find_project_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / "data").exists() and (cur / "src").exists():
            return cur
        cur = cur.parent
    return start.resolve().parent


PROJECT_ROOT = find_project_root(Path(__file__).resolve())

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GRAPH_EDGES_DIR = PROCESSED_DIR / "graph_edges"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
GNN_DIR = OUTPUTS_DIR / "gnn"
GRAPH_DIR = GNN_DIR / "graph"
GRAPH_SAGE_RUN_DIR = GNN_DIR / "graphsage" / "runs" / "graphsage_strong_run"
LIGHTGCN_RUN_DIR = GNN_DIR / "lightgcn" / "runs"

SRC_DIR = PROJECT_ROOT / "src"
EDA_DIR = SRC_DIR / "eda"


# =========================================================
# HELPERS
# =========================================================
def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def safe_torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


@st.cache_data
def load_mapping(path: Path, col1: str, col2: str) -> Dict[int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing mapping file: {path}")
    df = pd.read_csv(path)
    if col1 not in df.columns or col2 not in df.columns:
        raise ValueError(f"{path} must contain columns '{col1}' and '{col2}'. Found: {list(df.columns)}")
    return dict(zip(df[col1].astype(int), df[col2].astype(int)))


@st.cache_data
def load_movies() -> pd.DataFrame:
    candidates = [
        RAW_DIR / "movies.csv",
        RAW_DIR / "ml-20m" / "movies.csv",
    ]
    path = find_first_existing(candidates)
    if path is None:
        raise FileNotFoundError("Could not find movies.csv")

    df = pd.read_csv(path)
    required = {"movieId", "title", "genres"}
    if not required.issubset(df.columns):
        raise ValueError(f"{path} missing required columns {required}")
    return df[["movieId", "title", "genres"]].copy()


@st.cache_data
def load_tags() -> pd.DataFrame:
    candidates = [
        RAW_DIR / "tags.csv",
        RAW_DIR / "ml-20m" / "tags.csv",
    ]
    path = find_first_existing(candidates)
    if path is None:
        return pd.DataFrame(columns=["movieId", "tags"])

    df = pd.read_csv(path)
    if not {"movieId", "tag"}.issubset(df.columns):
        return pd.DataFrame(columns=["movieId", "tags"])

    df["tag"] = df["tag"].astype(str).str.strip()
    df = df[df["tag"] != ""].copy()

    out = (
        df.groupby("movieId")["tag"]
        .apply(lambda x: ", ".join(x.value_counts().head(5).index.tolist()))
        .reset_index()
    )
    out.columns = ["movieId", "tags"]
    return out


@st.cache_data
def enrich_with_metadata(df: pd.DataFrame) -> pd.DataFrame:
    movies = load_movies()
    tags = load_tags()

    out = df.merge(movies, on="movieId", how="left")
    out = out.merge(tags, on="movieId", how="left")

    for col in ["title", "genres", "tags"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].fillna("")

    return out


@st.cache_data
def load_train_df(rating_threshold: float = 4.0) -> pd.DataFrame:
    train_csv = PROCESSED_DIR / "train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train.csv: {train_csv}")

    df = pd.read_csv(train_csv)
    df = df[df["rating"] >= rating_threshold].copy()
    return df


@st.cache_data
def build_cf_indices(rating_threshold: float = 4.0):
    df = load_train_df(rating_threshold)

    unique_users = sorted(df["userId"].unique())
    unique_movies = sorted(df["movieId"].unique())

    user_to_idx = {u: i for i, u in enumerate(unique_users)}
    movie_to_idx = {m: i for i, m in enumerate(unique_movies)}
    idx_to_movie = {i: m for m, i in movie_to_idx.items()}

    rows = df["userId"].map(user_to_idx).values
    cols = df["movieId"].map(movie_to_idx).values
    vals = np.ones(len(df), dtype=np.float32)

    X = csr_matrix((vals, (rows, cols)), shape=(len(unique_users), len(unique_movies)))
    return X, user_to_idx, movie_to_idx, idx_to_movie


@st.cache_data
def load_model_results() -> pd.DataFrame:
    rows = [
        {
            "Model": "ItemKNN",
            "Type": "Collaborative Filtering",
            "Uses Graph": "No",
            "Uses Genre/Tag in Model": "No",
            "Recall@10": 0.6816839271694022,
            "Recall@20": 0.8010244139529666,
            "NDCG@10": 0.416620006130359,
            "NDCG@20": 0.4467179862121562,
        },
        {
            "Model": "MF (BPR)",
            "Type": "Matrix Factorization",
            "Uses Graph": "No",
            "Uses Genre/Tag in Model": "No",
            "Recall@10": 0.5751299211126482,
            "Recall@20": 0.7299759474582819,
            "NDCG@10": 0.3396661468348195,
            "NDCG@20": 0.3788661761979923,
        },
        {
            "Model": "LightGCN",
            "Type": "Graph Collaborative Filtering",
            "Uses Graph": "Yes",
            "Uses Genre/Tag in Model": "No",
            "Recall@10": 0.48881992317671896,
            "Recall@20": 0.6462440538271731,
            "NDCG@10": 0.28099756574754203,
            "NDCG@20": 0.3207586246112404,
        },
        {
            "Model": "GraphSAGE",
            "Type": "Heterogeneous GNN",
            "Uses Graph": "Yes",
            "Uses Genre/Tag in Model": "Yes",
            "Recall@10": 0.4829691434967312,
            "Recall@20": 0.6625753605860955,
            "NDCG@10": 0.24082769952443706,
            "NDCG@20": 0.28634970668970205,
        },
    ]
    return pd.DataFrame(rows)


# =========================================================
# ITEMKNN LIVE
# =========================================================
@st.cache_resource
def load_itemknn_components():
    X, user_to_idx, movie_to_idx, idx_to_movie = build_cf_indices()
    item_sim = cosine_similarity(X.T, dense_output=False)
    return X, item_sim, user_to_idx, movie_to_idx, idx_to_movie


def recommend_itemknn_live(user_id: int, top_k: int = 10):
    X, item_sim, user_to_idx, _, idx_to_movie = load_itemknn_components()

    if user_id not in user_to_idx:
        return None, f"userId {user_id} not found in ItemKNN training data."

    uidx = user_to_idx[user_id]
    user_profile = X[uidx]

    scores = user_profile @ item_sim
    scores = np.asarray(scores.todense()).ravel()

    seen_idx = user_profile.indices
    scores[seen_idx] = -1e9

    top_idx = np.argsort(-scores)[:top_k]

    rows = []
    for rank, midx in enumerate(top_idx, start=1):
        rows.append(
            {
                "userId": user_id,
                "rank": rank,
                "movieId": idx_to_movie[midx],
                "score": float(scores[midx]),
            }
        )

    rec_df = pd.DataFrame(rows)
    rec_df = enrich_with_metadata(rec_df)
    return rec_df, None


# =========================================================
# MF LIVE
# =========================================================
@st.cache_resource
def load_mf_components(n_factors: int = 64):
    X, user_to_idx, movie_to_idx, idx_to_movie = build_cf_indices()
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    user_factors = svd.fit_transform(X)
    item_factors = svd.components_.T
    return X, user_factors, item_factors, user_to_idx, movie_to_idx, idx_to_movie


def recommend_mf_live(user_id: int, top_k: int = 10):
    X, user_factors, item_factors, user_to_idx, _, idx_to_movie = load_mf_components()

    if user_id not in user_to_idx:
        return None, f"userId {user_id} not found in MF training data."

    uidx = user_to_idx[user_id]
    scores = user_factors[uidx] @ item_factors.T

    seen_idx = X[uidx].indices
    scores[seen_idx] = -1e9

    top_idx = np.argsort(-scores)[:top_k]

    rows = []
    for rank, midx in enumerate(top_idx, start=1):
        rows.append(
            {
                "userId": user_id,
                "rank": rank,
                "movieId": idx_to_movie[midx],
                "score": float(scores[midx]),
            }
        )

    rec_df = pd.DataFrame(rows)
    rec_df = enrich_with_metadata(rec_df)
    return rec_df, None


# =========================================================
# LIGHTGCN LIVE
# =========================================================
def import_lightgcn():
    candidates = [
        EDA_DIR / "stage_l3_lightgcn_model.py",
        SRC_DIR / "gnn" / "lightgcn_model.py",
    ]

    errors = []
    for file_path in candidates:
        if file_path.exists():
            spec = importlib.util.spec_from_file_location("lightgcn_module", str(file_path))
            mod = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(mod)
                if hasattr(mod, "LightGCN") and hasattr(mod, "build_norm_adj"):
                    return mod.LightGCN, mod.build_norm_adj
                errors.append(f"{file_path} exists but missing LightGCN/build_norm_adj")
            except Exception as e:
                errors.append(f"Failed importing {file_path}: {e}")

    raise ImportError("Could not import LightGCN:\n" + "\n".join(errors))


def load_lightgcn_state_robust(model: torch.nn.Module, state: Dict[str, torch.Tensor], n_users: int, n_items: int):
    if "emb.weight" in state:
        W = state["emb.weight"]
        with torch.no_grad():
            model.user_emb.weight.copy_(W[:n_users])
            model.item_emb.weight.copy_(W[n_users:n_users + n_items])
        return
    model.load_state_dict(state, strict=False)


@st.cache_resource
def load_lightgcn_components():
    edge_index_pt = GNN_DIR / "lightgcn" / "train_graph" / "edge_index.pt"
    edge_info_json = GNN_DIR / "lightgcn" / "train_graph" / "edge_index_info.json"

    candidates = [
        LIGHTGCN_RUN_DIR / "lightgcn_strong_run" / "model_final.pt",
        LIGHTGCN_RUN_DIR / "lightgcn_run" / "model_final.pt",
    ]
    checkpoint = find_first_existing(candidates)

    if checkpoint is None:
        raise FileNotFoundError("No LightGCN checkpoint found.")
    if not edge_index_pt.exists():
        raise FileNotFoundError(f"Missing edge_index.pt: {edge_index_pt}")
    if not edge_info_json.exists():
        raise FileNotFoundError(f"Missing edge_index_info.json: {edge_info_json}")

    info = pd.read_json(edge_info_json, typ="series")
    n_users = int(info["n_users"])
    n_items = int(info["n_items"])
    num_nodes = int(info["num_nodes"])

    emb_dim = 128 if "strong" in str(checkpoint) else 64
    K = 3

    LightGCN, build_norm_adj = import_lightgcn()
    edge_index = safe_torch_load(edge_index_pt)
    A_hat = build_norm_adj(edge_index, num_nodes=num_nodes)

    model = LightGCN(n_users=n_users, n_items=n_items, emb_dim=emb_dim, K=K)
    state = torch.load(checkpoint, map_location="cpu")
    load_lightgcn_state_robust(model, state, n_users, n_items)
    model.eval()

    with torch.no_grad():
        z_user, z_movie = model.get_all_embeddings(A_hat)

    z_user = F.normalize(z_user, dim=1).cpu()
    z_movie = F.normalize(z_movie, dim=1).cpu()

    user_map = load_mapping(GRAPH_EDGES_DIR / "user_id_mapping.csv", "userId", "user_idx")
    movie_map = load_mapping(GRAPH_EDGES_DIR / "movie_id_mapping.csv", "movieId", "movie_idx")
    movie_idx_to_id = {v: k for k, v in movie_map.items()}

    return z_user, z_movie, user_map, movie_idx_to_id


@st.cache_data
def load_seen_npz():
    npz_path = GNN_DIR / "data" / "train_pos_sets.npz"
    data = np.load(npz_path)
    return data["user_ptr"].astype(np.int64), data["pos_items_flat"].astype(np.int64)


def get_seen_items(user_idx: int) -> np.ndarray:
    user_ptr, pos_items_flat = load_seen_npz()
    s, e = user_ptr[user_idx], user_ptr[user_idx + 1]
    return pos_items_flat[s:e]


def recommend_lightgcn_live(user_id: int, top_k: int = 10):
    z_user, z_movie, user_map, movie_idx_to_id = load_lightgcn_components()

    if user_id not in user_map:
        return None, f"userId {user_id} not found in LightGCN mapping."

    user_idx = user_map[user_id]
    seen = set(get_seen_items(user_idx).tolist())

    user_vec = z_user[user_idx]
    scores = torch.matmul(z_movie, user_vec)

    if len(seen) > 0:
        seen_tensor = torch.tensor(list(seen), dtype=torch.long)
        scores[seen_tensor] = -1e9

    top_scores, top_idx = torch.topk(scores, k=top_k)
    top_idx = top_idx.numpy().tolist()

    rows = []
    for rank, movie_idx in enumerate(top_idx, start=1):
        rows.append(
            {
                "userId": user_id,
                "rank": rank,
                "movieId": movie_idx_to_id[movie_idx],
            }
        )

    rec_df = pd.DataFrame(rows)
    rec_df = enrich_with_metadata(rec_df)
    return rec_df, None


# =========================================================
# GRAPHSAGE LIVE
# =========================================================
def import_graphsage():
    file_path = EDA_DIR / "stage_g2_graphsage_recommender.py"
    if not file_path.exists():
        raise FileNotFoundError(f"GraphSAGE model file not found: {file_path}")

    spec = importlib.util.spec_from_file_location("graphsage_module", str(file_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GraphSAGERecommender


@st.cache_resource
def load_graphsage_model_and_embeddings():
    graph_pt = GRAPH_DIR / "hetero_graph_train.pt"
    checkpoint = GRAPH_SAGE_RUN_DIR / "model_final.pt"

    if not graph_pt.exists():
        raise FileNotFoundError(f"Missing graph file: {graph_pt}")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing GraphSAGE checkpoint: {checkpoint}")

    data = safe_torch_load(graph_pt)
    metadata = data.metadata()
    num_nodes_dict = {nt: int(data[nt].num_nodes) for nt in data.node_types}

    GraphSAGERecommender = import_graphsage()
    model = GraphSAGERecommender(
        metadata=metadata,
        num_nodes_dict=num_nodes_dict,
        emb_dim=128,
        hidden_dim=128,
        num_layers=3,
        dropout=0.05,
    )

    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        z_dict = model(data)

    z_user = F.normalize(z_dict["user"], dim=1).cpu()
    z_movie = F.normalize(z_dict["movie"], dim=1).cpu()

    user_map = load_mapping(GRAPH_EDGES_DIR / "user_id_mapping.csv", "userId", "user_idx")
    movie_map = load_mapping(GRAPH_EDGES_DIR / "movie_id_mapping.csv", "movieId", "movie_idx")
    movie_idx_to_id = {v: k for k, v in movie_map.items()}

    return z_user, z_movie, user_map, movie_idx_to_id


def recommend_graphsage_live(user_id: int, top_k: int = 10):
    z_user, z_movie, user_map, movie_idx_to_id = load_graphsage_model_and_embeddings()

    if user_id not in user_map:
        return None, f"userId {user_id} not found in GraphSAGE mapping."

    user_idx = user_map[user_id]
    seen = set(get_seen_items(user_idx).tolist())
    user_vec = z_user[user_idx]
    scores = torch.matmul(z_movie, user_vec)

    if len(seen) > 0:
        seen_tensor = torch.tensor(list(seen), dtype=torch.long)
        scores[seen_tensor] = -1e9

    top_scores, top_idx = torch.topk(scores, k=top_k)
    top_idx = top_idx.numpy().tolist()

    rows = []
    for rank, movie_idx in enumerate(top_idx, start=1):
        rows.append(
            {
                "userId": user_id,
                "rank": rank,
                "movieId": movie_idx_to_id[movie_idx],
            }
        )

    rec_df = pd.DataFrame(rows)
    rec_df = enrich_with_metadata(rec_df)
    return rec_df, None


# =========================================================
# ROUTER
# =========================================================
def get_recommendations(model_name: str, user_id: int, top_k: int = 10):
    if model_name == "ItemKNN":
        return recommend_itemknn_live(user_id, top_k)
    if model_name == "MF (BPR)":
        return recommend_mf_live(user_id, top_k)
    if model_name == "LightGCN":
        return recommend_lightgcn_live(user_id, top_k)
    if model_name == "GraphSAGE":
        return recommend_graphsage_live(user_id, top_k)
    return None, f"Unknown model: {model_name}"


# =========================================================
# UI
# =========================================================
st.title("Graph-Based Recommender System Dashboard")

tab1, tab2, tab3 = st.tabs(["Overview", "Model Comparison", "Recommendation Demo"])

with tab1:
    st.subheader("Project Overview")
    st.write(
        """
This project compares classical collaborative filtering and graph-based recommendation models on the MovieLens dataset.

The heterogeneous graph contains:
- **User** nodes
- **Movie** nodes
- **Genre** nodes
- **Tag** nodes
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", "138,238")
    c2.metric("Movies", "20,639")
    c3.metric("Genres", "19")
    c4.metric("Tags", "1,422")

    st.markdown("### Models")
    st.write(
        """
- **ItemKNN**: collaborative filtering baseline  
- **MF (BPR)**: matrix factorization baseline  
- **LightGCN**: graph collaborative filtering  
- **GraphSAGE**: heterogeneous GNN using genre and tag relations  
"""
    )

with tab2:
    st.subheader("Model Comparison")
    df_results = load_model_results()
    st.dataframe(df_results, width="stretch")

    metric = st.selectbox("Choose metric", ["Recall@10", "Recall@20", "NDCG@10", "NDCG@20"], index=0)
    chart_df = df_results[["Model", metric]].set_index("Model")
    st.bar_chart(chart_df)

with tab3:
    st.subheader("User Recommendation Demo")

    model_name = st.selectbox(
        "Choose model",
        ["GraphSAGE", "ItemKNN", "MF (BPR)", "LightGCN"],
    )

    user_id = st.number_input("Enter userId", min_value=1, value=3054, step=1)
    top_k = st.slider("Top K", min_value=5, max_value=20, value=10)

    st.caption("All four models are generated live. The first run may take longer because models are loaded and cached.")

    if st.button("Generate Recommendations"):
        with st.spinner("Generating recommendations..."):
            rec_df, err = get_recommendations(model_name, int(user_id), top_k)

        if err:
            st.error(err)
        else:
            if model_name == "GraphSAGE":
                st.info(
                    "GraphSAGE uses **genre and tag nodes inside the heterogeneous graph**, "
                    "so these features influence the recommendation model."
                )
            else:
                st.info(
                    "Genres and tags are displayed **only as metadata for explanation**. "
                    "They are **not used by this model** when computing recommendations."
                )

            display_cols = [c for c in ["rank", "movieId", "title", "genres", "tags"] if c in rec_df.columns]
            display_df = rec_df[display_cols].copy()

            if model_name != "GraphSAGE":
                if "genres" in display_df.columns:
                    display_df = display_df.rename(columns={"genres": "Genres (metadata)"})
                if "tags" in display_df.columns:
                    display_df = display_df.rename(columns={"tags": "Tags (metadata)"})
            else:
                if "genres" in display_df.columns:
                    display_df = display_df.rename(columns={"genres": "Genres (used in graph)"})
                if "tags" in display_df.columns:
                    display_df = display_df.rename(columns={"tags": "Tags (used in graph)"})

            st.success(f"Top {top_k} recommendations for userId {user_id} using {model_name}")
            st.dataframe(display_df, width="stretch")

            csv_data = display_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download recommendations CSV",
                data=csv_data,
                file_name=f"{model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_user_{user_id}_top{top_k}.csv",
                mime="text/csv",
            )