"""
cluster_analysis.py
--------------------
NICE Clinical Code Assistant — Data Analysis Engine

PURPOSE
-------
This module drives the "🔬 Data Analysis" tab in app.py. It performs
three levels of analysis on the combined_normalized_codes.csv dataset:

  1. EMBEDDING — converts every code's Term text into 384-float vectors
     using the same BAAI/bge-small-en model the retrieval pipeline uses,
     ensuring consistent semantic representation.

  2. CLUSTERING — runs KMeans in the full 384-dimensional embedding space
     to find groups of semantically similar codes. Cluster IDs are used
     as the "clinical cluster" reference for completeness backtesting.

  3. DIMENSIONALITY REDUCTION — runs t-SNE (and optionally UMAP) to
     compress 384 dimensions to 2 for scatter-plot visualisation.

All heavy computation is cached to disk in outputs/analysis_cache/.
First run takes ~60-120 seconds (embedding 5,700 terms). Every
subsequent run loads the cache in under 2 seconds.

OUTPUTS (Plotly figures, consumed by app.py's gr.Plot components)
-----------------------------------------------------------------
  fig_tsne_condition  — t-SNE scatter coloured by NICE condition/source
  fig_tsne_cluster    — t-SNE scatter coloured by KMeans cluster
  fig_cluster_bar     — stacked bar: codes per cluster × condition
  fig_heatmap         — completeness audit heatmap (from BT-03 design)
  fig_obs_scatter     — t-SNE scatter sized by Observations (NHS usage)

HOW TO USE
----------
  import cluster_analysis as ca
  results = ca.get_analysis_results()          # lazy-loads or runs analysis
  fig_tsne = results["fig_tsne_condition"]     # Plotly Figure object
  summary  = results["summary_md"]            # Markdown stats string
"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ─── Paths ───────────────────────────────────────────────────────
_HERE      = Path(__file__).parent
DATA_PATH  = _HERE / "combined_normalized_codes.csv"
CACHE_DIR  = _HERE / "outputs" / "analysis_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_EMBED_CACHE  = CACHE_DIR / "embeddings.npy"
_TSNE_CACHE   = CACHE_DIR / "tsne_coords.npy"
_KMEANS_CACHE = CACHE_DIR / "kmeans_labels.npy"
_DF_CACHE     = CACHE_DIR / "enriched_df.pkl"

# ─── Condition display names ─────────────────────────────────────
# Maps raw filename tokens from the Source column to human-readable
# condition names shown in the UI.
SOURCE_LABELS: dict[str, str] = {
    "ascvd":             "ASCVD",
    "all_prod":          "All Products",
    "antihypertensive":  "Antihypertensives",
    "t2dm":              "Type 2 Diabetes",
    "llt":               "Lipid Lowering Therapy",
    "hypertension":      "Hypertension",
    "dyslipidemia":      "Dyslipidaemia",
    "osa":               "Obstructive Sleep Apnoea",
    "ldl":               "LDL Cholesterol",
    "BMI":               "BMI",
    "hdl":               "HDL Cholesterol",
    "triglycerides":     "Triglycerides",
    "ethnicity":         "Ethnicity",
}

# NHS-aligned condition colour palette.
# Using distinct, accessible colours that avoid red/green collision.
CONDITION_COLOURS: dict[str, str] = {
    "ASCVD":                    "#005EB8",   # NHS Blue
    "Type 2 Diabetes":          "#009639",   # NHS Green
    "Hypertension":             "#DA291C",   # NHS Red
    "Dyslipidaemia":            "#FFB81C",   # NHS Yellow
    "Lipid Lowering Therapy":   "#41B6E6",   # NHS Light Blue
    "Antihypertensives":        "#7B5EA7",   # Purple
    "All Products":             "#AEB7BD",   # Grey
    "Obstructive Sleep Apnoea": "#00A499",   # NHS Teal
    "BMI":                      "#F36633",   # Orange
    "LDL Cholesterol":          "#003087",   # NHS Dark Blue
    "HDL Cholesterol":          "#78BE20",   # Lime Green
    "Triglycerides":            "#8C6E3E",   # Brown
    "Ethnicity":                "#E8C8E0",   # Pink
}

# KMeans K — 8 clusters covers the main NHS clinical groupings
K_CLUSTERS = 8

# t-SNE hyperparameters — perplexity 30 works well for ~5k points
TSNE_PERPLEXITY = 30
TSNE_ITERATIONS = 1000

# ─────────────────────────────────────────────────────────────────
# SECTION 1 — DATA LOADING
# ─────────────────────────────────────────────────────────────────

def _clean_source_name(raw: str) -> str:
    """
    Extract the condition name token from the raw NAAR filename.

    Raw format: 'DAAR_2025_004_{condition}_codes.txt'
    Examples:
      'DAAR_2025_004_t2dm_codes.txt'        → 'Type 2 Diabetes'
      'DAAR_2025_004_ascvd_codes (2).txt'   → 'ASCVD'
      'DAAR_2025_004_all_prod_codes.txt'    → 'All Products'
    """
    # Strip path artefacts and spaces
    s = raw.strip()
    # Remove DAAR prefix and trailing _codes*.txt
    s = s.replace("DAAR_2025_004_", "")
    s = s.split("_codes")[0]
    s = s.strip()
    # Look up in our label map; fall back to title-case of the token
    for key, label in SOURCE_LABELS.items():
        if s.lower().startswith(key.lower()):
            return label
    return s.replace("_", " ").title()


def load_data() -> pd.DataFrame:
    """
    Load and clean the combined_normalized_codes.csv file.

    Returns a DataFrame with columns:
      code        — SNOMED/READ code string
      term        — human-readable term
      condition   — clean condition label (from SOURCE_LABELS)
      observations— float NHS observation count (NaN if not recorded)
      source_raw  — original Source column value
    """
    df = pd.read_csv(DATA_PATH, dtype=str)

    # Normalise column names to lowercase with no spaces
    df.columns = [c.strip() for c in df.columns]

    # Build a clean working DataFrame
    out = pd.DataFrame()
    out["code"]        = df["Code"].astype(str).str.strip()
    out["term"]        = df["Term"].astype(str).str.strip()
    out["source_raw"]  = df["Source"].astype(str).str.strip()
    out["condition"]   = out["source_raw"].apply(_clean_source_name)

    # Observations is a count of NHS records — keep as float, NaN where absent
    if "Observations" in df.columns:
        out["observations"] = pd.to_numeric(df["Observations"], errors="coerce")
    else:
        out["observations"] = np.nan

    # Drop rows with empty terms (can't embed them)
    out = out[out["term"].str.len() > 0].reset_index(drop=True)

    return out


# ─────────────────────────────────────────────────────────────────
# SECTION 2 — EMBEDDING
# ─────────────────────────────────────────────────────────────────

def _compute_embeddings(terms: list[str]) -> np.ndarray:
    """
    Convert a list of clinical term strings into 384-float vectors.

    Uses the same BAAI/bge-small-en SentenceTransformer model as the
    retrieval pipeline, ensuring embeddings are semantically consistent
    with what ChromaDB contains.

    Args:
        terms: list of N strings

    Returns:
        numpy array of shape (N, 384)
    """
    print(f"[cluster_analysis] Embedding {len(terms):,} terms with bge-small-en ...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("BAAI/bge-small-en")
    embeddings = model.encode(
        terms,
        batch_size=128,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity via dot product
    )
    print(f"[cluster_analysis] Embedding complete — shape: {embeddings.shape}")
    return embeddings.astype(np.float32)


def _load_or_compute_embeddings(df: pd.DataFrame) -> np.ndarray:
    """Return cached embeddings or compute and cache them."""
    if _EMBED_CACHE.exists():
        print("[cluster_analysis] Loading embeddings from cache ...")
        emb = np.load(_EMBED_CACHE)
        if emb.shape[0] == len(df):
            return emb
        print("[cluster_analysis] Cache shape mismatch — recomputing ...")

    emb = _compute_embeddings(df["term"].tolist())
    np.save(_EMBED_CACHE, emb)
    return emb


# ─────────────────────────────────────────────────────────────────
# SECTION 3 — KMEANS CLUSTERING
# ─────────────────────────────────────────────────────────────────

# Human-readable names for each of the 8 KMeans clusters.
# These are assigned after inspecting the centroid's nearest codes
# (see _name_clusters() below which prints the top terms per cluster).
# They are intentionally set as module-level state so the UI can
# display them without re-running the analysis.
_CLUSTER_NAMES: dict[int, str] = {}

# Default names used before post-hoc labelling
_DEFAULT_CLUSTER_NAMES = {
    0: "Cluster 0",
    1: "Cluster 1",
    2: "Cluster 2",
    3: "Cluster 3",
    4: "Cluster 4",
    5: "Cluster 5",
    6: "Cluster 6",
    7: "Cluster 7",
}


def _run_kmeans(embeddings: np.ndarray, k: int = K_CLUSTERS) -> np.ndarray:
    """
    Fit KMeans on the embedding matrix and return cluster labels.

    Args:
        embeddings: (N, 384) float32 array
        k:          number of clusters

    Returns:
        labels: (N,) int array of cluster IDs 0..k-1
    """
    print(f"[cluster_analysis] Running KMeans (k={k}) ...")
    from sklearn.cluster import KMeans as _KMeans
    km = _KMeans(
        n_clusters=k,
        n_init=20,        # 20 random initialisations → more stable centroids
        max_iter=500,
        random_state=42,
    )
    labels = km.fit_predict(embeddings)
    print(f"[cluster_analysis] KMeans complete — cluster sizes: {np.bincount(labels).tolist()}")
    return labels.astype(np.int32)


def _auto_name_clusters(df: pd.DataFrame) -> dict[int, str]:
    """
    Automatically name each cluster by finding the most common
    condition among its members, then appending the single most
    representative term word.

    This gives names like "Type 2 Diabetes · glucose" rather than
    "Cluster 3" — much more useful for the UI.
    """
    names = {}
    for cid in range(K_CLUSTERS):
        mask = df["cluster"] == cid
        if not mask.any():
            names[cid] = f"Cluster {cid}"
            continue

        # Most common condition in this cluster
        top_cond = df[mask]["condition"].value_counts().index[0]

        # Most distinctive short word in the terms (exclude stopwords)
        _stopwords = {
            "the", "of", "with", "and", "for", "due", "to", "a", "an",
            "in", "type", "level", "serum", "plasma", "blood", "disorder",
            "finding", "disease", "therapy", "treatment", "medication",
        }
        all_terms = " ".join(df[mask]["term"].str.lower().tolist())
        words = [
            w.strip("(),.-") for w in all_terms.split()
            if len(w) > 3 and w.lower() not in _stopwords
        ]
        from collections import Counter
        if words:
            top_word = Counter(words).most_common(1)[0][0].title()
        else:
            top_word = ""

        names[cid] = f"C{cid}: {top_cond}" + (f" · {top_word}" if top_word else "")

    return names


def _load_or_compute_kmeans(embeddings: np.ndarray) -> np.ndarray:
    """Return cached KMeans labels or compute and cache them."""
    if _KMEANS_CACHE.exists():
        print("[cluster_analysis] Loading KMeans labels from cache ...")
        labels = np.load(_KMEANS_CACHE)
        if labels.shape[0] == embeddings.shape[0]:
            return labels
        print("[cluster_analysis] Cache shape mismatch — recomputing ...")

    labels = _run_kmeans(embeddings)
    np.save(_KMEANS_CACHE, labels)
    return labels


# ─────────────────────────────────────────────────────────────────
# SECTION 4 — t-SNE DIMENSIONALITY REDUCTION
# ─────────────────────────────────────────────────────────────────

def _run_tsne(embeddings: np.ndarray) -> np.ndarray:
    """
    Reduce embedding matrix to 2D using t-SNE for visualisation.

    t-SNE is run in the full 384-dimensional space (not on a PCA
    reduction) to preserve the fine-grained structure that would be
    lost if we pre-compressed to e.g. 50 dimensions first.

    Args:
        embeddings: (N, 384) float32 array

    Returns:
        coords: (N, 2) float32 array — [tsne_x, tsne_y]
    """
    print(f"[cluster_analysis] Running t-SNE (perplexity={TSNE_PERPLEXITY}) ...")
    from sklearn.manifold import TSNE
    tsne = TSNE(
        n_components=2,
        perplexity=TSNE_PERPLEXITY,
        n_iter=TSNE_ITERATIONS,
        init="pca",          # PCA init → more reproducible layout
        learning_rate="auto",
        random_state=42,
        n_jobs=-1,
    )
    coords = tsne.fit_transform(embeddings).astype(np.float32)
    print(f"[cluster_analysis] t-SNE complete — coords shape: {coords.shape}")
    return coords


def _load_or_compute_tsne(embeddings: np.ndarray) -> np.ndarray:
    """Return cached t-SNE coords or compute and cache them."""
    if _TSNE_CACHE.exists():
        print("[cluster_analysis] Loading t-SNE coords from cache ...")
        coords = np.load(_TSNE_CACHE)
        if coords.shape[0] == embeddings.shape[0]:
            return coords
        print("[cluster_analysis] Cache shape mismatch — recomputing ...")

    coords = _run_tsne(embeddings)
    np.save(_TSNE_CACHE, coords)
    return coords


# ─────────────────────────────────────────────────────────────────
# SECTION 5 — PLOTLY FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────

_PLOT_LAYOUT = dict(
    font_family="Arial",
    paper_bgcolor="white",
    plot_bgcolor="#F0F4F5",
    margin=dict(l=20, r=20, t=50, b=20),
)


def _fig_tsne_by_condition(df: pd.DataFrame) -> go.Figure:
    """
    t-SNE scatter plot coloured by clinical condition.

    Each point is one code. Hover shows the code ID, full term,
    condition, and NHS observation count (if available).
    Points are semi-transparent to show dense cluster regions.
    """
    fig = px.scatter(
        df,
        x="tsne_x", y="tsne_y",
        color="condition",
        color_discrete_map=CONDITION_COLOURS,
        hover_name="term",
        hover_data={
            "code":         True,
            "condition":    True,
            "observations": ":.0f",
            "tsne_x":       False,
            "tsne_y":       False,
        },
        title="t-SNE: SNOMED Codes by Clinical Condition",
        labels={"tsne_x": "", "tsne_y": "", "condition": "Condition"},
        opacity=0.65,
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(
        **_PLOT_LAYOUT,
        legend=dict(
            orientation="v",
            x=1.01, y=0.5,
            font_size=11,
        ),
        xaxis=dict(showticklabels=False, showgrid=True, gridcolor="#D8DDE0"),
        yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#D8DDE0"),
        height=540,
    )
    return fig


def _fig_tsne_by_cluster(df: pd.DataFrame, cluster_names: dict[int, str]) -> go.Figure:
    """
    t-SNE scatter plot coloured by KMeans cluster.

    Uses a qualitative palette so every cluster has a distinct colour
    regardless of how many clusters there are.
    """
    df = df.copy()
    df["cluster_label"] = df["cluster"].map(cluster_names)

    fig = px.scatter(
        df,
        x="tsne_x", y="tsne_y",
        color="cluster_label",
        hover_name="term",
        hover_data={
            "code":          True,
            "condition":     True,
            "cluster_label": True,
            "tsne_x":        False,
            "tsne_y":        False,
        },
        title="t-SNE: SNOMED Codes by KMeans Cluster",
        labels={"tsne_x": "", "tsne_y": "", "cluster_label": "Cluster"},
        color_discrete_sequence=px.colors.qualitative.Bold,
        opacity=0.65,
    )
    fig.update_traces(marker_size=4)
    fig.update_layout(
        **_PLOT_LAYOUT,
        legend=dict(
            orientation="v",
            x=1.01, y=0.5,
            font_size=10,
        ),
        xaxis=dict(showticklabels=False, showgrid=True, gridcolor="#D8DDE0"),
        yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#D8DDE0"),
        height=540,
    )
    return fig


def _fig_cluster_composition(df: pd.DataFrame, cluster_names: dict[int, str]) -> go.Figure:
    """
    Stacked horizontal bar chart showing the condition composition
    of every cluster.

    This answers: "Does KMeans cluster X correspond to one condition,
    or does it mix multiple conditions?" A cluster mixing obesity and
    T2DM codes is the metabolic bridge cluster predicted by BT-03.
    """
    df = df.copy()
    df["cluster_label"] = df["cluster"].map(cluster_names)

    # Count codes per (cluster, condition)
    counts = (
        df.groupby(["cluster_label", "condition"])
        .size()
        .reset_index(name="count")
    )

    fig = px.bar(
        counts,
        x="count",
        y="cluster_label",
        color="condition",
        orientation="h",
        color_discrete_map=CONDITION_COLOURS,
        title="Cluster Composition — Codes per Condition per Cluster",
        labels={
            "count":         "Number of Codes",
            "cluster_label": "KMeans Cluster",
            "condition":     "Condition",
        },
    )
    fig.update_layout(
        **_PLOT_LAYOUT,
        height=max(400, len(cluster_names) * 55),
        legend=dict(orientation="v", x=1.01, y=0.5, font_size=11),
        yaxis=dict(autorange="reversed"),
        xaxis=dict(showgrid=True, gridcolor="#D8DDE0"),
        bargap=0.2,
    )
    return fig


def _fig_completeness_heatmap(df: pd.DataFrame, cluster_names: dict[int, str]) -> go.Figure:
    """
    Completeness audit heatmap (BT-03 design).

    Rows = KMeans clusters, Columns = conditions.
    Cell value = proportion of codes in that condition which fall
    into that cluster. This shows which clusters each condition
    "owns" — and which clusters appear in multiple conditions
    (those are the multimorbidity bridging clusters).

    Bright cells on one row that span multiple condition columns
    = bridging cluster.
    """
    conditions = df["condition"].unique().tolist()
    df = df.copy()
    df["cluster_label"] = df["cluster"].map(cluster_names)
    cluster_labels = sorted(df["cluster_label"].unique())

    # Build matrix: proportion of each condition's codes in each cluster
    matrix = np.zeros((len(cluster_labels), len(conditions)))
    for ci, cond in enumerate(conditions):
        cond_total = (df["condition"] == cond).sum()
        if cond_total == 0:
            continue
        for ri, clabel in enumerate(cluster_labels):
            n = ((df["condition"] == cond) & (df["cluster_label"] == clabel)).sum()
            matrix[ri, ci] = n / cond_total

    # Annotate with percentage strings
    text_matrix = [[f"{v:.0%}" if v > 0.01 else "" for v in row] for row in matrix]

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=conditions,
        y=cluster_labels,
        text=text_matrix,
        texttemplate="%{text}",
        colorscale=[
            [0.0, "#F0F4F5"],   # pale NHS background — near-zero
            [0.2, "#41B6E6"],   # NHS light blue
            [0.5, "#005EB8"],   # NHS blue
            [1.0, "#003087"],   # NHS dark blue
        ],
        showscale=True,
        colorbar=dict(title="Proportion of condition's codes"),
        hovertemplate="Cluster: %{y}<br>Condition: %{x}<br>Proportion: %{z:.1%}<extra></extra>",
    ))
    fig.update_layout(
        **_PLOT_LAYOUT,
        title="Cluster Completeness Heatmap — Condition Proportion per Cluster<br>"
              "<sub>Clusters appearing prominently across multiple conditions are multimorbidity bridging clusters</sub>",
        xaxis=dict(
            tickangle=-30,
            title="",
            showgrid=False,
        ),
        yaxis=dict(
            title="",
            showgrid=False,
            autorange="reversed",
        ),
        height=max(450, len(cluster_labels) * 55),
    )
    return fig


def _fig_observations_scatter(df: pd.DataFrame) -> go.Figure:
    """
    t-SNE scatter with point size proportional to NHS Observations
    (number of patient records). Only codes with a recorded
    observation count are shown.

    This combines the geographic (t-SNE) and usage-frequency
    dimensions in one view — large points are high-usage codes
    whose position in the embedding space shows which clinical
    cluster they belong to.
    """
    obs_df = df[df["observations"].notna() & (df["observations"] > 0)].copy()

    # Log-scale the size to prevent extreme outliers dominating
    obs_df["log_obs"] = np.log10(obs_df["observations"] + 1)
    obs_df["obs_display"] = obs_df["observations"].apply(
        lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
    )

    fig = px.scatter(
        obs_df,
        x="tsne_x", y="tsne_y",
        color="condition",
        size="log_obs",
        size_max=22,
        color_discrete_map=CONDITION_COLOURS,
        hover_name="term",
        hover_data={
            "code":        True,
            "condition":   True,
            "obs_display": True,
            "tsne_x":      False,
            "tsne_y":      False,
            "log_obs":     False,
        },
        title="t-SNE: Code Usage Frequency (bubble size = NHS Observations, log scale)",
        labels={
            "tsne_x":      "",
            "tsne_y":      "",
            "condition":   "Condition",
            "obs_display": "Observations",
        },
        opacity=0.75,
    )
    fig.update_layout(
        **_PLOT_LAYOUT,
        legend=dict(orientation="v", x=1.01, y=0.5, font_size=11),
        xaxis=dict(showticklabels=False, showgrid=True, gridcolor="#D8DDE0"),
        yaxis=dict(showticklabels=False, showgrid=True, gridcolor="#D8DDE0"),
        height=540,
    )
    return fig


# ─────────────────────────────────────────────────────────────────
# SECTION 6 — SUMMARY STATISTICS
# ─────────────────────────────────────────────────────────────────

def _build_summary_md(df: pd.DataFrame, cluster_names: dict[int, str]) -> str:
    """
    Generate a Markdown summary panel shown above the plots.
    Includes: total codes, conditions, cluster sizes, and the
    bridge-cluster detection result from BT-03.
    """
    total       = len(df)
    n_conds     = df["condition"].nunique()
    n_clusters  = df["cluster"].nunique()

    lines = [
        f"**{total:,} codes** across **{n_conds} conditions** · **{n_clusters} KMeans clusters**",
        "",
        "**Codes per condition:**",
        "| Condition | Codes | With Observations |",
        "|-----------|-------|------------------|",
    ]
    for cond, grp in df.groupby("condition"):
        n_obs = grp["observations"].notna().sum()
        lines.append(f"| {cond} | {len(grp):,} | {n_obs:,} |")

    lines += [
        "",
        "**Cluster overview:**",
        "| Cluster | Size | Top condition |",
        "|---------|------|---------------|",
    ]
    for cid in range(n_clusters):
        mask  = df["cluster"] == cid
        size  = mask.sum()
        if size == 0:
            continue
        top_c = df[mask]["condition"].value_counts().index[0]
        lines.append(f"| {cluster_names.get(cid, f'C{cid}')} | {size:,} | {top_c} |")

    # Bridge cluster detection — find clusters that appear in ≥3 conditions
    bridge_lines = []
    for cid in range(n_clusters):
        mask  = df["cluster"] == cid
        n_cds = df[mask]["condition"].nunique()
        if n_cds >= 3:
            top_conds = ", ".join(df[mask]["condition"].value_counts().head(3).index.tolist())
            bridge_lines.append(
                f"- **{cluster_names.get(cid, f'C{cid}')}** — {n_cds} conditions "
                f"({top_conds}, ...)"
            )

    if bridge_lines:
        lines += [
            "",
            "**🔗 Multimorbidity bridging clusters detected** *(appear in ≥3 conditions)*:",
        ] + bridge_lines

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────

# Module-level cache so repeated calls to get_analysis_results()
# within one app session return immediately without re-running.
_RESULTS_CACHE: dict | None = None


def get_analysis_results(force_recompute: bool = False) -> dict:
    """
    Run the full analysis pipeline and return all Plotly figures.

    This is the only function app.py calls. It is safe to call
    multiple times — all computation is cached after the first call.

    Args:
        force_recompute: if True, delete disk caches and recompute
                         everything from scratch.

    Returns a dict with keys:
        fig_tsne_condition  — Plotly Figure
        fig_tsne_cluster    — Plotly Figure
        fig_cluster_bar     — Plotly Figure
        fig_heatmap         — Plotly Figure
        fig_obs_scatter     — Plotly Figure
        summary_md          — Markdown string
        df                  — enriched DataFrame (for further analysis)
        cluster_names       — dict {cluster_id: label}
    """
    global _RESULTS_CACHE
    if _RESULTS_CACHE is not None and not force_recompute:
        return _RESULTS_CACHE

    if force_recompute:
        for p in [_EMBED_CACHE, _TSNE_CACHE, _KMEANS_CACHE, _DF_CACHE]:
            p.unlink(missing_ok=True)
        _RESULTS_CACHE = None

    # ── Step 1: load data ─────────────────────────────────────────
    print("[cluster_analysis] Loading data ...")
    df = load_data()
    print(f"[cluster_analysis] Loaded {len(df):,} codes across {df['condition'].nunique()} conditions")

    # ── Step 2: embeddings ────────────────────────────────────────
    embeddings = _load_or_compute_embeddings(df)

    # ── Step 3: KMeans ───────────────────────────────────────────
    kmeans_labels = _load_or_compute_kmeans(embeddings)
    df["cluster"] = kmeans_labels.astype(int)

    # ── Step 4: t-SNE ─────────────────────────────────────────────
    tsne_coords = _load_or_compute_tsne(embeddings)
    df["tsne_x"] = tsne_coords[:, 0]
    df["tsne_y"] = tsne_coords[:, 1]

    # ── Step 5: auto-name clusters ────────────────────────────────
    cluster_names = _auto_name_clusters(df)

    # ── Step 6: build all figures ─────────────────────────────────
    print("[cluster_analysis] Building Plotly figures ...")
    results = {
        "fig_tsne_condition": _fig_tsne_by_condition(df),
        "fig_tsne_cluster":   _fig_tsne_by_cluster(df, cluster_names),
        "fig_cluster_bar":    _fig_cluster_composition(df, cluster_names),
        "fig_heatmap":        _fig_completeness_heatmap(df, cluster_names),
        "fig_obs_scatter":    _fig_observations_scatter(df),
        "summary_md":         _build_summary_md(df, cluster_names),
        "df":                 df,
        "cluster_names":      cluster_names,
    }

    _RESULTS_CACHE = results
    print("[cluster_analysis] Analysis complete.")
    return results


def clear_cache() -> None:
    """Delete all cached files, forcing a full recompute on next call."""
    global _RESULTS_CACHE
    _RESULTS_CACHE = None
    for p in [_EMBED_CACHE, _TSNE_CACHE, _KMEANS_CACHE]:
        p.unlink(missing_ok=True)
    print("[cluster_analysis] Cache cleared.")


# ─────────────────────────────────────────────────────────────────
# SECTION 8 — BT-03 BACKTESTING UTILITIES
# (can be called standalone from command line or from app.py)
# ─────────────────────────────────────────────────────────────────

def compute_cluster_distribution(code_set: set, df: pd.DataFrame) -> dict[int, int]:
    """
    Given a set of code strings and the enriched DataFrame,
    return a dict of {cluster_id: count_of_codes_in_that_cluster}.

    This is the "cluster fingerprint" of a code list, as described
    in BT-03. Used in backtest_cluster_completeness() below.
    """
    from collections import Counter
    subset = df[df["code"].isin(code_set)]
    return dict(Counter(subset["cluster"].tolist()))


def backtest_cluster_completeness(
    nice_lists:   dict[str, set],
    agent_lists:  dict[str, set],
    df:           pd.DataFrame,
    cluster_names: dict[int, str],
) -> pd.DataFrame:
    """
    Compare the cluster distributions of NICE gold-standard code lists
    to agent-generated code lists, following the BT-03 design.

    Produces a DataFrame with one row per (condition, cluster) pair,
    showing whether the agent under- or over-represents each cluster
    relative to the gold standard.

    Args:
        nice_lists:    {condition_name: set_of_codes}  — gold standard
        agent_lists:   {condition_name: set_of_codes}  — agent output
        df:            enriched DataFrame (from get_analysis_results)
        cluster_names: {cluster_id: label}

    Returns:
        DataFrame with columns: condition, cluster_id, cluster_label,
        nice_proportion, agent_proportion, discrepancy, gap_direction
    """
    rows = []
    for condition, nice_codes in nice_lists.items():
        nice_dist  = compute_cluster_distribution(nice_codes, df)
        agent_dist = compute_cluster_distribution(
            agent_lists.get(condition, set()), df
        )
        nice_total  = sum(nice_dist.values())  or 1
        agent_total = sum(agent_dist.values()) or 1

        for cid in set(nice_dist) | set(agent_dist):
            nice_p  = nice_dist.get(cid, 0)  / nice_total
            agent_p = agent_dist.get(cid, 0) / agent_total
            diff    = agent_p - nice_p
            rows.append({
                "condition":      condition,
                "cluster_id":     cid,
                "cluster_label":  cluster_names.get(cid, f"Cluster {cid}"),
                "nice_proportion":  round(nice_p,  4),
                "agent_proportion": round(agent_p, 4),
                "discrepancy":      round(diff,    4),
                "gap_direction": (
                    "UNDER_REPRESENTED" if diff < -0.05
                    else "OVER_REPRESENTED" if diff > 0.05
                    else "BALANCED"
                ),
            })

    return pd.DataFrame(rows)


def find_bridge_clusters(
    condition_a_codes: set,
    condition_b_codes: set,
    df:                pd.DataFrame,
    min_codes:         int = 2,
) -> dict:
    """
    Identify clusters that are meaningfully represented in BOTH
    condition code lists — these are the multimorbidity bridging
    clusters described in BT-03.

    Args:
        condition_a_codes: set of code strings for condition A
        condition_b_codes: set of code strings for condition B
        df:                enriched DataFrame
        min_codes:         minimum codes in a cluster to count it

    Returns dict with bridge_clusters, bridge_codes, and bridge_recall.
    """
    dist_a = compute_cluster_distribution(condition_a_codes, df)
    dist_b = compute_cluster_distribution(condition_b_codes, df)

    clusters_a = {c for c, n in dist_a.items() if n >= min_codes}
    clusters_b = {c for c, n in dist_b.items() if n >= min_codes}
    bridges    = clusters_a & clusters_b

    cond_b_df   = df[df["code"].isin(condition_b_codes)]
    bridge_nice = set(cond_b_df[cond_b_df["cluster"].isin(bridges)]["code"])

    return {
        "bridge_cluster_ids": sorted(bridges),
        "bridge_codes_count": len(bridge_nice),
        "bridge_codes":       bridge_nice,
    }


# ─────────────────────────────────────────────────────────────────
# CLI entry point — run directly to pre-build the cache
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Building analysis cache — this may take 1–2 minutes on first run.\n")
    results = get_analysis_results()
    print("\nDone. Figures ready:")
    for k, v in results.items():
        if k != "df" and k != "cluster_names" and k != "summary_md":
            print(f"  {k}: {type(v).__name__}")
    print("\nSummary:")
    print(results["summary_md"])
