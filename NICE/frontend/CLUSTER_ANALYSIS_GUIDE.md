# NICE Clinical Code Assistant — Data Analysis Deep Dive
## A Complete Technical Reference for `cluster_analysis.py` and the `app.py` Data Analysis Tab

> **Who this is for.** This document assumes you understand Python and have used pandas before, but have not necessarily worked with machine learning, embeddings, or dimensionality reduction. Every concept is introduced from first principles before the code is shown. By the end you should be able to explain to a colleague — in plain English — exactly why each line exists and what it does to your data.

---

## Part 1 — The Problem We Are Solving

Before we look at any code, it is worth being precise about the question the analysis is trying to answer, because every design decision in both files flows from it.

Your NICE project has thirteen code lists — sets of SNOMED codes that clinical analysts have validated for conditions including Type 2 Diabetes, Hypertension, ASCVD, Obstructive Sleep Apnoea, and others. These code lists did not fall out of thin air. A human expert read clinical records, consulted guidelines, and decided that this specific set of codes, and not some other set, correctly defines the cohort for this condition.

The question we are asking is: **do the codes within each list form coherent groups in semantic space, and do the groups across different conditions reveal the clinical overlaps we expect?**

The reason this matters is spelled out in your BT-03 document under "The Multimorbidity Blind Spot." Chronic conditions do not occur independently. A patient with obesity also tends to have hypertension, and a patient with hypertension tends to have cardiovascular disease. The codes that exist specifically *at the intersection* of two conditions — codes that only become clinically relevant when both are present — are the hardest for an automated system to retrieve, because they are semantically equidistant from both condition query strings and may surface in neither search.

If we can map all 5,700 codes into a two-dimensional space that preserves their semantic relationships, we can *see* those intersections. We can ask whether our retrieval system is finding codes from the right region of that space. And we can formally measure the gap between what a gold-standard human expert chose and what the system retrieved.

That is what these two files do together.

---

## Part 2 — The Architecture at a Glance

The two files have entirely separate responsibilities that touch at a single, clean boundary.

```
cluster_analysis.py                     app.py
────────────────────────────────        ─────────────────────────────────────
Owns:                                   Owns:
  Data loading                            Gradio UI layout
  Embedding computation                   Tab structure
  KMeans clustering                       Button event wiring
  t-SNE projection                        Figure display (gr.Plot)
  Figure building                         run_data_analysis() wrapper
  Caching to disk
  BT-03 backtesting utilities

Exports one public function:
  get_analysis_results()          ──►   Called by run_data_analysis()
  └── returns dict of figures            └── unpacks and returns 7 values
      and summary Markdown                   to 7 Gradio output components
```

The key architectural rule is that `cluster_analysis.py` knows nothing about Gradio. It returns plain Python objects — Plotly Figure instances and strings. `app.py` knows nothing about how those figures are produced. This separation means you could replace the entire analysis engine (different clustering algorithm, different embedding model) without touching any UI code.

---

## Part 3 — Understanding Embeddings

This is the foundation of everything that follows. Before we can cluster codes or project them into 2D, we need to convert their text descriptions into numbers that a computer can reason about mathematically.

### 3.1 — What a vector is

A vector is simply a list of numbers. A 3-dimensional vector `[0.2, -0.8, 0.5]` describes a point in 3D space — you can imagine it as coordinates on an x, y, z graph. Our embedding model produces 384-dimensional vectors. You cannot visualise 384 dimensions, but the mathematics works the same way: each code description becomes a point in a 384-dimensional space.

The crucial property is **proximity**: two codes whose descriptions are semantically similar — "Type 2 diabetes mellitus" and "Non-insulin dependent diabetes mellitus" — will produce vectors that are numerically close to each other. Two codes from entirely different clinical domains — "Serum triglyceride level" and "Obstructive sleep apnoea" — will produce vectors that are far apart. This proximity is measured using cosine similarity, which measures the angle between two vectors rather than their distance (this is better for text because it is insensitive to the length of the description).

### 3.2 — The embedding model: `BAAI/bge-small-en`

The model used is `BAAI/bge-small-en` from the SentenceTransformers library. It was trained on hundreds of millions of text pairs — pairs where a human said "these two sentences mean the same thing" or "these are related." Through this training it learned to map text into a 384-dimensional space where related texts land close together.

We use this specific model because it is already used by the retrieval pipeline in ChromaDB. This is a design decision with real consequences: the clusters you find in the analysis will correspond to the same semantic groupings that the retrieval system uses when it searches for codes. If a cluster contains codes that are always retrieved together, that is because the retrieval system sees them as semantically close — which is exactly what the cluster reflects.

### 3.3 — How the code does it

```python
# cluster_analysis.py — Section 2: _compute_embeddings()

def _compute_embeddings(terms: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-small-en")

    embeddings = model.encode(
        terms,
        batch_size=128,           # process 128 terms at a time (GPU/CPU memory management)
        show_progress_bar=True,
        normalize_embeddings=True # crucial: makes all vectors unit length
    )

    return embeddings.astype(np.float32)
```

**Input:** A Python list of 5,700+ strings, e.g.:
```python
terms = [
    "Type 2 diabetes mellitus",
    "Essential hypertension",
    "Serum triglyceride level",
    "Obstructive sleep apnoea",
    ...
]
```

**Output:** A numpy array of shape `(5781, 384)` — 5,781 rows (one per code), 384 columns (one per embedding dimension).

```python
# What the output looks like conceptually:
embeddings = np.array([
    # "Type 2 diabetes mellitus"     → 384 floats
    [ 0.023, -0.147,  0.892, -0.031,  0.201, ...],
    # "Essential hypertension"       → 384 floats
    [-0.118,  0.334, -0.041,  0.887, -0.009, ...],
    # "Serum triglyceride level"     → 384 floats
    [ 0.441,  0.012,  0.007, -0.223,  0.678, ...],
    ...
])
# shape: (5781, 384)
```

**Why `normalize_embeddings=True`?** Normalisation makes every vector exactly unit length (its magnitude equals 1). After normalisation, cosine similarity between two vectors equals their dot product — a much faster computation. KMeans and t-SNE both work better on normalised vectors because every code has equal influence regardless of how long its description is.

### 3.4 — Caching to disk

Computing embeddings for 5,700 terms takes 60–120 seconds on a CPU. We only want to do that once. The cache logic wraps every heavy computation with a load-or-compute pattern:

```python
# cluster_analysis.py — _load_or_compute_embeddings()

_EMBED_CACHE = CACHE_DIR / "embeddings.npy"  # outputs/analysis_cache/embeddings.npy

def _load_or_compute_embeddings(df: pd.DataFrame) -> np.ndarray:
    if _EMBED_CACHE.exists():
        emb = np.load(_EMBED_CACHE)            # fast: loads ~8MB file in <0.1 seconds
        if emb.shape[0] == len(df):            # sanity check: same number of codes?
            return emb
        # If rows don't match, the CSV changed — recompute

    emb = _compute_embeddings(df["term"].tolist())
    np.save(_EMBED_CACHE, emb)                 # save as numpy binary — exact, fast
    return emb
```

The shape check `emb.shape[0] == len(df)` is the guard against a subtle bug: if someone adds codes to the CSV, the cached embeddings from the previous run would have the wrong number of rows. The mismatch triggers a recompute.

The same pattern is applied identically for KMeans labels (`_load_or_compute_kmeans`) and t-SNE coordinates (`_load_or_compute_tsne`). Each saves a `.npy` file. After the first full run, the entire analysis loads in under 2 seconds.

---

## Part 4 — KMeans Clustering

### 4.1 — What KMeans does conceptually

Imagine you have 5,700 points scattered across a 384-dimensional space. KMeans asks: "What are the best 8 cluster centres I can place so that every point is assigned to its nearest centre, and the total distance from each point to its assigned centre is minimised?"

The algorithm works iteratively:
1. Place 8 centres randomly.
2. Assign every point to its nearest centre.
3. Move each centre to the average position of all its assigned points.
4. Repeat steps 2–3 until the assignments stop changing.

The result is 8 clusters where codes within a cluster are more similar to each other than to codes in any other cluster. Because we are clustering in the 384-dimensional embedding space (not in the 2D t-SNE space), the clusters reflect genuine semantic similarity — not just visual proximity in the plot.

**Critical rule:** We always cluster in the original high-dimensional space and only use t-SNE for visualisation. Clustering on t-SNE coordinates would give wrong results because t-SNE deliberately distorts distances to make the 2D layout readable. Two points that appear close in the t-SNE plot may not actually be close in 384D, and vice versa.

### 4.2 — Choosing k=8

Why 8 clusters? The NICE dataset contains 13 conditions, but several of them cluster tightly in semantic space:
- LDL, HDL, and triglycerides are all lipid measurement codes — they will naturally land in the same cluster
- ASCVD and antihypertensives share a cardiovascular medication cluster
- Ethnicity codes form a completely isolated cluster with no overlap elsewhere

After these natural merges, 8 clusters is approximately right for the clinical structure of the data. You could run the elbow method (plot inertia vs k) to find a more principled answer, but 8 gives interpretable, clinically meaningful groups without over-segmenting.

```python
# cluster_analysis.py — _run_kmeans()

def _run_kmeans(embeddings: np.ndarray, k: int = K_CLUSTERS) -> np.ndarray:
    from sklearn.cluster import KMeans as _KMeans

    km = _KMeans(
        n_clusters=k,
        n_init=20,        # try 20 different random starting positions
        max_iter=500,     # run up to 500 iterations per attempt
        random_state=42,  # reproducible results
    )

    labels = km.fit_predict(embeddings)
    # fit_predict is equivalent to: km.fit(embeddings) then km.predict(embeddings)
    # It returns an integer for every row: which cluster (0–7) that code belongs to

    return labels.astype(np.int32)

# Example output:
# labels = [3, 3, 7, 0, 0, 5, 1, 3, 2, ...]
# This means:
#   code 0 ("Type 2 diabetes mellitus") → cluster 3
#   code 1 ("Non-insulin dependent DM") → cluster 3
#   code 2 ("Serum triglycerides level") → cluster 7
#   code 3 ("Essential hypertension")   → cluster 0
```

The `n_init=20` is important. KMeans is sensitive to its starting positions. Running 20 initialisations and keeping the best result (the one with lowest total intra-cluster distance) gives much more stable, reproducible clusters than the default `n_init=10`.

### 4.3 — Auto-naming clusters

The cluster IDs `{0, 1, 2, ..., 7}` are meaningless to a human reader. The `_auto_name_clusters()` function generates readable names by inspecting the contents of each cluster:

```python
# cluster_analysis.py — _auto_name_clusters()

def _auto_name_clusters(df: pd.DataFrame) -> dict[int, str]:
    names = {}
    for cid in range(K_CLUSTERS):
        mask = df["cluster"] == cid

        # Which condition appears most often in this cluster?
        top_cond = df[mask]["condition"].value_counts().index[0]

        # What is the most frequent non-trivial word across all terms in this cluster?
        _stopwords = {"the", "of", "with", "and", "level", "serum", "finding", ...}
        all_terms  = " ".join(df[mask]["term"].str.lower().tolist())
        words      = [w for w in all_terms.split() if len(w) > 3 and w not in _stopwords]
        top_word   = Counter(words).most_common(1)[0][0].title()

        names[cid] = f"C{cid}: {top_cond} · {top_word}"

    return names

# Example output:
# {
#   0: "C0: ASCVD · Cholesterol",
#   1: "C1: Antihypertensives · Amlodipine",
#   2: "C2: Type 2 Diabetes · Glucose",
#   3: "C3: Hypertension · Blood",
#   4: "C4: Lipid Lowering Therapy · Statin",
#   5: "C5: Ethnicity · British",
#   6: "C6: ASCVD · Ischaemic",
#   7: "C7: Dyslipidaemia · Triglyceride",
# }
```

These names are used as labels on the cluster scatter plot, the composition bar chart, the heatmap rows, and the summary statistics table. They are stored in a module-level dict `cluster_names` and passed as an argument to every figure-building function so the same labels appear consistently across all visualisations.

---

## Part 5 — t-SNE Dimensionality Reduction

### 5.1 — The problem: you cannot plot 384 dimensions

KMeans has assigned every code to one of 8 clusters, and those cluster assignments are meaningful. But how do you show a human analyst where those clusters are and how they relate to each other? You cannot scatter-plot 384-dimensional data directly.

The naive solution is to just use the first two dimensions of the embedding matrix as x and y coordinates. This almost never works well — dimension 0 and dimension 1 of an embedding typically capture the strongest single pattern in the data (often something trivial like description length), and projecting onto just two of 384 dimensions throws away almost all the information.

What we need is a method that compresses 384 dimensions to 2 while **preserving the structure** — keeping nearby points nearby and keeping distant points distant.

### 5.2 — What t-SNE does

**t-SNE** (t-distributed Stochastic Neighbour Embedding, Maaten & Hinton, 2008) works in two phases:

**Phase 1:** For each point in the original high-dimensional space, measure how likely every other point is to be its "neighbour" — assign a probability based on distance, so nearby points get high probability and distant points get near-zero probability. Do this for every pair. The result is a probability distribution over neighbours in the original space.

**Phase 2:** Place all the points randomly in 2D. Assign similar probabilities to neighbours in 2D. Then iteratively adjust the 2D positions to minimise the difference between the original-space probability distribution and the 2D probability distribution. Points that were close in 384D attract each other in 2D; points that were distant repel each other.

The name "t-distributed" refers to a key detail: the 2D neighbour probabilities use a Student's t-distribution (with heavier tails than a Gaussian). This is what prevents the "crowding problem" — without the heavy tails, all the points would collapse into the centre because there is more 2D space to fill than the 384D structure implies.

### 5.3 — The perplexity parameter

The most important t-SNE hyperparameter is **perplexity**. It controls how many neighbours each point "pays attention to" when computing the Phase 1 probability distributions. Low perplexity = each point considers only its very nearest neighbours (tight local structure preserved, global structure lost). High perplexity = each point considers a wider neighbourhood (global structure better preserved, fine-grained local clusters less sharp).

A rule of thumb is to set perplexity to roughly `sqrt(N)` where N is the number of points. For ~5,700 points, `sqrt(5700) ≈ 75`. We use 30, which is the conventional default and tends to produce clean, interpretable plots for datasets in the 1,000–10,000 range. You could experiment with higher values (50–100) if you want to see more of the global structure at the expense of within-cluster clarity.

```python
# cluster_analysis.py — _run_tsne()

def _run_tsne(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.manifold import TSNE

    tsne = TSNE(
        n_components=2,           # we want 2D output
        perplexity=30,            # neighbourhood size
        n_iter=1000,              # optimisation iterations
        init="pca",               # start from a PCA projection (reproducible)
        learning_rate="auto",     # sklearn auto-sets this based on dataset size
        random_state=42,          # reproducible layout
        n_jobs=-1,                # use all CPU cores
    )

    coords = tsne.fit_transform(embeddings)
    # Input:  (5781, 384) float32 array
    # Output: (5781, 2)   float32 array

    return coords.astype(np.float32)

# Example output:
# coords = [
#   [ 12.4,  -8.1],   # "Type 2 diabetes mellitus"     → position in 2D
#   [ 13.1,  -7.9],   # "Non-insulin dependent DM"     → close to above
#   [-22.6,  15.4],   # "Serum triglycerides level"    → different region
#   [ 19.8,   2.3],   # "Essential hypertension"       → yet another region
#   ...
# ]
# shape: (5781, 2)
```

The `init="pca"` setting deserves explanation. By default, t-SNE starts from random positions in 2D, which means two runs on the same data produce completely different layouts (same clusters, different orientations and positions). Using `init="pca"` seeds the starting positions from the first two principal components of the data, which gives much more reproducible results across runs. Combined with `random_state=42`, the output is deterministic.

### 5.4 — What t-SNE does NOT preserve

This is critically important. t-SNE preserves **local neighbourhood structure** well, but it does **not** preserve:
- The distance between different clusters (two clusters being far apart in the plot does not mean they are far apart in 384D)
- The sizes of clusters (a small tight cluster in the plot may represent more points than a large spread-out cluster)
- The absolute positions of anything (the axes have no units; the numbers on the axes are meaningless)

This is why we use t-SNE only for visualisation, never for analysis. The KMeans clusters are found in 384D and are the analytical truth. The t-SNE layout is a picture that helps a human see the structure that KMeans found.

---

## Part 6 — The Five Plotly Figures

All five figures are built in Section 5 of `cluster_analysis.py`. They share a common layout dictionary `_PLOT_LAYOUT` that enforces the NHS visual style (white paper background, pale blue plot background, Arial font) consistently across all figures.

```python
_PLOT_LAYOUT = dict(
    font_family="Arial",
    paper_bgcolor="white",
    plot_bgcolor="#F0F4F5",    # NHS pale
    margin=dict(l=20, r=20, t=50, b=20),
)
# Every figure calls fig.update_layout(**_PLOT_LAYOUT, ...)
# The ** unpacks the dict into keyword arguments
```

### 6.1 — Figure 1: t-SNE by Condition

```python
# cluster_analysis.py — _fig_tsne_by_condition()

# Input:  df — the enriched DataFrame with columns: tsne_x, tsne_y, condition, code, term, observations
# Output: a Plotly Figure object

fig = px.scatter(
    df,
    x="tsne_x",                        # x position from t-SNE
    y="tsne_y",                        # y position from t-SNE
    color="condition",                 # colour each point by which NICE list it came from
    color_discrete_map=CONDITION_COLOURS,  # our NHS colour palette dict
    hover_name="term",                 # shows full term as the tooltip title
    hover_data={
        "code":         True,          # show code ID in tooltip
        "condition":    True,          # show condition
        "observations": ":.0f",        # show observations as whole number
        "tsne_x":       False,         # hide the raw coordinates (meaningless to analysts)
        "tsne_y":       False,
    },
    opacity=0.65,                      # semi-transparent: dense regions become darker
)
fig.update_traces(marker_size=4)       # 4px dots: big enough to see, small enough to not overlap
```

**What this plot tells you:** Each dot is one code. Dots of the same colour come from the same NICE condition list. If the plot shows mostly clear separate colour regions, the conditions are well-separated in semantic space and your retrieval pipeline will rarely confuse codes from different conditions. If colours are deeply interleaved in a region, those conditions share clinical territory — and codes from that region are prime candidates for multimorbidity bridging.

**Reading the hover tooltip:** Hovering over any point shows the full term name, the code ID (so you can look it up in SNOMED), the condition it came from, and the NHS observation count if available. This means the plot is also an interactive reference tool.

### 6.2 — Figure 2: t-SNE by Cluster

```python
# cluster_analysis.py — _fig_tsne_by_cluster()

df["cluster_label"] = df["cluster"].map(cluster_names)
# Before: df["cluster"] = [3, 3, 7, 0, ...]
# After:  df["cluster_label"] = ["C3: Hypertension · Blood", "C3: ...", "C7: ...", "C0: ...", ...]

fig = px.scatter(
    df,
    x="tsne_x", y="tsne_y",
    color="cluster_label",
    color_discrete_sequence=px.colors.qualitative.Bold,  # 11 distinct colours
    opacity=0.65,
)
```

**What this plot tells you:** This is the same layout as Figure 1, but coloured by KMeans assignment instead of condition. This lets you verify that the KMeans algorithm found structure that makes clinical sense. If a cluster contains codes from only one condition and forms a clean region in the plot, the cluster is clinically coherent. If a cluster mixes codes from two or three conditions but forms a tight region, you are looking at a multimorbidity bridging cluster.

**The key diagnostic comparison:** Open Figures 1 and 2 side by side. Find a region where two conditions overlap in Figure 1. Look at the same region in Figure 2. If that region is all one cluster colour, KMeans correctly identified it as a shared space. If it has multiple cluster colours, the bridging structure was split across clusters — less ideal for the BT-03 completeness tests.

### 6.3 — Figure 3: Cluster Composition Bar Chart

```python
# cluster_analysis.py — _fig_cluster_composition()

# Group by (cluster_label, condition) and count codes
counts = (
    df.groupby(["cluster_label", "condition"])
    .size()
    .reset_index(name="count")
)

# Example counts DataFrame:
# cluster_label              condition            count
# C0: ASCVD · Cholesterol    ASCVD                312
# C0: ASCVD · Cholesterol    Antihypertensives      8
# C0: ASCVD · Cholesterol    Lipid Lowering         4
# C2: Type 2 Diabetes        Type 2 Diabetes       87
# C2: Type 2 Diabetes        ASCVD                 14
# ...

fig = px.bar(
    counts,
    x="count",          # bar length = number of codes
    y="cluster_label",  # one row per cluster
    color="condition",  # stack bar segments by condition
    orientation="h",    # horizontal bars (easier to read long cluster names)
    color_discrete_map=CONDITION_COLOURS,
)
```

**What this plot tells you:** Each horizontal bar is one cluster. The coloured segments show how many codes from each condition ended up in that cluster. A bar that is almost entirely one colour is a single-condition cluster. A bar with significant segments from multiple conditions is a bridging cluster — this is the visual representation of the "multimorbidity bridging" concept from BT-03.

**A concrete example to understand the structure:**

Suppose you see cluster `C3: Hypertension` has a bar with:
- 90 codes from Hypertension (red)
- 40 codes from ASCVD (blue)
- 15 codes from Antihypertensives (purple)

This tells you that from KMeans' perspective, 40 ASCVD codes are more similar to the hypertension cluster than to any other cluster. Those 40 codes are candidates for inclusion in a comorbidity query about "hypertension with cardiovascular disease" — and they are exactly the bridging codes that the BT-03 framework is designed to help you find.

### 6.4 — Figure 4: The Completeness Heatmap

This is the most analytically important figure. It implements the BT-03 cluster fingerprinting concept directly.

```python
# cluster_analysis.py — _fig_completeness_heatmap()

# The matrix: rows = clusters, columns = conditions
# Cell value = what proportion of that condition's codes fall in that cluster
# e.g. cell (C2: Type 2 Diabetes, Type 2 Diabetes) = 0.45
# means 45% of all Type 2 Diabetes codes belong to cluster C2

matrix = np.zeros((len(cluster_labels), len(conditions)))
for ci, cond in enumerate(conditions):
    cond_total = (df["condition"] == cond).sum()   # total codes for this condition
    for ri, clabel in enumerate(cluster_labels):
        n = ((df["condition"] == cond) & (df["cluster_label"] == clabel)).sum()
        matrix[ri, ci] = n / cond_total            # proportion, not count

# We normalise to proportions (not raw counts) because the conditions
# have very different numbers of codes: ASCVD has 2179, BMI has 9.
# Comparing raw counts would make ASCVD dominate every cluster.
# Proportions put every condition on equal footing.
```

The heatmap is then rendered as a colour-scaled grid:

```python
fig = go.Figure(data=go.Heatmap(
    z=matrix,                # the proportion values (0.0 to 1.0)
    x=conditions,            # column labels
    y=cluster_labels,        # row labels
    text=text_matrix,        # percentage strings for annotation
    colorscale=[
        [0.0, "#F0F4F5"],    # near-zero → pale NHS background
        [0.2, "#41B6E6"],    # light → NHS light blue
        [0.5, "#005EB8"],    # medium → NHS blue
        [1.0, "#003087"],    # high → NHS dark blue
    ],
))
```

**How to read the heatmap:**

Each row is one KMeans cluster. Each column is one NICE condition. Imagine you are reading across a single row. If you see a bright cell under one condition and pale cells under all others, that cluster "belongs" to one condition. If you see bright cells under two or three conditions in the same row, that cluster is shared — it is a bridging cluster.

Now imagine reading down a single column. Each bright cell tells you which cluster that condition's codes concentrate in. A condition with codes spread across many clusters has a diverse semantic footprint. A condition with codes concentrated in one cluster has a narrow footprint.

**The BT-03 diagnostic:** When you query for "obesity with hypertension", the ideal code list should draw from the obesity cluster(s) and the hypertension cluster(s), but also from any bridging cluster that appears bright in both the obesity column and the hypertension column. If your retrieval system only returns codes from the obesity cluster and hypertension cluster and misses the bridging cluster, the heatmap tells you exactly which cluster to interrogate.

### 6.5 — Figure 5: Observations Bubble Chart

```python
# cluster_analysis.py — _fig_observations_scatter()

# Filter to only codes that have a recorded observation count
obs_df = df[df["observations"].notna() & (df["observations"] > 0)].copy()

# Log-scale the observation count to control for extreme outliers
# (one code might have 168 million observations; another might have 50)
# Without log scaling, the most common code would dominate the entire chart
obs_df["log_obs"] = np.log10(obs_df["observations"] + 1)
# log10(168,110,661) ≈ 8.2
# log10(50)          ≈ 1.7
# This gives a reasonable range for bubble sizes

fig = px.scatter(
    obs_df,
    x="tsne_x", y="tsne_y",
    size="log_obs",       # bubble size = log10(observations)
    size_max=22,          # cap the maximum bubble size in pixels
    color="condition",
    color_discrete_map=CONDITION_COLOURS,
    opacity=0.75,
)
```

**What this plot tells you:** This is the same t-SNE layout as Figures 1 and 2, but now the dot size encodes clinical importance. Large bubbles are codes that appear frequently in NHS records — these are the codes that will determine whether a cohort analysis includes a meaningful number of patients. Small bubbles are rare codes that might be clinically important for specific patients but represent a small volume.

The combination of position (where in the clinical landscape) and size (how important in the NHS) gives you a prioritised view of the code space. If a bridging cluster region contains large bubbles, the bridging codes are high-volume and important to get right. If it contains only small bubbles, the bridging codes matter for individual patients but may not affect cohort counts significantly.

---

## Part 7 — The Summary Statistics Builder

```python
# cluster_analysis.py — _build_summary_md()

# Input:  df (enriched), cluster_names dict
# Output: a Markdown string with tables and bullet points

def _build_summary_md(df: pd.DataFrame, cluster_names: dict[int, str]) -> str:
    lines = [
        f"**{len(df):,} codes** across **{df['condition'].nunique()} conditions** ...",
        "",
        "**Codes per condition:**",
        "| Condition | Codes | With Observations |",
        ...
    ]

    # Bridge cluster detection — a cluster is a bridge if it contains
    # codes from 3 or more different conditions
    for cid in range(K_CLUSTERS):
        mask   = df["cluster"] == cid
        n_cds  = df[mask]["condition"].nunique()
        if n_cds >= 3:
            # This cluster spans at least 3 conditions — it is a bridge
            bridge_lines.append(f"- **{cluster_names[cid]}** — {n_cds} conditions ...")
```

The threshold of 3 conditions for bridge detection is a design choice. A cluster shared between 2 conditions could just be coincidental overlap. A cluster shared between 3 or more conditions is almost certainly a genuine bridging cluster representing a shared clinical concept.

The summary Markdown is displayed in a `gr.Markdown` component above the plots in the UI. It gives the analyst the key numbers without having to read the visualisations.

---

## Part 8 — The BT-03 Backtesting Utilities

Section 8 of `cluster_analysis.py` contains three standalone functions that implement the formal completeness testing described in your BT-03 document. They are not called by the UI directly — they are utilities that can be called from the command line or from a separate backtesting script.

### 8.1 — `compute_cluster_distribution()`

```python
# Input:
code_set = {"44054006", "73211009", "197761014"}  # a set of SNOMED code strings
df       = results["df"]                            # enriched DataFrame

# What it does:
# Filters the DataFrame to only the rows whose code appears in code_set
# Counts how many of those codes fall in each cluster
# Returns a dict: {cluster_id: count}

result = compute_cluster_distribution(code_set, df)
# Example output:
# {2: 2, 5: 1}
# means: 2 of these codes are in cluster 2, 1 is in cluster 5
```

This is the "cluster fingerprint" of a code list. The gold-standard NICE code list for Type 2 Diabetes has a particular fingerprint (most codes in clusters 2 and 5, perhaps). If your agent's code list for the same condition has a very different fingerprint (mostly cluster 0 and 3), that difference tells you the agent is finding codes from the wrong part of semantic space.

### 8.2 — `backtest_cluster_completeness()`

```python
# Input:
nice_lists  = {
    "Type 2 Diabetes": {"44054006", "197761014", "280571000006116", ...},  # 221 codes
    "Hypertension":    {"99042012", "64168014", "790091000006115", ...},   # 143 codes
}
agent_lists = {
    "Type 2 Diabetes": {"44054006", "197761014"},   # agent found only 2 codes (bad recall)
    "Hypertension":    {"99042012", "64168014", "73211009", ...},  # agent found 140
}
df             = results["df"]
cluster_names  = results["cluster_names"]

# Output: a DataFrame with one row per (condition, cluster) pair
result = backtest_cluster_completeness(nice_lists, agent_lists, df, cluster_names)

#   condition         cluster_id  cluster_label           nice_proportion  agent_proportion  discrepancy  gap_direction
#   Type 2 Diabetes   2           C2: T2DM · Glucose      0.62             0.50              -0.12        UNDER_REPRESENTED
#   Type 2 Diabetes   5           C5: T2DM · Insulin      0.28             0.50               0.22        OVER_REPRESENTED
#   Hypertension      3           C3: HTN · Blood         0.71             0.73               0.02        BALANCED
```

The `discrepancy` column (agent_proportion minus nice_proportion) tells you where the agent is falling short. A discrepancy of -0.12 means the agent is 12 percentage points under-representing that cluster relative to the gold standard. The `gap_direction` flag (UNDER_REPRESENTED/OVER_REPRESENTED/BALANCED) uses a ±0.05 threshold — anything within 5 percentage points is considered balanced.

This output directly answers the BT-03 question: "which clinical clusters is the agent failing to find?" Armed with this information, you can investigate whether the retrieval pipeline needs different sub-queries for the under-represented clusters, or whether the embedding model is poorly calibrated for those clinical concepts.

### 8.3 — `find_bridge_clusters()`

```python
# Input:
condition_a_codes = nice_lists["Type 2 Diabetes"]   # 221 codes
condition_b_codes = nice_lists["Hypertension"]       # 143 codes
df                = results["df"]

# Step 1: find which clusters contain ≥2 codes from condition A
dist_a = compute_cluster_distribution(condition_a_codes, df)
clusters_a = {c for c, n in dist_a.items() if n >= 2}
# e.g. clusters_a = {2, 5, 3}

# Step 2: same for condition B
clusters_b = {c for c, n in dist_b.items() if n >= 2}
# e.g. clusters_b = {3, 0, 7}

# Step 3: intersection = clusters that are meaningfully in BOTH
bridges = clusters_a & clusters_b
# e.g. bridges = {3}  ← cluster 3 has ≥2 codes from both T2DM and Hypertension

# Step 4: find the specific codes from condition B's list that are in the bridge clusters
bridge_codes = set(
    df[df["code"].isin(condition_b_codes) & df["cluster"].isin(bridges)]["code"]
)

# Output:
{
    "bridge_cluster_ids": [3],
    "bridge_codes_count": 12,
    "bridge_codes":       {"790091000006115", "99042012", ...}
}
```

These bridge codes are the ones the agent must find to produce a complete comorbidity code list. If the agent misses them, the resulting cohort will systematically exclude patients who are coded with those intersection-specific codes rather than the more common single-condition codes.

---

## Part 9 — How `app.py` Consumes the Analysis

### 9.1 — The `run_data_analysis()` wrapper

`app.py` does not call any `cluster_analysis` functions directly except one: `get_analysis_results()`. Everything else is encapsulated in a thin wrapper function:

```python
# app.py — Section B3

def run_data_analysis(force: bool = False):
    """
    Called when the user clicks "🔬 Run Analysis" or "♻️ Clear Cache & Recompute".

    Returns 7 values that Gradio maps positionally to 7 output components.
    The 'force' parameter is False for "Run Analysis" and True for "Recompute".
    """
    try:
        results = cluster_analysis.get_analysis_results(force_recompute=force)

        status = "✅ **Analysis complete.** ..."

        return (
            status,                            # → analysis_status (gr.Markdown)
            results["fig_tsne_condition"],     # → fig_tsne_cond   (gr.Plot)
            results["fig_tsne_cluster"],       # → fig_tsne_clust  (gr.Plot)
            results["fig_cluster_bar"],        # → fig_cluster_bar (gr.Plot)
            results["fig_heatmap"],            # → fig_heatmap     (gr.Plot)
            results["fig_obs_scatter"],        # → fig_obs         (gr.Plot)
            results["summary_md"],             # → analysis_summary (gr.Markdown)
        )
    except Exception as e:
        empty = _empty_figure(str(e))
        err   = f"⚠️ **Analysis failed:** {e}\n\n..."
        return err, empty, empty, empty, empty, empty, ""
```

**Why is the return a tuple of 7 values?** Gradio's event system maps return values to output components positionally. When `btn_run_analysis.click()` fires and `run_data_analysis()` returns, Gradio takes value 1 (the status string) and sends it to `analysis_status`, takes value 2 (the first figure) and sends it to `fig_tsne_cond`, and so on in order. If the tuple had 6 values but `_outputs` listed 7 components, Gradio would raise a `ValueError` at runtime. **Every early-return path** (the exception handler) must also return exactly 7 values — which is why the except block returns 7 items with placeholder empties.

### 9.2 — The `_empty_figure()` placeholder

```python
# app.py — _empty_figure()

def _empty_figure(msg: str = "Run analysis to see results"):
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",  # coordinates are fractions of the plot area, not data values
        x=0.5, y=0.5,                # centre of the plot
        showarrow=False,
        font=dict(size=14, color="#768692", family="Arial"),
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#F0F4F5",
        xaxis=dict(visible=False),   # hide axes — there is no data to label
        yaxis=dict(visible=False),
        height=400,
    )
    return fig
```

This produces a blank placeholder figure with a centred message. It is returned when the analysis has not yet been run (so the plot components show something sensible rather than being empty), and also when the analysis fails (so the error message appears where the figure would have been rather than in a separate error panel).

### 9.3 — Event wiring in `_build_interface()`

```python
# app.py — _build_interface()

# These are the 7 Gradio components the analysis tab updates
_analysis_outputs = [
    analysis_status,    # gr.Markdown — status message
    fig_tsne_cond,      # gr.Plot — figure 1
    fig_tsne_clust,     # gr.Plot — figure 2
    fig_cluster_bar,    # gr.Plot — figure 3
    fig_heatmap,        # gr.Plot — figure 4
    fig_obs,            # gr.Plot — figure 5
    analysis_summary,   # gr.Markdown — summary stats
]

# "Run Analysis" button — use cached results if available
btn_run_analysis.click(
    fn=lambda: run_data_analysis(force=False),
    inputs=[],          # no inputs needed — the function reads from disk
    outputs=_analysis_outputs,
).then(
    fn=lambda: gr.update(visible=True),   # make the summary panel visible
    outputs=[analysis_summary],
)

# "Clear Cache & Recompute" button — force fresh computation
btn_recompute.click(
    fn=lambda: run_data_analysis(force=True),
    inputs=[],
    outputs=_analysis_outputs,
).then(
    fn=lambda: gr.update(visible=True),
    outputs=[analysis_summary],
)
```

The `.then()` call chained after each `.click()` is a second event that fires after the first completes. It makes the summary Markdown panel visible — it starts as `visible=False` so it does not take up space before the analysis has run. After `run_data_analysis()` returns, Gradio updates its 7 outputs, then fires the `.then()` which makes the summary visible.

**Why `inputs=[]`?** The `run_data_analysis()` function takes no Gradio state as input. It gets everything it needs from `cluster_analysis.get_analysis_results()` which reads from disk (or memory cache). There is nothing the user can configure that affects what gets computed. This is intentional — the analysis is deterministic given the CSV file. If you wanted to add a `k` slider to let the user choose the number of clusters, you would add it as a Gradio input, pass it to `run_data_analysis(k=k)`, and pass it through to `_run_kmeans(embeddings, k=k)`.

### 9.4 — Why the analysis tab is a separate `gr.Tab` rather than a button in the existing UI

The original UI is a single page. Adding the analysis visualisations to that page would make it extremely long and slow to load (five Plotly figures take a moment to render). Using `gr.Tabs` means:

1. The analysis tab's components do not render until the user navigates to that tab (Gradio's lazy rendering).
2. The clinical search workflow is completely unaffected — the chatbot, feedback buttons, and accordions work exactly as before.
3. The URL stays at `http://localhost:7860` — this is a single-page application, not multiple pages.

The tab structure:

```python
with gr.Tabs(elem_classes=["tab-nav"]):

    with gr.Tab("🔍 Clinical Search"):
        # All existing app.py content — chatbot, search, accordions
        # UNCHANGED from previous versions

    with gr.Tab("🔬 Data Analysis"):
        # New content — buttons, status, 5 plots, summary
        # Completely self-contained
```

The two tabs share the NHS header and footer (rendered outside the `gr.Tabs` block) but have completely independent content and event handlers.

---

## Part 10 — The Complete Data Flow

This is the end-to-end sequence of operations from "user clicks Run Analysis" to "plots appear on screen."

```
USER clicks "🔬 Run Analysis"
        │
        ▼
app.py: btn_run_analysis.click fires
        │
        ▼
app.py: run_data_analysis(force=False)
        │
        ▼
cluster_analysis.get_analysis_results(force_recompute=False)
        │
        ├── _RESULTS_CACHE is not None?
        │     YES → return cached dict immediately (< 1ms)
        │
        ├── _RESULTS_CACHE is None
        │     │
        │     ▼
        │   load_data()
        │     reads combined_normalized_codes.csv
        │     applies _clean_source_name() to Source column
        │     returns df: (5781, 5) [code, term, condition, observations, source_raw]
        │
        ├── _load_or_compute_embeddings(df)
        │     outputs/analysis_cache/embeddings.npy exists?
        │       YES → np.load(cache) → (5781, 384) float32
        │       NO  → _compute_embeddings(df["term"].tolist())
        │               SentenceTransformer("BAAI/bge-small-en").encode(terms)
        │               → (5781, 384) float32
        │               np.save(cache)
        │
        ├── _load_or_compute_kmeans(embeddings)
        │     outputs/analysis_cache/kmeans_labels.npy exists?
        │       YES → np.load(cache) → (5781,) int32
        │       NO  → _run_kmeans(embeddings, k=8)
        │               KMeans(n_clusters=8, n_init=20).fit_predict(embeddings)
        │               → (5781,) int32    [values 0..7]
        │               np.save(cache)
        │     df["cluster"] = labels
        │
        ├── _load_or_compute_tsne(embeddings)
        │     outputs/analysis_cache/tsne_coords.npy exists?
        │       YES → np.load(cache) → (5781, 2) float32
        │       NO  → _run_tsne(embeddings)
        │               TSNE(n_components=2, perplexity=30, init="pca").fit_transform(embeddings)
        │               → (5781, 2) float32
        │               np.save(cache)
        │     df["tsne_x"] = coords[:, 0]
        │     df["tsne_y"] = coords[:, 1]
        │
        ├── _auto_name_clusters(df)
        │     for each cluster 0..7:
        │       find most common condition
        │       find most frequent non-stopword in terms
        │       build label: "C{id}: {condition} · {word}"
        │     returns cluster_names dict
        │
        ├── Build 5 figures:
        │     _fig_tsne_by_condition(df)           → fig_tsne_condition
        │     _fig_tsne_by_cluster(df, names)      → fig_tsne_cluster
        │     _fig_cluster_composition(df, names)  → fig_cluster_bar
        │     _fig_completeness_heatmap(df, names) → fig_heatmap
        │     _fig_observations_scatter(df)        → fig_obs_scatter
        │
        ├── _build_summary_md(df, names)           → summary_md string
        │
        └── Store all in _RESULTS_CACHE (module-level dict)
            return results dict
        │
        ▼
app.py: run_data_analysis() unpacks results dict into 7-tuple
        (status_md, fig1, fig2, fig3, fig4, fig5, summary_md)
        │
        ▼
Gradio maps 7-tuple to 7 output components positionally:
        analysis_status  ← status_md
        fig_tsne_cond    ← fig1
        fig_tsne_clust   ← fig2
        fig_cluster_bar  ← fig3
        fig_heatmap      ← fig4
        fig_obs          ← fig5
        analysis_summary ← summary_md
        │
        ▼
.then() fires: gr.update(visible=True) → analysis_summary becomes visible
        │
        ▼
BROWSER renders 5 interactive Plotly plots
USER can hover, pan, zoom, and click any point
```

---

## Part 11 — The Module-Level Cache and Why It Matters

There is a subtlety in how the in-memory cache works that is worth understanding clearly.

```python
# cluster_analysis.py — top of Section 7

_RESULTS_CACHE: dict | None = None  # module-level variable

def get_analysis_results(force_recompute: bool = False) -> dict:
    global _RESULTS_CACHE              # declare we are modifying the module-level variable
    if _RESULTS_CACHE is not None and not force_recompute:
        return _RESULTS_CACHE          # return immediately — all computation skipped
    ...
    _RESULTS_CACHE = results           # store so next call returns immediately
    return results
```

Python modules are singletons — when `app.py` does `import cluster_analysis`, Python loads the module once and caches it. Every subsequent import (even in other files) gets the same cached module object. This means `_RESULTS_CACHE` is shared across the entire application process.

Consequence: the first time the user clicks "Run Analysis", the full pipeline runs (1–120 seconds depending on whether the disk cache exists). If the user navigates away and comes back, or clicks the button again, `_RESULTS_CACHE` is not None and the function returns in microseconds.

The `force=True` path (from the "Clear Cache & Recompute" button) sets `_RESULTS_CACHE = None` and deletes the `.npy` disk files before rerunning. This is the only way to see new results if the CSV file has been updated.

---

## Part 12 — Dependencies Reference

| Package | Where used | Why needed |
|---|---|---|
| `numpy` | All sections | Array storage, `.npy` cache files, matrix operations |
| `pandas` | Data loading, figure building | DataFrame operations, groupby, pivot |
| `sentence_transformers` | `_compute_embeddings()` | BAAI/bge-small-en embedding model |
| `sklearn.cluster.KMeans` | `_run_kmeans()` | KMeans clustering algorithm |
| `sklearn.manifold.TSNE` | `_run_tsne()` | t-SNE dimensionality reduction |
| `plotly.express` | Scatter and bar figures | Interactive scatter and bar charts |
| `plotly.graph_objects` | Heatmap figure | Low-level Plotly API for custom heatmap |
| `gradio` | `app.py` only | `gr.Plot`, `gr.Tabs`, `gr.Tab`, `gr.Markdown` |

Install any missing packages with:
```bash
pip install plotly scikit-learn sentence-transformers
# numpy, pandas, and gradio should already be installed
```

---

## Part 13 — Quick Reference: All Public Functions

### `cluster_analysis.py`

| Function | Inputs | Output | Called by |
|---|---|---|---|
| `load_data()` | none | `pd.DataFrame` (5781 × 5) | `get_analysis_results()` |
| `get_analysis_results(force_recompute)` | `bool` | `dict` of 8 items | `app.py run_data_analysis()` |
| `clear_cache()` | none | none | CLI or `get_analysis_results(force=True)` |
| `compute_cluster_distribution(code_set, df)` | `set[str]`, `pd.DataFrame` | `dict[int, int]` | `backtest_cluster_completeness()`, `find_bridge_clusters()` |
| `backtest_cluster_completeness(nice, agent, df, names)` | 4 args | `pd.DataFrame` | backtesting scripts |
| `find_bridge_clusters(codes_a, codes_b, df)` | 3 args | `dict` | backtesting scripts |

### `app.py` (analysis-related)

| Function | Inputs | Output | Called by |
|---|---|---|---|
| `run_data_analysis(force)` | `bool` | `tuple` of 7 values | Gradio button events |
| `_empty_figure(msg)` | `str` | `go.Figure` | `run_data_analysis()` on error/startup |

---

## Part 14 — Further Experiments You Can Run

Now that the infrastructure is in place, here are natural extensions to try:

**Experiment 1: Change the number of clusters.**
Edit `K_CLUSTERS = 8` in `cluster_analysis.py` to `K_CLUSTERS = 12` and click "Clear Cache & Recompute". Smaller k merges clinical groups; larger k splits them. Try to find the value where every cluster has a meaningful, distinct clinical interpretation.

**Experiment 2: Run the backtesting utilities against your agent.**
Once your retrieval system produces code lists, call `backtest_cluster_completeness()` from a notebook:
```python
import cluster_analysis as ca
results = ca.get_analysis_results()
df = results["df"]
cluster_names = results["cluster_names"]

nice_lists  = {"Type 2 Diabetes": set(df[df["condition"]=="Type 2 Diabetes"]["code"])}
agent_lists = {"Type 2 Diabetes": your_agent.retrieve("type 2 diabetes")}

report = ca.backtest_cluster_completeness(nice_lists, agent_lists, df, cluster_names)
print(report[report["gap_direction"] == "UNDER_REPRESENTED"])
```

**Experiment 3: Try UMAP instead of t-SNE.**
UMAP (Uniform Manifold Approximation and Projection) is faster than t-SNE and preserves global structure better. Install with `pip install umap-learn`, then replace the `TSNE` call in `_run_tsne()` with:
```python
import umap
reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
coords  = reducer.fit_transform(embeddings)
```
Delete the t-SNE cache file and click "Clear Cache & Recompute" to see the UMAP layout instead.

**Experiment 4: Inspect individual clusters.**
Using the enriched DataFrame from `get_analysis_results()`, you can print every term in any cluster:
```python
df = ca.get_analysis_results()["df"]
cluster_names = ca.get_analysis_results()["cluster_names"]

# Find which cluster ID is the "bridging" cluster
bridge_id = 3  # replace with actual ID from the heatmap
print(df[df["cluster"] == bridge_id][["code", "term", "condition"]].to_string())
```
This lets you read every code in a cluster and assess whether KMeans found a clinically coherent grouping.
