# 10 — Annotated Code Skeletons: Every Technique, Explained Line by Line

> **How to use this document.** This is the practical companion to the conceptual documents. Every code block here is designed to be copy-paste ready as a starting skeleton, with annotations explaining not just *what* the code does but *why* each design decision was made and *how* it connects to the broader project. Work through these in order — each block builds on the last.

---

## 1. EDA and Correlation Heatmaps

The first thing you do with any new dataset is look at it — not to build models, but to build intuition. The code below goes beyond a basic heatmap by showing you how to prepare data in the way this project actually needs, including the log-transformation of usage counts that is essential before any correlation analysis on clinical code frequency data.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Load and prepare the usage frequency data.
# OpenCodeCounts typically arrives as a CSV with one row per SNOMED code
# and columns for the code ID, description, and annual usage count.
# -----------------------------------------------------------------------
df_codes = pd.read_csv("data/processed/snomed_with_usage.csv")

# Usage frequency follows a power law distribution — a handful of codes
# are used millions of times, while most are used rarely or never.
# This creates a severe right skew that makes raw counts nearly useless
# for correlation analysis. The log transform compresses the range and
# makes the distribution approximately normal, which is what Pearson
# correlation requires. Always log-transform before correlating.
df_codes['log_usage'] = np.log1p(df_codes['usage_count'])
# np.log1p() computes log(1 + x), which handles zero safely —
# log(0) is undefined, so codes with zero usage would cause errors otherwise.

# -----------------------------------------------------------------------
# Step 1: Understand the distribution before doing anything else.
# The describe() output tells you the median vs. mean usage frequency —
# if the mean is much larger than the median (as it will be here),
# that confirms the power-law distribution and validates the log transform.
# -----------------------------------------------------------------------
print("=== Raw Usage Count Distribution ===")
print(df_codes['usage_count'].describe())

print("\n=== Log-Transformed Usage Distribution ===")
print(df_codes['log_usage'].describe())

# -----------------------------------------------------------------------
# Step 2: Flag which codes belong to which conditions.
# In this project, you will have loaded the NICE example code lists
# and can create binary indicator columns for condition membership.
# These binary flags are what you correlate against each other —
# "when obesity codes are used, which other condition codes tend to
# appear in the same patient population?"
# -----------------------------------------------------------------------
# Create binary presence flags per condition (1 = code is in that list)
condition_list_df = pd.read_csv("data/processed/nice_example_lists.csv")

for condition in condition_list_df['condition_category'].unique():
    codes_for_condition = set(
        condition_list_df.loc[
            condition_list_df['condition_category'] == condition, 'snomed_code'
        ].astype(str)
    )
    col_name = f"in_{condition.lower().replace(' ', '_')}_list"
    df_codes[col_name] = df_codes['snomed_code'].astype(str).isin(codes_for_condition).astype(int)

# -----------------------------------------------------------------------
# Step 3: Build the correlation matrix for the condition flags.
# This answers: "Which condition code lists are most similar to each other
# in terms of which codes they contain?"
# High positive correlation between two conditions means they share many
# codes — this is the signature of clinical comorbidity.
# -----------------------------------------------------------------------
condition_columns = [c for c in df_codes.columns if c.startswith('in_') and c.endswith('_list')]

correlation_matrix = df_codes[condition_columns].corr()

plt.figure(figsize=(12, 10))
# fmt='.2f' shows two decimal places; vmin/vmax set the colour scale range
# so that zero correlation is always the midpoint (white in coolwarm).
heatmap = sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt='.2f',
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={'label': 'Pearson Correlation'}
)
plt.title("Correlation of NICE Code List Overlap by Condition", fontsize=14, pad=20)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("outputs/eda_condition_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
```

---

## 2. Feature Engineering with Graph and Semantic Features

This block brings together the graph features from `07_snomed_graph_architecture.md` and the embedding features from `08_embeddings_and_semantic_search.md` into a single unified feature matrix. This is the DataFrame that will feed every model in the project.

```python
import networkx as nx
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------
# Part A — Graph Feature Engineering
# Build the SNOMED polyhierarchy as a DiGraph and compute structural
# features for every code. These features capture information about
# where a code sits in the clinical knowledge hierarchy.
# -----------------------------------------------------------------------

def compute_graph_features(G: nx.DiGraph, code_list: list[str]) -> pd.DataFrame:
    """
    Compute graph-theoretic features for each SNOMED code.
    
    We compute four features that each capture a different structural
    property:
    - in_degree: number of parent concepts (>1 = polyhierarchical bridging concept)
    - out_degree: number of child concepts (0 = leaf/most specific)
    - subtree_size: total descendants (larger = broader category)
    - degree_centrality: normalised connectivity (higher = more "hub-like")
    """
    graph_features = []
    
    for code in code_list:
        if code not in G.nodes:
            # Some codes may not appear in the graph (e.g., new codes added
            # after our SNOMED version was downloaded). Assign neutral values.
            graph_features.append({
                'snomed_code': code,
                'graph_in_degree': 0,
                'graph_out_degree': 0,
                'graph_subtree_size': 0,
                'graph_degree_centrality': 0.0
            })
            continue
        
        graph_features.append({
            'snomed_code': code,
            # Predecessors in our parent→child directed graph = parent concepts
            'graph_in_degree':         G.in_degree(code),
            # Successors = child concepts (more specific subtypes)
            'graph_out_degree':        G.out_degree(code),
            # All reachable descendants = total subtree size
            'graph_subtree_size':      len(nx.descendants(G, code)),
            # Centraliy: how connected is this code relative to all others?
            'graph_degree_centrality': nx.degree_centrality(G).get(code, 0.0)
        })
    
    return pd.DataFrame(graph_features)


# -----------------------------------------------------------------------
# Part B — Semantic Feature Engineering
# Generate embeddings and compute cosine similarity of each code
# to its condition category. This measures "how semantically central
# is this code to the concept it's supposed to represent?"
# -----------------------------------------------------------------------

def compute_semantic_features(
        df: pd.DataFrame,
        model: SentenceTransformer,
        condition_query_map: dict[str, str]) -> pd.DataFrame:
    """
    For each condition, compute the cosine similarity between every
    code description and the condition's canonical query string.
    
    condition_query_map example:
    {
        'obesity': 'obesity overweight high body mass index BMI',
        'hypertension': 'hypertension high blood pressure elevated BP'
    }
    
    Codes with high similarity to their target condition are more
    semantically central to that concept — a stronger positive signal
    than codes that match only loosely.
    """
    # Encode all code descriptions in one batch — much faster than one by one
    code_texts = df['description'].tolist()
    code_embeddings = model.encode(code_texts, normalize_embeddings=True, 
                                   show_progress_bar=True, batch_size=256)
    
    semantic_features = {'snomed_code': df['snomed_code'].tolist()}
    
    for condition_name, query_text in condition_query_map.items():
        # Encode the condition query
        query_embedding = model.encode([query_text], normalize_embeddings=True)
        
        # Cosine similarity = dot product when vectors are unit-normalised
        # Result is a 1D array of similarity scores, one per code
        similarities = np.dot(code_embeddings, query_embedding.T).flatten()
        
        col_name = f"semantic_sim_{condition_name.replace(' ', '_')}"
        semantic_features[col_name] = similarities.tolist()
    
    return pd.DataFrame(semantic_features)


# -----------------------------------------------------------------------
# Part C — Assemble the Full Feature Matrix
# Merge all feature sources into one DataFrame ready for modelling.
# Scale numerical features for models that are sensitive to scale
# (logistic regression), but keep unscaled versions for tree models
# (Random Forest, Gradient Boosting), which don't need scaling.
# -----------------------------------------------------------------------

def build_feature_matrix(
        codes_df: pd.DataFrame,
        graph_features_df: pd.DataFrame,
        semantic_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble and clean the unified feature matrix.
    This DataFrame is the single input to all modelling steps.
    """
    # Merge all feature sources on the common key
    feature_matrix = (
        codes_df[['snomed_code', 'description', 'log_usage', 'in_qof',
                  'source_count', 'cluster_id', 'deprecated_flag']]
        .merge(graph_features_df, on='snomed_code', how='left')
        .merge(semantic_features_df, on='snomed_code', how='left')
    )
    
    # Fill any remaining NaN values with sensible defaults
    feature_matrix.fillna({
        'log_usage': 0.0,
        'in_qof': False,
        'source_count': 0,
        'cluster_id': -1,
        'deprecated_flag': False,
        'graph_in_degree': 0,
        'graph_out_degree': 0,
        'graph_subtree_size': 0,
        'graph_degree_centrality': 0.0,
    }, inplace=True)
    
    # Encode boolean columns as integers for model compatibility
    feature_matrix['in_qof'] = feature_matrix['in_qof'].astype(int)
    feature_matrix['deprecated_flag'] = feature_matrix['deprecated_flag'].astype(int)
    
    return feature_matrix
```

---

## 3. Correlation and Linear Regression

Linear regression here serves two purposes: understanding the relationship between code features and usage (which builds intuition before supervised classification), and establishing an interpretable baseline that can be presented to NICE stakeholders as evidence of the features' validity.

```python
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------
# Linear regression: which code features predict log usage frequency?
# We use statsmodels because it gives us full statistical output —
# p-values, confidence intervals, R-squared — which are essential for
# understanding the quality of each feature before we use it in a
# more complex model.
# -----------------------------------------------------------------------

feature_cols = ['in_qof', 'source_count', 'graph_in_degree',
                'graph_out_degree', 'graph_degree_centrality']

X = feature_matrix[feature_cols]
y = feature_matrix['log_usage']

# sm.add_constant() adds an intercept column (a column of 1s) to X.
# Without this, the regression is forced through the origin, which is
# almost never the right assumption.
X_with_const = sm.add_constant(X)

ols_model = sm.OLS(y, X_with_const).fit()

# The summary() output gives you everything you need to judge each feature:
# - coef: the estimated effect of a 1-unit increase on log usage
# - t: the t-statistic (larger magnitude = stronger evidence the effect is real)
# - P>|t|: p-value. Conventionally, p < 0.05 means the feature is
#   "statistically significant" — the effect is unlikely to be due to chance.
# - R-squared: what fraction of variance in log usage is explained by the model
print(ols_model.summary())

# What to look for in the output:
# - 'in_qof' should have a large positive coefficient (QOF codes are heavily used)
# - 'source_count' should be positive (codes in more sources tend to be used more)
# - 'graph_in_degree' may be negative (polyhierarchical bridging codes may
#   be more specific and thus used less frequently than general codes)
# - R-squared will likely be moderate — this is expected because usage
#   frequency is determined by many factors beyond code structure.
```

---

## 4. KMeans Clustering and t-SNE / UMAP Visualisation

The most important thing to understand about this section is that KMeans is not just a visualisation tool — the cluster IDs it produces become metadata in the vector store and features in the supervised classifier. The t-SNE and UMAP plots are how you validate that the clusters make clinical sense before you rely on them.

```python
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import plotly.express as px

# -----------------------------------------------------------------------
# STEP 1: Determine the right number of clusters
# The elbow method and silhouette analysis together give more reliable
# guidance than either alone. The elbow is the point where adding more
# clusters stops producing meaningful improvement in inertia (within-cluster
# sum of squares). The silhouette score measures how well-separated the
# clusters are — higher is better, 1.0 is perfect, 0 is overlapping.
# -----------------------------------------------------------------------

inertia_values = []
silhouette_values = []
K_range = range(5, 51, 5)  # Test K = 5, 10, 15, ..., 50

# Load the embedding matrix (built in Phase 2)
embeddings = np.load("data/processed/code_embeddings.npy")

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    # n_init=10 means KMeans restarts with 10 different random initialisations
    # and picks the best result. Higher n_init = more reliable clusters but slower.
    labels = km.fit_predict(embeddings)
    inertia_values.append(km.inertia_)
    
    # silhouette_score requires at least 2 clusters and can be slow on large
    # datasets, so we sample 5000 points for speed
    sample_idx = np.random.choice(len(embeddings), min(5000, len(embeddings)),
                                   replace=False)
    sil = silhouette_score(embeddings[sample_idx], labels[sample_idx])
    silhouette_values.append(sil)
    print(f"K={k:3d} | Inertia: {km.inertia_:,.0f} | Silhouette: {sil:.4f}")

# Plot elbow curve — look for the "kink" where the rate of improvement slows
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(list(K_range), inertia_values, 'b-o')
ax1.set_xlabel("Number of Clusters (K)")
ax1.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
ax1.set_title("Elbow Method — Select K at the 'Kink'")
ax2.plot(list(K_range), silhouette_values, 'r-o')
ax2.set_xlabel("Number of Clusters (K)")
ax2.set_ylabel("Silhouette Score (higher = better)")
ax2.set_title("Silhouette Analysis — Select Peak K")
plt.tight_layout()
plt.savefig("outputs/clustering_k_selection.png", dpi=150)
plt.show()

# -----------------------------------------------------------------------
# STEP 2: Fit the final KMeans model
# Once you have selected K (let's say K=20 based on the elbow analysis),
# fit the final model and store cluster assignments in the feature matrix.
# -----------------------------------------------------------------------

FINAL_K = 20  # Replace with your chosen K from the analysis above

kmeans_final = KMeans(n_clusters=FINAL_K, random_state=42, n_init=10)
cluster_labels = kmeans_final.fit_predict(embeddings)

feature_matrix['cluster_id'] = cluster_labels

# -----------------------------------------------------------------------
# STEP 3: Characterise each cluster
# For each cluster, find the 10 codes whose embeddings are closest to
# the cluster centroid — these are the "most representative" codes and
# help you understand what the cluster represents clinically.
# -----------------------------------------------------------------------

def characterise_clusters(kmeans, embeddings, feature_matrix, top_n=10):
    """
    For each cluster, identify the most central codes and suggest a label.
    Returns a dict mapping cluster_id to a list of representative codes.
    """
    cluster_descriptions = {}
    
    for cluster_id in range(kmeans.n_clusters):
        centroid = kmeans.cluster_centers_[cluster_id]
        cluster_mask = feature_matrix['cluster_id'] == cluster_id
        cluster_indices = np.where(cluster_mask)[0]
        cluster_embeddings = embeddings[cluster_indices]
        
        # Distance of each code's embedding from the cluster centroid
        distances = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        
        # The codes with the SMALLEST distance are the most central
        top_indices = cluster_indices[np.argsort(distances)[:top_n]]
        top_codes = feature_matrix.iloc[top_indices][['snomed_code', 'description']].to_dict('records')
        
        cluster_descriptions[cluster_id] = top_codes
        print(f"\nCluster {cluster_id} (n={cluster_mask.sum()}):")
        for code_info in top_codes[:5]:
            print(f"  {code_info['snomed_code']}: {code_info['description']}")
    
    return cluster_descriptions

cluster_chars = characterise_clusters(kmeans_final, embeddings, feature_matrix)


# -----------------------------------------------------------------------
# STEP 4: UMAP visualisation
# We use UMAP rather than t-SNE for the main visualisation because:
# (a) it is much faster on large datasets
# (b) it better preserves global structure alongside local structure
# We use t-SNE as a complementary view to check local neighbourhoods.
#
# n_neighbors controls how much local vs global structure is preserved —
# lower values emphasise local clusters; higher values emphasise global layout.
# min_dist controls how tightly points cluster together in the projection.
# -----------------------------------------------------------------------

# Subsample for visualisation — 20,000 points is enough to see structure
sample_size = min(20000, len(embeddings))
sample_idx = np.random.choice(len(embeddings), sample_size, replace=False)
embeddings_sample = embeddings[sample_idx]
sample_df = feature_matrix.iloc[sample_idx].copy()

reducer = umap.UMAP(n_components=2, random_state=42,
                    n_neighbors=15, min_dist=0.1)
embedding_2d = reducer.fit_transform(embeddings_sample)

sample_df['umap_x'] = embedding_2d[:, 0]
sample_df['umap_y'] = embedding_2d[:, 1]
sample_df['cluster_label'] = sample_df['cluster_id'].astype(str)
# Mark codes that appear in any NICE example list
sample_df['in_nice_list'] = (sample_df['source_count'] > 0).astype(int)

# Interactive Plotly scatter — hover reveals the code and description
fig = px.scatter(
    sample_df, x='umap_x', y='umap_y',
    color='cluster_label',
    symbol='in_nice_list',  # NICE codes shown with a different marker
    hover_data=['snomed_code', 'description', 'in_qof'],
    title='UMAP Projection of SNOMED CT Embedding Space',
    labels={'cluster_label': 'Cluster', 'in_nice_list': 'In NICE List'},
    width=1200, height=800
)
fig.write_html("outputs/umap_snomed_clusters.html")
fig.show()
```

---

## 5. Supervised Learning with Random Forest and SHAP

This is where the project's analytical work crystallises into the most tangible output: a trained model that can predict whether a clinical code should be included in a NICE list, with full explanations for each prediction. The SHAP section is especially important — run it, understand it, and show its outputs to the team before building the agent.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import shap

# -----------------------------------------------------------------------
# Build the classification dataset
# Positive class (y=1): codes in the NICE example lists
# Negative class (y=0): codes NOT in the NICE lists but in the same
# semantic neighbourhood (hard negatives — see Phase 4, Feature 4.1)
# -----------------------------------------------------------------------

feature_cols = [
    'log_usage', 'in_qof', 'source_count',
    'graph_in_degree', 'graph_out_degree', 'graph_subtree_size',
    'graph_degree_centrality', 'cluster_id'
] + [c for c in feature_matrix.columns if c.startswith('semantic_sim_')]

# Create binary target label — 1 if code appears in any NICE example list
feature_matrix['in_any_nice_list'] = (
    feature_matrix[[c for c in feature_matrix.columns if c.startswith('in_') and c.endswith('_list')]]
    .any(axis=1).astype(int)
)

X = feature_matrix[feature_cols].fillna(0)
y = feature_matrix['in_any_nice_list']

print(f"Class distribution: {y.value_counts().to_dict()}")
# You will see severe imbalance — far more 0s than 1s. This is normal.

# -----------------------------------------------------------------------
# Handle class imbalance with SMOTE
# SMOTE (Synthetic Minority Over-sampling Technique) creates synthetic
# positive examples by interpolating between existing positive examples.
# This is preferable to simply oversampling (which duplicates examples)
# or undersampling (which discards majority examples).
# -----------------------------------------------------------------------

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print(f"After SMOTE: {pd.Series(y_resampled).value_counts().to_dict()}")


# -----------------------------------------------------------------------
# Stratified K-Fold Cross-Validation
# Stratified ensures each fold has the same class ratio as the full dataset.
# We use 5 folds — a standard choice that balances bias vs. variance.
# We report recall as the primary metric because missing a relevant code
# (false negative) is worse than recommending an extra code (false positive).
# -----------------------------------------------------------------------

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# class_weight='balanced' tells sklearn to weight the loss function
# inversely proportional to class frequencies — another way to handle
# imbalance, often used in combination with SMOTE.
rf_model = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=None,        # Unrestricted trees — reduces bias
    min_samples_leaf=5,    # Prevents overfitting to tiny leaf nodes
    random_state=42,
    n_jobs=-1              # Use all available CPU cores
)

cv_results = cross_validate(
    rf_model, X_resampled, y_resampled,
    cv=skf,
    scoring=['recall', 'precision', 'f1', 'roc_auc'],
    return_train_score=True
)

print("\n=== Cross-Validation Results ===")
for metric in ['recall', 'precision', 'f1', 'roc_auc']:
    test_scores = cv_results[f'test_{metric}']
    print(f"{metric:12}: {test_scores.mean():.4f} ± {test_scores.std():.4f}")

# Fit final model on all data for SHAP analysis
rf_model.fit(X_resampled, y_resampled)


# -----------------------------------------------------------------------
# SHAP Explainability — The Most Important Part
# TreeExplainer is the fast, exact SHAP method for tree-based models.
# It computes the exact Shapley value for each feature and each prediction.
#
# shap_values has shape (n_samples, n_features).
# Positive SHAP values push the prediction TOWARD inclusion (class 1).
# Negative SHAP values push AWAY from inclusion (class 0).
# The magnitude tells you how strongly each feature contributed.
# -----------------------------------------------------------------------

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X)

# The RF model produces SHAP values for both classes [0] and [1].
# We want class 1 (inclusion probability), so index [1].
shap_values_inclusion = shap_values[1] if isinstance(shap_values, list) else shap_values

# Global feature importance — which features matter most overall?
# This is what you show to NICE stakeholders to explain the model.
shap.summary_plot(shap_values_inclusion, X, feature_names=feature_cols,
                  plot_type="bar", show=False)
plt.title("Global Feature Importance (SHAP) — Code Inclusion Classifier")
plt.tight_layout()
plt.savefig("outputs/shap_global_importance.png", dpi=150, bbox_inches='tight')
plt.show()

# Beeswarm plot — shows the distribution of each feature's impact across
# all samples. Red = high feature value; Blue = low feature value.
# A red dot on the right for 'in_qof' confirms QOF pushes strongly toward
# inclusion, which is exactly what we expect and what NICE wants to see.
shap.summary_plot(shap_values_inclusion, X, feature_names=feature_cols, show=False)
plt.tight_layout()
plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches='tight')
plt.show()


# -----------------------------------------------------------------------
# Per-code explanation — translate SHAP values into human-readable rationale
# This function is what the agent's scoring tool calls to generate the
# 'rationale' field in the provenance record.
# -----------------------------------------------------------------------

def generate_shap_rationale(code_idx: int,
                             shap_values: np.ndarray,
                             X: pd.DataFrame,
                             feature_names: list) -> str:
    """
    Convert SHAP values for a single code into a plain-English rationale
    string suitable for inclusion in a NICE provenance record.
    
    The function only mentions the top 3 features by |SHAP value| to
    keep the rationale concise and readable.
    """
    code_shap = shap_values[code_idx]
    
    # Sort features by absolute SHAP contribution (most impactful first)
    feature_importance = sorted(
        zip(feature_names, code_shap, X.iloc[code_idx]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    
    rationale_parts = []
    for feat_name, shap_val, feat_value in feature_importance[:3]:
        direction = "supporting inclusion" if shap_val > 0 else "suggesting caution"
        
        # Generate a human-readable description for each feature
        if feat_name == 'in_qof' and feat_value == 1:
            rationale_parts.append("mandated by QOF Business Rules v49 2024-25")
        elif feat_name == 'log_usage':
            annual_count = int(np.expm1(feat_value))
            rationale_parts.append(f"nationally recorded ~{annual_count:,} times/year")
        elif feat_name == 'graph_in_degree' and feat_value > 1:
            rationale_parts.append(
                f"polyhierarchical concept with {int(feat_value)} parent "
                "classifications (clinically relevant across multiple domains)"
            )
        elif feat_name.startswith('semantic_sim_'):
            condition = feat_name.replace('semantic_sim_', '').replace('_', ' ')
            rationale_parts.append(
                f"high semantic similarity to {condition} "
                f"(score: {feat_value:.2f})"
            )
        elif feat_name == 'deprecated_flag' and feat_value == 1:
            rationale_parts.append("usage trend indicates possible deprecation — verify")
    
    if not rationale_parts:
        return "Included based on composite scoring model; recommend clinical review."
    
    return f"Included based on: {'; '.join(rationale_parts)}."
```

---

## 6. Time Series Forecasting with Prophet

Prophet's strength in this context is its ability to handle missing data, outliers, and the irregular reporting cycles of NHS data releases without complex manual configuration. The annotations below explain the key parameters in terms that are specific to clinical code usage data.

```python
from prophet import Prophet
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Prophet expects a DataFrame with exactly two columns:
# 'ds' — the date column (datetime format)
# 'y'  — the metric to forecast (usage count, after log transform)
# -----------------------------------------------------------------------

def forecast_code_usage(code_usage_history: pd.DataFrame,
                         snomed_code: str,
                         forecast_months: int = 12) -> dict:
    """
    Forecast future usage for a single SNOMED code and detect whether
    its usage is growing, stable, or declining.
    
    Args:
        code_usage_history: DataFrame with columns ['reporting_date', 
                             'snomed_code', 'usage_count']
        snomed_code: The code to analyse
        forecast_months: How many months ahead to forecast
    
    Returns:
        Dict with trend classification and optional deprecation flag
    """
    # Filter to the target code and prepare for Prophet
    code_df = code_usage_history[
        code_usage_history['snomed_code'] == snomed_code
    ].copy()
    
    if len(code_df) < 4:
        # Prophet requires at least a few data points. Codes with fewer
        # than 4 reporting periods cannot be reliably modelled — return
        # a neutral classification rather than a potentially misleading forecast.
        return {"snomed_code": snomed_code, "trend": "insufficient_data",
                "deprecated_flag": False, "forecast": None}
    
    prophet_df = code_df[['reporting_date', 'usage_count']].rename(
        columns={'reporting_date': 'ds', 'usage_count': 'y'}
    )
    # Log-transform for the same reason as in EDA — usage counts are right-skewed
    prophet_df['y'] = np.log1p(prophet_df['y'])
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
    
    # -----------------------------------------------------------------------
    # Configure the Prophet model.
    # yearly_seasonality=True: NHS reporting has annual patterns (some codes
    # peak in winter due to seasonal illness, others are flat year-round).
    # weekly_seasonality=False: we have monthly data, not daily, so weekly
    # seasonality is not meaningful here.
    # changepoint_prior_scale: controls how flexible the trend is.
    # A higher value (0.5) means the model allows sharper trend changes —
    # useful for detecting when a code has been deprecated or superseded.
    # -----------------------------------------------------------------------
    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.5,  # Allow sharp trend changes (deprecation)
        interval_width=0.95           # 95% confidence interval on the forecast
    )
    m.fit(prophet_df)
    
    # Build the forecast dataframe and predict
    future = m.make_future_dataframe(periods=forecast_months, freq='M')
    forecast = m.predict(future)
    
    # -----------------------------------------------------------------------
    # Classify the trend based on the slope of the predicted trend component.
    # Prophet decomposes the forecast into trend + seasonality + noise.
    # We look at the trend component to understand long-run direction.
    # -----------------------------------------------------------------------
    trend_component = forecast['trend']
    trend_slope = (trend_component.iloc[-1] - trend_component.iloc[0]) / len(trend_component)
    
    # Deprecation heuristic: if recent usage is very low AND trend is strongly
    # declining, flag for deprecation review. Thresholds are approximate —
    # tune these based on validation against known-deprecated codes.
    recent_usage = np.expm1(prophet_df['y'].iloc[-3:].mean())
    is_declining_fast = trend_slope < -0.05  # Losing >5% per month on log scale
    deprecated_flag = recent_usage < 100 and is_declining_fast
    
    if trend_slope > 0.02:
        trend_label = "growing"
    elif trend_slope < -0.02:
        trend_label = "declining"
    else:
        trend_label = "stable"
    
    # Plot the forecast for visual inspection
    fig = m.plot(forecast)
    plt.title(f"Usage Forecast for SNOMED {snomed_code} (Trend: {trend_label})")
    plt.xlabel("Date")
    plt.ylabel("Log Usage Count")
    plt.tight_layout()
    
    return {
        "snomed_code": snomed_code,
        "trend": trend_label,
        "trend_slope": round(float(trend_slope), 4),
        "deprecated_flag": deprecated_flag,
        "recent_annual_usage": int(recent_usage),
        "forecast": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict('records')
    }
```

---

## 7. Full EDA and Linear Regression Integration Example

This final skeleton brings together the EDA, correlation, and regression patterns into a single notebook-style workflow that produces the exact outputs needed for the Phase 1 deliverables in the project plan.

```python
import statsmodels.api as sm
import numpy as np
import pandas as pd

# Predict log_usage from code structural features to validate feature quality
X_reg = feature_matrix[['in_qof', 'source_count', 'graph_in_degree',
                          'graph_out_degree', 'graph_degree_centrality']]
y_reg = feature_matrix['log_usage']

X_reg_const = sm.add_constant(X_reg)
ols = sm.OLS(y_reg, X_reg_const).fit()

print(ols.summary())

# Extract the key numbers for the Phase 1 findings summary
print("\n=== Key Findings for Phase 1 Report ===")
for feat in ['in_qof', 'source_count', 'graph_in_degree']:
    coef = ols.params[feat]
    pval = ols.pvalues[feat]
    sig = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "n.s."))
    direction = "increases" if coef > 0 else "decreases"
    print(f"  {feat}: A unit increase {direction} log usage by {abs(coef):.3f} {sig}")

print(f"\n  Model R²: {ols.rsquared:.3f}")
print(f"  Interpretation: The features explain {ols.rsquared*100:.1f}% of variance in national usage frequency.")
```

---

*This completes the annotated code skeleton document. Every snippet here is designed to run against the data structures produced in the earlier project phases. The next step is to follow the project plan in `06_project_plan_and_features.md` and begin Phase 0 data ingestion, using these skeletons as your starting point for each feature.*
