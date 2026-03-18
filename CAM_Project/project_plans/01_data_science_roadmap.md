# 01 — Data Science Roadmap: From EDA to Agentic Workflow

> **Purpose:** This document maps the full analytical journey for the NICE clinical code project — from first look at raw data all the way to a production-ready AI agent. Each phase builds directly on the last, and each technique is explained in terms of *what question it answers* and *why it matters for this specific problem*.

---

## Phase 0 — Environment Setup and Data Ingestion

Before any analysis, the priority is getting all data sources into a consistent, queryable format. Think of this phase as building the library before you start reading the books.

**What to do:**

Start by loading the NICE example code lists (the DAAR_2025_004 `.txt` files) into pandas DataFrames. These files are tab- or pipe-delimited and contain SNOMED codes alongside their textual descriptions. Parse them carefully, as real-world clinical files often have encoding issues (UTF-8 vs Latin-1) and inconsistent delimiters.

Next, download the QOF Business Rules spreadsheet (publicly available from NHS Digital). This is typically an Excel workbook with multiple tabs — one per clinical domain. You will need to flatten this into a single DataFrame with columns for the condition name, indicator code, and SNOMED code list.

Finally, pull the OpenCodeCounts data. This is a CSV-style publication showing each SNOMED code alongside a frequency count of how often it appears in English primary care records. This becomes your "usage weight" feature.

**Key libraries:** `pandas`, `openpyxl`, `pdfplumber`, `requests`, `python-dotenv`

**Output:** A clean, unified data store where each row represents one clinical code and carries: the code itself, its text description, its source(s), its usage frequency, and whether it appears in a NICE gold-standard list.

---

## Phase 1 — Exploratory Data Analysis (EDA)

EDA is where you form hypotheses. You are not yet building models — you are asking the data "what are you?", "where are your gaps?", and "what patterns are already visible to the naked eye?"

### 1.1 Univariate Exploration

Begin by understanding the distribution of usage frequencies. Clinical codes follow a heavily right-skewed, roughly power-law distribution — a small number of codes are used millions of times, while the vast majority are used rarely or not at all. Plot this with a log-scale histogram using `seaborn.histplot` and you'll immediately see this structure.

This matters because: if a code has near-zero usage frequency nationally, there's a high chance it is deprecated, overly specific, or a legacy code being phased out. Usage frequency becomes a key feature later.

Also examine how many unique codes appear per condition in the NICE example lists. This gives you a baseline expectation of list size, which you'll need when evaluating your model's outputs.

### 1.2 Missing Value Analysis

Clinical data has characteristic patterns of missingness. Some codes have descriptions missing from the data dictionary. Some codes appear in QOF but have no usage count because they are so rarely used they fall below a suppression threshold. Create a heatmap with `seaborn.heatmap` on a missingness indicator matrix — this visual pattern often reveals that missingness is systematic rather than random, which affects your modelling strategy.

### 1.3 Correlation Analysis

Once you have your feature matrix (frequency, hierarchy depth, source count, description length, etc.), compute a Pearson correlation matrix with `pandas.DataFrame.corr()` and visualise it as a heatmap. You are looking for:

- Which features co-vary with "inclusion in NICE list" (your eventual target variable)?
- Are any features so highly correlated with each other that you'd be double-counting them as model inputs?

For example, you might find that "number of sources this code appears in" and "usage frequency" are strongly positively correlated — both capture the concept of "how mainstream is this code?", so you might combine them into a single composite feature.

### 1.4 Bivariate and Group-Level Analysis

Break your data down by condition category and source type. Use grouped box plots (seaborn `boxplot` with condition as the hue) to compare the frequency distributions of codes that made it into NICE lists versus those that didn't. This is often visually striking — included codes tend to cluster at the high end of the frequency spectrum, but there are important exceptions (rare conditions with genuinely low-frequency codes).

**Key libraries:** `pandas`, `seaborn`, `matplotlib`, `plotly`

---

## Phase 2 — Feature Engineering

Raw SNOMED codes and text descriptions are not directly usable by most ML models. Feature engineering transforms them into numeric representations that capture clinically relevant structure.

### 2.1 Numerical Features from Structured Sources

From the data you've already loaded, derive: code usage frequency (raw and log-transformed, since the distribution is skewed), number of distinct sources the code appears in (QOF, reference set, NICE list, etc.), a binary flag for whether it appears in QOF (this is your strongest signal for "officially sanctioned"), hierarchy depth in the SNOMED tree (root concepts are broad; leaf concepts are specific — both extremes can be relevant for different reasons), and description length (longer descriptions tend to be more specific, shorter ones more general).

### 2.2 Text Embeddings — The Key Transformation

This is where the project becomes genuinely interesting. The text description of each SNOMED code (e.g. "Type 2 diabetes mellitus with peripheral neuropathy") carries enormous information. We want to turn this text into a dense numeric vector that captures its *meaning*, not just its keywords.

Use `sentence-transformers` with a pre-trained model like `all-MiniLM-L6-v2` as your starting point, then experiment with `pritamdeka/S-PubMedBert-MS-MARCO` which is fine-tuned on biomedical text and will produce better embeddings for clinical descriptions.

The result is a 384- or 768-dimension vector for every code description. Two codes that describe similar clinical concepts will have vectors that are geometrically close to each other in this high-dimensional space, even if they share no keywords. This is the foundation of the entire RAG system.

### 2.3 Hierarchical Features from SNOMED

SNOMED CT has a formal concept hierarchy. A code like "Obesity caused by hypothyroidism" is a child of "Obesity", which is a child of "Disorder of energy balance", and so on. You can encode an approximate hierarchy depth by parsing the NHS SNOMED browser API, and also create a "breadth" feature representing how many sibling concepts exist at the same level (broad concepts with many siblings are general; narrow concepts with few siblings are specific).

**Key libraries:** `sentence-transformers`, `spacy`, `scikit-learn` (for feature scaling and pipeline)

---

## Phase 3 — Dimensionality Reduction and Clustering (Unsupervised Learning)

With 384-dimensional embedding vectors for potentially 100,000+ codes, direct visualisation is impossible. This phase reduces that dimensionality so we can see the structure of the clinical code space.

### 3.1 t-SNE: Visualising the Clinical Code Landscape

t-SNE (t-Distributed Stochastic Neighbour Embedding) is a non-linear dimensionality reduction technique that is excellent at preserving *local* structure — codes that are semantically similar end up near each other in the 2D projection, even if the global layout is distorted.

Apply `sklearn.manifold.TSNE` with `n_components=2` and `perplexity=30` to a sample of your embedded codes. Colour the points by condition category. What you should see is that the SNOMED embedding space naturally organises into clinically coherent clusters — diabetes codes group together, cardiovascular codes form another region, metabolic codes form another. This is powerful validation that our embedding approach is working.

A practical note on t-SNE: it does not scale well to very large datasets because it has O(n²) memory complexity. For datasets larger than ~50,000 points, use UMAP instead (`umap-learn` package), which is faster and better at preserving global structure alongside local structure. Use both and compare — they often reveal different aspects of the data's structure.

### 3.2 KMeans Clustering: Finding Natural Groups

Once you have reduced embeddings, apply KMeans clustering (`sklearn.cluster.KMeans`) to identify natural groupings in the code space. The key challenge is choosing K — how many clusters make sense clinically?

Use the **elbow method** (plot inertia vs K) and the **silhouette score** (`sklearn.metrics.silhouette_score`) to guide your choice. But also bring clinical judgement: if K=20 produces clusters where one cluster is labelled "cardiovascular medication" and another is "cardiovascular procedure", that's probably too fine-grained; if K=5 produces a cluster that mixes obesity codes with antidepressants, that's too coarse.

In the context of this project, KMeans clustering serves two purposes. First, it can reveal whether a research question spans multiple natural clusters (suggesting the analyst needs to be broad in their code search). Second, it can help you detect anomalies — a code in the "obesity" cluster that appears in the "respiratory" cluster might be an error, or might be a genuinely important comorbidity link (sleep apnoea, for example).

### 3.3 Hierarchical Clustering for Condition Overlap

Use `scipy.cluster.hierarchy.dendrogram` on the NICE example code lists to understand how much the different condition code lists overlap. This produces a tree structure showing which conditions share the most codes (e.g. obesity and hypertension share medication codes; obesity and T2DM share glucose-monitoring codes). This has direct practical value: it tells your agent which reference lists to consult when building a comorbidity code set.

**Key libraries:** `scikit-learn` (KMeans, t-SNE), `umap-learn`, `scipy`, `plotly`, `seaborn`

---

## Phase 4 — Supervised Learning: Predicting Code Inclusion

Now we use the NICE gold-standard example lists as labelled training data to train a classifier.

### 4.1 The Classification Framing

Reframe the problem: given all the features you've engineered for a clinical code (frequency, embedding vector, hierarchy depth, source flags, cluster membership), can we predict whether NICE would include it in a code list for condition X?

Your training data: for each NICE example list, codes in the list are labelled 1 (include); a random sample of similar codes *not* in the list are labelled 0 (exclude). This is a binary classification problem.

**Important caveat:** The negative class (codes not in the NICE list) is far larger than the positive class. This is class imbalance, and it will bias a naïve classifier toward always predicting "exclude". Use `imbalanced-learn`'s SMOTE or class weighting in sklearn to handle this.

### 4.2 Starting Simple: Logistic Regression as a Baseline

Before reaching for complex models, always establish a baseline. Logistic regression (`sklearn.linear_model.LogisticRegression`) is interpretable and fast. It will tell you, in clear coefficient terms, which features are most predictive of inclusion. You'll likely find that `in_qof` (binary flag), `log_frequency`, and `source_count` have the largest positive coefficients — meaning codes that appear in QOF, are used frequently, and appear in multiple sources are most likely to be included.

This interpretability is valuable for NICE because you can say: "the model is essentially applying a weighted version of the same heuristics your analysts use, just at scale."

### 4.3 Beyond Logistic Regression: Random Forest and Gradient Boosting

Logistic regression assumes linear decision boundaries. The real relationship between features and inclusion is likely non-linear (e.g. very high frequency codes that are too broad shouldn't be included; only high-frequency codes in the right semantic cluster should be). Random Forest (`sklearn.ensemble.RandomForestClassifier`) and Gradient Boosting (`sklearn.ensemble.GradientBoostingClassifier` or `xgboost`) can capture these interactions.

Use SHAP values (`shap.TreeExplainer`) to interpret these models. For each predicted inclusion, SHAP tells you exactly which features pushed the prediction toward "include" or "exclude". This directly supports NICE's auditability requirement — you can generate a per-code explanation like: *"Included: high usage frequency (+0.4), appears in QOF 2024-25 (+0.6), semantically close to confirmed obesity codes (+0.3). Not excluded: hierarchy depth is within expected range."*

### 4.4 Evaluation Strategy

Use stratified k-fold cross-validation (`sklearn.model_selection.StratifiedKFold`) rather than a simple train/test split, because your labelled dataset is relatively small (bounded by the number of NICE code lists you have). Measure precision, recall, F1, and AUROC. For this application, **recall is more important than precision** — it is worse to miss a relevant code (false negative) than to recommend a few extra codes that the analyst can then remove (false positive).

**Key libraries:** `scikit-learn`, `imbalanced-learn`, `shap`, `xgboost` (optional)

---

## Phase 5 — Time Series Analysis

The time series angle is often overlooked in projects like this, but it is genuinely relevant here.

### 5.1 Code Usage Trends

The NHS SNOMED usage data is published annually. If you can obtain multiple years of this data, you can model each clinical code as a time series of usage counts. Codes that are growing steadily in usage are becoming more standardised — they are increasingly likely to be the "right" code for a condition. Codes whose usage is declining may be deprecated or superseded.

Apply seasonal decomposition (`statsmodels.tsa.seasonal.seasonal_decompose`) and stationarity tests (Augmented Dickey-Fuller via `statsmodels.tsa.stattools.adfuller`) to characterise each code's time series before modelling.

### 5.2 ARIMA and Prophet for Forecasting

For codes with sufficient historical data, fit an ARIMA model (`pmdarima.auto_arima` to select orders automatically) or Meta's Prophet model to forecast future usage. This could allow you to flag codes that are "on the rise" — even if their current frequency is medium, they may soon become high-frequency mainstream codes worth including.

Prophet is particularly useful if you believe usage has annual seasonality (e.g. certain codes spike in winter, corresponding to flu season or cold-weather cardiovascular events).

### 5.3 Detecting Structural Breaks

SNOMED codes are periodically deprecated and replaced. A structural break in the time series — a sudden cliff-drop in usage — often signals that a code has been superseded. The `ruptures` library implements change-point detection algorithms that can automatically flag these events. This is directly relevant to NICE's requirement to monitor code lists for "drift" when SNOMED updates are released.

**Key libraries:** `statsmodels`, `prophet`, `pmdarima`, `sktime`

---

## Phase 6 — The Agentic RAG Pipeline

All previous phases were building blocks and learning exercises. This is the phase where they combine into the final system. See `02_agentic_workflow_design.md` for the full design, but here is the data science perspective.

The system works as follows: clinical code descriptions (and their associated features from Phases 2–4) are embedded and stored in a vector database (ChromaDB or FAISS). When a research question arrives, the agent uses semantic similarity search to retrieve candidate codes, then applies a scoring function that combines similarity score, usage frequency, QOF membership, and supervised model confidence to rank the candidates. The LLM then synthesises these ranked candidates into a structured code list with per-code rationale.

The entire pipeline is orchestrated with LangChain, which allows the agent to decide for itself when to search the vector store, when to look up QOF rules, and when to ask for clarification from the user.

**Key libraries:** `langchain`, `langchain-anthropic`, `chromadb`, `sentence-transformers`, `anthropic`

---

## Summary: How Each Technique Serves the Project

Correlation and linear regression give you interpretable baselines that build intuition and establish which features matter most before you touch a complex model. KMeans clustering reveals the natural structure of the clinical code space and helps detect when a research question spans multiple condition clusters. t-SNE and UMAP create visual maps of the code space that make it possible to sanity-check whether your embeddings are capturing clinical meaning. Supervised learning with gradient boosting and SHAP trains a code inclusion classifier on NICE gold-standard lists and produces per-code explanations that satisfy the auditability requirement. Time series analysis identifies code usage trends and detects deprecation events, supporting the code list monitoring requirement. The agentic RAG pipeline brings everything together into an end-to-end system that an analyst can interact with in plain English.

---

*Next: See `02_agentic_workflow_design.md` for the full system architecture.*
