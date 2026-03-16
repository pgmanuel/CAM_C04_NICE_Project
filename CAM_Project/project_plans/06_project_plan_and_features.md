# 06 — Project Plan: Phases, Features, and Team Task Breakdown

> **How to use this document.** This is the working project plan. It breaks the full project into six phases with clear deliverables, then decomposes each phase into assignable features — discrete, bounded pieces of work that a single team member can own from start to finish. Each feature has a definition of "done" so there is no ambiguity about when it is complete.

---

## Guiding Principles for the Plan

Before diving into the phases, it is worth being explicit about three principles that should govern how the team works.

**Build in order of trust.** The NICE project is fundamentally about producing outputs that NICE analysts will trust enough to act on. Every phase is ordered so that the team understands the data and builds confidence in their findings before adding complexity. We do not start with the agentic workflow. We start with the data — understanding its shape, its gaps, its quirks — and we earn the right to build the agent by first demonstrating that our foundations are solid.

**Every feature should produce a shareable artefact.** Whether it is a plot, a cleaned DataFrame, a trained model, a JSON provenance record, or a markdown summary of findings, every feature should produce something concrete that can be reviewed by the rest of the team. This prevents situations where a team member has done substantial work that only lives in their head or an untested notebook.

**Treat uncertainty as a first-class output.** Throughout the project, when you don't know something — whether a code is correct, whether a cluster is clinically meaningful, whether a model's prediction is reliable — that uncertainty should be explicitly represented in the output rather than hidden behind a single number or a confident-sounding sentence. Surfacing uncertainty is a deliverable, not a sign of incomplete work.

---

## Phase 0 — Environment, Data Ingestion, and Baseline Understanding

**Duration estimate:** 1–2 weeks  
**Goal:** Everyone on the team can open a notebook, load all data sources into clean DataFrames, and describe the shape of the data in plain English. No modelling happens in this phase.

### Feature 0.1 — Repository Setup and Shared Conventions

The team needs a shared codebase from the start, not a collection of individual notebooks that diverge. Set up a Git repository with a standard folder structure: `data/raw/` for original source files, `data/processed/` for cleaned outputs, `notebooks/` for exploratory work (numbered sequentially per phase), `src/` for reusable Python modules, `outputs/` for artefacts to share with NICE, and `tests/` for validation scripts.

Install all packages from `requirements.txt`, set up a shared `.env` template for API keys (never commit actual keys), and configure MLflow with a local tracking server. The definition of done for this feature is that every team member can clone the repo, run `pip install -r requirements.txt`, and successfully execute a "hello world" notebook that loads one data file and prints its shape.

### Feature 0.2 — QOF Business Rules Ingestion

Parse the QOF Business Rules v49 2024–25 Excel workbooks into a clean, unified DataFrame. The target schema is `[indicator_id, domain, condition_name, snomed_code, code_description, refset_id]`. This will require careful `openpyxl` handling because the workbooks have complex multi-tab structures with merged cells and linked reference set files.

The definition of done is a saved `data/processed/qof_v49_clean.csv` with no null codes, a documented count of how many indicators and codes were successfully parsed, and a short notebook section that demonstrates a lookup by condition name returning the expected results.

### Feature 0.3 — NICE Example Code Lists Ingestion

Load all DAAR_2025_004 `.txt` files into a unified DataFrame with schema `[source_file, snomed_code, description, condition_category]`. The condition category should be derived from the filename (e.g., `DAAR_2025_004_hypertension_codes.txt` → `hypertension`). Parse carefully, checking for encoding issues (UTF-8 vs Latin-1 is a common problem with NHS data files) and inconsistent delimiters.

The definition of done is a saved `data/processed/nice_example_lists.csv` and a printed summary showing the code count per condition — this will be your first data point about the expected scale of code lists for different conditions.

### Feature 0.4 — OpenCodeCounts Usage Data Ingestion

Download and parse the OpenCodeCounts dataset published by the Bennett Institute. This is a CSV-style publication with schema roughly `[snomed_code, description, annual_count, gp_count]`. Join it with the NICE example lists DataFrame on `snomed_code` to immediately see the usage frequency of codes that NICE has previously selected. This join is one of the most informative things you can do in the entire project.

The definition of done is a saved `data/processed/snomed_with_usage.csv` and a preliminary finding: what proportion of NICE example list codes have high (>100,000 annual occurrences), medium (1,000–100,000), or low (<1,000) usage frequencies?

---

## Phase 1 — Exploratory Data Analysis (EDA)

**Duration estimate:** 1–2 weeks  
**Goal:** Characterise the data in detail. Produce a set of EDA findings that will directly inform modelling decisions in later phases. Every finding should be written up in a shared notebook with a plain-English interpretation alongside every visualisation.

### Feature 1.1 — Usage Frequency Distribution Analysis

Plot the distribution of national usage frequencies for all SNOMED codes in OpenCodeCounts. Use a log-scale histogram because the distribution is a power law (a handful of codes are used tens of millions of times; the vast majority are used rarely). Compute the median, mean, and key percentiles (25th, 75th, 95th, 99th). Then overlay the distribution of NICE example list codes on the same plot. This visualisation will almost certainly show that NICE codes cluster toward the high end of the frequency distribution — confirming that usage frequency is a valid predictive feature.

### Feature 1.2 — Correlation Analysis of Features vs. Inclusion

For each code in the combined dataset, compute the engineered features described in Phase 2 of the roadmap (you can approximate them at this stage with simpler versions: `in_qof` binary flag, `log_usage`, `source_count`). Compute a Pearson correlation matrix between all features and the binary label "in NICE example list". Visualise as a heatmap. Write up which features have the strongest positive correlation with inclusion — this will form the basis of the feature importance discussion in the supervised learning phase.

### Feature 1.3 — Code List Overlap Analysis (Hierarchical Clustering on NICE Lists)

Compute a pairwise overlap matrix between all NICE example code lists: for each pair of conditions, what fraction of their codes are shared? Visualise this as a dendrogram using `scipy.cluster.hierarchy`. The dendrogram will reveal which conditions are most similar in their code requirements. Specifically, we expect to see obesity cluster closely with T2DM and hypertension, and hypertension cluster closely with dyslipidaemia — reflecting the known clinical relationship between these metabolic conditions. This is the first concrete test of whether the data structure matches clinical knowledge.

### Feature 1.4 — Missing Value and Data Quality Report

Produce a systematic assessment of data quality across all sources: what proportion of codes in QOF have matching entries in OpenCodeCounts? What proportion of NICE example list codes appear in QOF? Are there codes in the NICE lists that have no description in the NHS Data Dictionary? Document all findings. These gaps are not problems to be solved immediately — they are realities to be understood and communicated to NICE.

---

## Phase 2 — Feature Engineering and Embeddings

**Duration estimate:** 2 weeks  
**Goal:** Build the feature matrix and the embedding-based vector store that underpin all modelling in subsequent phases. This is primarily engineering work, but it has significant research decisions embedded in it.

### Feature 2.1 — Numerical Feature Engineering

Compute the full feature vector for each SNOMED code in the combined dataset: `log_usage_frequency`, `in_qof` (binary), `source_count` (how many of our data sources mention this code), `hierarchy_depth` (approximate, from SNOMED tree structure), and `description_character_length`. Save as `data/processed/code_features.csv`. Verify that the features match the intuitions from the EDA phase — `in_qof` should be positively correlated with `log_usage_frequency`.

### Feature 2.2 — Text Embedding Generation

Embed all SNOMED code descriptions using `sentence-transformers`. Start with `all-MiniLM-L6-v2` as a fast baseline, then run `pritamdeka/S-PubMedBert-MS-MARCO` as the biomedical-domain model. For each model, compute embeddings for all codes and store them as a numpy array with a corresponding code index. Measure wall-clock time for both models on your full dataset — this is relevant because the biomedical model is slower and if it is not meaningfully better on a retrieval test, the speed trade-off may not be worth it.

To evaluate which model is better, take ten NICE example list codes as "query codes", retrieve the top-20 nearest neighbours for each from the full SNOMED dataset, and ask a team member to judge whether the neighbours are clinically relevant. This is a quick human evaluation that will guide the choice of embedding model for all downstream work.

### Feature 2.3 — ChromaDB Vector Store Construction

Load all embeddings from Feature 2.2 into ChromaDB with the following metadata schema per code: `snomed_code`, `description`, `in_qof` (boolean), `log_usage`, `source_count`, `qof_indicator_ids` (list), `data_source_versions` (dict). Persist the vector store to disk at `data/vectorstore/snomed_codes_pubmedbert`. Write a `retrieve(query: str, top_k: int) -> list[dict]` function and test it against five manually chosen research queries, verifying that the results are clinically plausible.

---

## Phase 3 — Unsupervised Learning (Clustering and Dimensionality Reduction)

**Duration estimate:** 1–2 weeks  
**Goal:** Understand the natural structure of the clinical code space. Produce visualisations that the team and NICE stakeholders can use to interpret the embedding space.

### Feature 3.1 — KMeans Clustering with Elbow and Silhouette Analysis

Apply KMeans to the embedding matrix from Phase 2. Test K values from 5 to 50 and compute inertia and silhouette score for each. Plot the elbow curve. Select a final K and assign cluster IDs to all codes. Write a cluster characterisation notebook: for each cluster, show the top-10 most central codes (by proximity to the cluster centroid) and give each cluster a descriptive label (e.g., "Cluster 4 — Metabolic and Endocrine Disorders"). Add the cluster ID as a new column in `code_features.csv` and push to the vector store as metadata.

### Feature 3.2 — t-SNE and UMAP Visualisations

Apply t-SNE with `perplexity=30` and UMAP with `n_neighbours=15` to a sample of 20,000 codes from the embedding matrix. Produce interactive Plotly scatter plots coloured by cluster ID, with hover tooltips showing the code and description. Produce a second version coloured by "in NICE example list" to visualise whether NICE-selected codes cluster together or are distributed across the code space. Share these visualisations with the full team as a checkpoint — they should provoke discussion about whether the clustering makes clinical sense.

### Feature 3.3 — Multimorbidity Network Analysis

Implement the network-based approach referenced in the research document. Build a co-occurrence graph where nodes are clinical conditions and edge weights represent the number of SNOMED codes shared between their code lists (using the NICE example lists as the node-condition mapping). Use `networkx` to visualise this graph and apply a community detection algorithm to identify the multimorbidity clusters referenced in the academic literature (cardiovascular cluster, metabolic cluster, mixed cardiometabolic cluster). Compare the communities found computationally with what clinical literature describes — this is your second major validation test.

---

## Phase 4 — Supervised Learning and SHAP Explainability

**Duration estimate:** 2 weeks  
**Goal:** Train, evaluate, and explain a code inclusion classifier. The trained classifier and its SHAP explainer become permanent components of the agent's scoring function.

### Feature 4.1 — Dataset Construction for Classification

Build the labelled training dataset. Positive examples are codes from the NICE example lists (label = 1). Negative examples require care: do not randomly sample from all SNOMED codes, as the vast majority are from completely unrelated clinical domains. Instead, sample negatives from codes in the same semantic neighbourhood (same t-SNE region or same KMeans cluster) as the positive examples, but not in the NICE lists. This is called "hard negative mining" and produces a classifier that learns meaningful clinical distinctions rather than trivially obvious ones. Document the final class distribution and the sampling strategy.

### Feature 4.2 — Baseline Logistic Regression

Train a logistic regression classifier on the labelled dataset from Feature 4.1. Use stratified k-fold cross-validation with five folds. Report precision, recall, F1, and AUROC. Log results to MLflow. Report the model coefficients — which features have the largest positive coefficients? This should align with the correlations found in Feature 1.2; if it doesn't, investigate why.

### Feature 4.3 — Random Forest with SHAP

Train a Random Forest classifier with the same cross-validation setup. Compare performance to logistic regression. Compute SHAP values using `shap.TreeExplainer`. Produce the following SHAP plots: a global feature importance bar chart, a SHAP beeswarm plot showing the distribution of feature contributions, and per-code SHAP waterfall plots for five specific example codes (two high-confidence inclusions, two borderline cases, one clear exclusion). These plots form part of the project's final presentation to NICE.

### Feature 4.4 — Composite Scoring Function

Build the `score_and_rank_candidates()` function that combines the Random Forest probability, usage frequency percentile, QOF membership flag, and source count into a composite score. Tune the weights using grid search against the NICE gold-standard lists. Save the final weights as a YAML configuration file so they can be updated without changing code. This function is the last piece needed before the agentic pipeline can be assembled.

---

## Phase 5 — Time Series Analysis

**Duration estimate:** 1 week (can be run in parallel with Phase 4)  
**Goal:** Enrich the code metadata with trend and deprecation signals from historical usage data.

### Feature 5.1 — Usage Trend Analysis

If multiple years of NHS SNOMED usage data are available, build a time series for each code and fit a linear trend. Classify each code as "growing", "stable", or "declining" based on the trend slope and significance. Add this classification as a metadata field in the vector store. If only one year of data is available, document this as a limitation and note that the time series analysis should be revisited when subsequent data releases become available.

### Feature 5.2 — Deprecation Detection

Apply change-point detection (using the `ruptures` library) to any codes that show a sudden step-change in usage. Flag codes where the post-change usage is near-zero as "potentially deprecated". Cross-reference with SNOMED release notes where available to validate the flags. The output is a deprecation flag field in the vector store that the agent checks before including any code.

---

## Phase 6 — Agentic Pipeline Assembly and Evaluation

**Duration estimate:** 3 weeks  
**Goal:** Assemble all components into the working agent, implement the full auditability stack, evaluate against the NICE gold-standard lists, and produce a presentation-ready demonstration.

### Feature 6.1 — Tool Library Implementation

Implement all six agent tools described in `02_agentic_workflow_design.md` as properly typed, documented Python functions decorated with the `@tool` decorator. Each tool should have comprehensive docstrings — these are read by the LLM to decide when to call the tool, so the quality of the docstring directly affects the quality of the agent's reasoning. Write a unit test for each tool that verifies it returns the expected schema.

### Feature 6.2 — Agent Assembly with LangChain

Assemble the `AgentExecutor` with all six tools, the Claude model via the Anthropic API, and the system prompt from `02_agentic_workflow_design.md`. Implement the `RunLogger` class from `05_auditability_and_monitoring.md`. Run the agent against three test queries: obesity alone, hypertension alone, and obesity with T2DM. Inspect the intermediate steps. Debug any cases where the agent makes unexpected tool calls or reaches incorrect conclusions.

### Feature 6.3 — Provenance Record Generation

Implement the `ProvenanceRecord` dataclass and the post-processing step that translates SHAP contributions and tool call outputs into per-code provenance records. Verify that for a sample of ten codes, every field in the provenance record is populated and the rationale text accurately reflects the underlying evidence. This feature directly implements the auditability requirement.

### Feature 6.4 — Evaluation Against NICE Gold-Standard Lists

Run the completed agent against all DARE_2025_004 conditions and measure performance using the evaluation metrics from `05_auditability_and_monitoring.md`: Cohen's Kappa, recall at K, precision at K, and count of borderline codes correctly flagged for review. Log all results to MLflow. Write a finding summary: where does the system perform best? Where are the biggest gaps? What types of codes does it tend to miss?

### Feature 6.5 — Final Demonstration Build

Prepare a demonstration that can be shown to NICE stakeholders. This should include: an interactive run of the agent on a live query, the t-SNE visualisation with NICE codes highlighted, three example provenance records showing the evidence chain for a HIGH, MEDIUM, and REVIEW code, the drift monitoring report structure (can be simulated if new QOF data is not yet available), and a one-page evaluation summary showing performance metrics with comparison to a baseline of "QOF only" (i.e., what performance would you get if the code list was simply every code in QOF for the target condition, with no AI assistance?). That baseline comparison is critical — it tells NICE exactly what value the AI component is adding.

---

## Summary Timeline

The following is an indicative timeline assuming a team of three to four people working in parallel across phases where tasks permit. Phases 1 and 2 can begin simultaneously once Phase 0 is complete. Phase 4 and Phase 5 can run in parallel. Phase 6 requires both to be complete first.

Week 1–2: Phase 0 (Environment and Data Ingestion)
Week 3–4: Phase 1 (EDA) running alongside Phase 2 Feature 2.1 (Feature Engineering)
Week 5–6: Phase 2 Features 2.2 and 2.3 (Embeddings and Vector Store)
Week 7–8: Phase 3 (Clustering) and Phase 5 (Time Series) in parallel
Week 9–10: Phase 4 (Supervised Learning)
Week 11–13: Phase 6 (Agentic Pipeline Assembly and Evaluation)
Week 14: Final demonstration preparation and documentation

---

*This completes the six-document planning package. Reading order: `00` → `01` → `02` → `03` → `04` → `05` → `06`.*
