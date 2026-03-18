# REPO-00 — Repository Structure: Gap Analysis and Definitive Layout

> **Purpose of this document.** The pasted repository skeleton is a reasonable starting point, but it was written without knowledge of our full planning suite (documents 00–10 and BT-00–BT-06). This document performs a systematic gap analysis and then presents the definitive, authoritative folder structure that every team member must use. It is the single source of truth for where everything lives.

---

## Part 1 — Gap Analysis: What the Original Structure Missed

The table below maps every folder, file, and artefact required by our planning and backtesting documents against what the original suggestion provided. Gaps are explained rather than just listed, because understanding *why* something is missing helps you understand *what it does* when you create it.

### Folder-Level Gaps

**`data/raw/open_code_counts/`** is missing. The OpenCodeCounts dataset (Bennett Institute / University of Oxford) is one of the four primary data sources in our project and has its own dedicated download script. It is not the same as `snomed_usage/` — OpenCodeCounts provides code-level frequency counts across GP patient records, while the NHS Digital SNOMED publication (`snomed_usage/`) is a different statistical release with different granularity and formatting. Conflating them into one folder would make the data provenance tracking in our PROV-O audit trail ambiguous.

**`data/raw/open_codelists/`** is missing. The OpenSAFELY / Bennett Institute OpenCodelists API returns validated, community-curated SNOMED code lists. These are a third-tier source (below QOF and NHS Reference Sets but above pure semantic search) and are explicitly cited in the agent's confidence tier logic in `02_agentic_workflow_design.md`. They require their own raw storage folder and an API response cache so repeated agent runs don't hit the API unnecessarily.

**`data/raw/nhs_data_dictionary/`** is missing. The NHS Data Dictionary is the reference source for code descriptions and hierarchical relationships. It is a distinct source from the Reference Sets — it provides the definitional content, not the operational subsets. The `03_understanding_qof.md` and `07_snomed_graph_architecture.md` documents both reference it as an ingestion target.

**`data/processed/features/`**, **`data/processed/embeddings/`**, and **`data/processed/merged/`** are missing as subdirectories. Our project produces several distinct types of processed data: the numerical feature matrix (from Phase 2), the embedding vectors (from Phase 2 Feature 2.2), and the merged Master Lookup Table (from `REPO-02`). Putting all of these into a flat `data/processed/` folder would produce a directory with dozens of files and no way to understand which were inputs to which downstream steps.

**`data/vectorstore/`** is entirely missing from the original structure. This is the ChromaDB persistent vector database that underpins the entire RAG semantic search pipeline. It is created once during Phase 2 and then read thousands of times by the agent and the backtesting suite. It must be version-controlled alongside the embedding model name and the data source versions that were used to build it (see `config/data_source_versions.yaml`).

**`data/backtest/`** and **`data/timeseries/`** are missing. The backtesting suite (BT-00 through BT-06) produces structured JSON and CSV outputs for each condition tested — precision/recall tables, attribution reports, cluster completeness matrices, temporal validity classifications, and XAI reports. These cannot live in `data/processed/` because they are outputs of the backtesting pipeline, not inputs to the modelling pipeline.

**`src/`** is entirely absent from the original structure. The `06_project_plan_and_features.md` explicitly specifies that reusable Python code must live in `src/` as importable modules, not in `scripts/` as one-off executable files. Every tool function for the agent (from `09_multi_agent_patterns.md`), every feature engineering function (from `10_annotated_code_skeletons.md`), and every backtesting function (from BT-01 through BT-06) needs to be a properly structured Python module in `src/` so that notebooks, scripts, and the agent can all import from a single shared location.

**`config/`** is missing. Our planning documents reference three configuration files: `scoring_weights.yaml` (the composite scoring weights from Feature 4.4 of the project plan), `data_source_versions.yaml` (the version tracking for QOF, SNOMED, and OpenCodeCounts that feeds the PROV-O audit trail), and `agent_config.yaml` (the LLM model name, temperature, and tool parameters). These must live in `config/` rather than hardcoded in source files so that any team member can update them without touching Python code.

**`outputs/`** is missing. The project plan specifies that every feature produces a shareable artefact. These artefacts — plots, reports, trained model files, provenance records — cannot live in `data/` (which is source data) or in `src/` (which is code). They need their own `outputs/` tree with subdirectories for plots, reports, provenance records, XAI reports, and backtesting results.

**`tests/`** is missing. The project plan requires unit tests for every agent tool (from `09_multi_agent_patterns.md` which specifies "write a unit test for each tool that verifies it returns the expected schema") and integration tests for the full pipeline. Without `tests/`, there is no way to run continuous validation as the codebase evolves.

**`mlruns/`** is missing. MLflow experiment tracking (specified in `05_auditability_and_monitoring.md`, Layer 4) stores its data in `mlruns/` by default. This directory should be in `.gitignore` (it can become very large) but must exist locally for MLflow to function.

**`docs/`** is missing. All seventeen markdown documents produced in our planning suite (00–10 and BT-00–BT-06) are the project's knowledge base. They need a dedicated home in the repository. Every team member cloning the repo should immediately find them in `docs/`.

**`.env.template`** and **`.gitignore`** are missing. The project plan explicitly states "never commit actual keys" and references a `.env` template. Without these two files, there is a risk that API keys (Anthropic, OpenAI) get accidentally committed to the repository.

### Script-Level Gaps

The original structure proposes three scripts. Our project needs eleven. The additional eight are: `download_open_code_counts.py` (separate from SNOMED usage), `download_nhs_refsets.py` (NHS Reference Set portal download), `download_open_codelists.py` (OpenCodelists API cache builder), `build_master_lookup_table.py` (the MLT construction pipeline from `REPO-02`), `build_vectorstore.py` (ChromaDB construction from `08_embeddings_and_semantic_search.md`), `run_agent.py` (the production agent entry point from `09_multi_agent_patterns.md`), `run_backtest.py` (the backtesting harness from BT-05), and `monitor_drift.py` (the scheduled drift monitoring from `05_auditability_and_monitoring.md`).

### Master Lookup Table Schema Gaps

The proposed MLT has four fields. Our documents collectively require seventeen. The missing thirteen fields are: `log_usage` (the log-transformed usage count that all models use as a feature), `source_count` (number of authoritative sources referencing this code, a critical scoring input), `cluster_id` (KMeans cluster assignment from Phase 3), `in_degree` (SNOMED graph in-degree from `07_snomed_graph_architecture.md`), `out_degree`, `subtree_size`, `degree_centrality` (all graph features), `deprecated_flag` (time series output from Phase 5), `usage_trend` (growing/stable/declining from Phase 5), `embedding_vector_id` (pointer to the ChromaDB record), `qof_indicators` (comma-separated list of QOF indicator IDs this code satisfies), `in_open_codelists` (whether the code appears in OpenSAFELY validated lists), and `data_source_versions` (the version snapshot used when this row was last updated, for PROV-O).

---

## Part 2 — The Definitive Repository Structure

```
nice-clinical-codes/
│
├── .env.template                   # API key template — copy to .env, never commit .env
├── .gitignore                      # Excludes .env, mlruns/, data/vectorstore/, __pycache__
├── README.md                       # Entry point — see REPO-03_project_readme.md for content
├── requirements.txt                # All Python dependencies — see the requirements.txt file
│
│── config/
│   ├── agent_config.yaml           # LLM model name, temperature, max_iterations, tool params
│   ├── data_source_versions.yaml   # Version tracking for QOF, SNOMED, OpenCodeCounts
│   └── scoring_weights.yaml        # Composite scoring weights (tuned in Feature 4.4)
│
├── data/
│   ├── raw/                        # Original, unmodified source files — never edit these
│   │   ├── snomed_usage/           # NHS Digital SNOMED Code Usage in Primary Care publications
│   │   │   └── usage_2024_25.txt   # Downloaded by scripts/download_snomed_usage.py
│   │   ├── qof_rules/              # QOF Business Rules Excel workbooks
│   │   │   └── qof_v49_2024_25/    # All tabs from the v49 workbook, extracted as-is
│   │   ├── nhs_ref_sets/           # NHS England Reference Set portal downloads
│   │   │   └── primary_care_domain/ # UK Primary Care Domain Reference Sets
│   │   ├── open_code_counts/       # Bennett Institute OpenCodeCounts dataset
│   │   │   └── opencodecounts_latest.csv
│   │   ├── open_codelists/         # OpenSAFELY/OpenCodelists API response cache (JSON)
│   │   │   └── api_cache/          # One JSON file per API response, named by codelist slug
│   │   └── nhs_data_dictionary/    # NHS Data Dictionary reference files
│   │       └── snomed_descriptions_uk_latest.txt  # SNOMED RF2 Descriptions file
│   │
│   ├── gold_standard/              # NICE example code lists — these are your ground truth
│   │   ├── DAAR_2025_004_all_prod_codes.txt
│   │   ├── DAAR_2025_004_antihypertensive_codes.txt
│   │   ├── DAAR_2025_004_ascvd_codes.txt
│   │   ├── DAAR_2025_004_BMI_codes.txt
│   │   ├── DAAR_2025_004_dyslipidemia_codes.txt
│   │   ├── DAAR_2025_004_ethnicity_codes.txt
│   │   ├── DAAR_2025_004_hdl_codes.txt
│   │   ├── DAAR_2025_004_hypertension_codes.txt
│   │   ├── DAAR_2025_004_ldl_codes.txt
│   │   ├── DAAR_2025_004_llt_codes.txt
│   │   ├── DAAR_2025_004_osa_codes.txt
│   │   ├── DAAR_2025_004_t2dm_codes.txt
│   │   └── DAAR_2025_004_triglycerides_codes.txt
│   │
│   ├── processed/                  # Cleaned, merged, feature-engineered outputs
│   │   ├── merged/                 # Master Lookup Table and joined datasets
│   │   │   ├── master_lookup_table.csv       # The MLT — central artefact of Phase 0
│   │   │   ├── nice_example_lists.csv        # All DAAR files merged into one DataFrame
│   │   │   └── snomed_with_usage.csv         # SNOMED codes joined with usage counts
│   │   ├── qof_v49_clean.csv                 # Parsed QOF rules (Feature 0.2)
│   │   ├── features/                         # Feature matrix outputs from Phase 2
│   │   │   ├── code_features.csv             # Numerical features per code
│   │   │   ├── graph_features.csv            # NetworkX-derived graph features
│   │   │   └── full_feature_matrix.csv       # Merged, ready for modelling
│   │   └── embeddings/                       # Embedding vectors and model metadata
│   │       ├── code_embeddings_pubmedbert.npy  # Shape: (n_codes, 768)
│   │       ├── code_index.csv                  # Maps row index → snomed_code
│   │       └── embedding_model_card.json       # Model name, version, batch size used
│   │
│   ├── vectorstore/                # ChromaDB persistent vector database
│   │   └── snomed_codes_pubmedbert/  # Named by embedding model for traceability
│   │       └── chroma.sqlite3        # ChromaDB internal storage
│   │
│   ├── backtest/                   # All backtesting outputs — separate from processed
│   │   ├── run_logs/               # Structured JSON run logs (one per agent run)
│   │   ├── attribution_reports/    # Pipeline attribution CSVs (one per condition)
│   │   ├── temporal_validity/      # Temporal classification results (BT-04)
│   │   └── final_grades/          # Condition-level grading rubric results (BT-06)
│   │
│   └── timeseries/                 # Historical usage data for time series analysis
│       └── usage_history/          # One CSV per code with reporting_date, usage_count
│
├── src/                            # Importable Python modules — the project's shared library
│   ├── __init__.py
│   ├── ingestion/                  # Data loading and validation functions
│   │   ├── __init__.py
│   │   ├── load_qof.py             # QOF Business Rules parser (Feature 0.2)
│   │   ├── load_gold_standard.py   # DAAR files parser (Feature 0.3)
│   │   ├── load_usage_data.py      # SNOMED usage and OpenCodeCounts loaders
│   │   └── load_ref_sets.py        # NHS Reference Set loaders
│   │
│   ├── features/                   # Feature engineering functions (Phase 2)
│   │   ├── __init__.py
│   │   ├── numerical_features.py   # log_usage, source_count, etc.
│   │   ├── graph_features.py       # NetworkX graph feature computation
│   │   ├── embeddings.py           # SentenceTransformer embedding generation
│   │   └── feature_matrix.py       # Assembles the full feature matrix
│   │
│   ├── models/                     # ML model training and inference (Phase 4)
│   │   ├── __init__.py
│   │   ├── classifier.py           # Random Forest + SHAP code inclusion classifier
│   │   ├── clustering.py           # KMeans + UMAP + t-SNE (Phase 3)
│   │   └── timeseries.py           # Prophet + ruptures time series analysis (Phase 5)
│   │
│   ├── tools/                      # Agent tool library (for LangChain @tool decorators)
│   │   ├── __init__.py
│   │   ├── qof_lookup.py           # Tool 1: QOF Business Rules lookup
│   │   ├── semantic_search.py      # Tool 2: ChromaDB semantic similarity search
│   │   ├── hierarchy_explorer.py   # Tool 3: SNOMED polyhierarchy traversal
│   │   ├── usage_lookup.py         # Tool 4: OpenCodeCounts usage frequency lookup
│   │   ├── open_codelists.py       # Tool 5: OpenSAFELY validated codelist lookup
│   │   └── scoring.py              # Tool 6: Composite scoring and ranking function
│   │
│   ├── agent/                      # LangChain / LangGraph orchestration (Phase 6)
│   │   ├── __init__.py
│   │   ├── prompts.py              # System prompts — the NICE reasoning protocol
│   │   ├── executor.py             # AgentExecutor setup (LangChain)
│   │   └── graph.py                # LangGraph multi-agent state machine
│   │
│   ├── audit/                      # Provenance and auditability (05_auditability_and_monitoring)
│   │   ├── __init__.py
│   │   ├── run_logger.py           # RunLogger class — structured JSON run logs
│   │   ├── provenance.py           # ProvenanceRecord dataclass + PROV-O generation
│   │   └── xai_report.py           # XAI report generator (human-readable audit output)
│   │
│   └── backtest/                   # Backtesting functions (BT-01 through BT-06)
│       ├── __init__.py
│       ├── usage_baseline.py       # BT-01: Usage frequency profile comparison
│       ├── structural.py           # BT-02: Polyhierarchy recall analysis
│       ├── clustering.py           # BT-03: Cluster completeness audit
│       ├── temporal.py             # BT-04: Temporal validity and drift detection
│       ├── attribution.py          # BT-05: Pipeline error attribution
│       └── grading.py              # BT-06: NICE grading rubric and metrics
│
├── scripts/                        # One-off executable scripts — thin wrappers around src/
│   ├── download_snomed_usage.py    # Pull NHS Digital SNOMED usage publication
│   ├── download_open_code_counts.py  # Pull Bennett Institute OpenCodeCounts dataset
│   ├── download_nhs_refsets.py     # Pull NHS Reference Set portal data
│   ├── download_open_codelists.py  # Cache OpenSAFELY API responses locally
│   ├── parse_qof.py                # Parse QOF Excel workbook → processed CSV
│   ├── initialize_gold_standard.py # Validate and register DAAR files (not a placeholder)
│   ├── build_master_lookup_table.py  # Assemble the MLT from all processed sources
│   ├── build_vectorstore.py        # Embed all codes and build ChromaDB index
│   ├── run_agent.py                # CLI entry point for the production agent
│   ├── run_backtest.py             # Execute the full backtesting suite
│   └── monitor_drift.py            # Scheduled drift monitoring for archived code lists
│
├── notebooks/                      # Exploratory Jupyter notebooks — numbered by phase
│   ├── phase_0_ingestion/          # Data loading and validation notebooks
│   │   ├── 00_01_load_qof.ipynb
│   │   ├── 00_02_load_gold_standard.ipynb
│   │   └── 00_03_load_usage_data.ipynb
│   ├── phase_1_eda/                # EDA notebooks (Features 1.1–1.4)
│   │   ├── 01_01_usage_frequency_distribution.ipynb
│   │   ├── 01_02_correlation_analysis.ipynb
│   │   ├── 01_03_code_list_overlap.ipynb
│   │   └── 01_04_data_quality_report.ipynb
│   ├── phase_2_features/           # Feature engineering notebooks
│   │   ├── 02_01_numerical_features.ipynb
│   │   ├── 02_02_embeddings.ipynb
│   │   └── 02_03_vectorstore_build.ipynb
│   ├── phase_3_clustering/         # Unsupervised learning notebooks
│   │   ├── 03_01_kmeans_elbow.ipynb
│   │   ├── 03_02_tsne_umap_visualisation.ipynb
│   │   └── 03_03_multimorbidity_network.ipynb
│   ├── phase_4_supervised/         # Supervised learning notebooks
│   │   ├── 04_01_dataset_construction.ipynb
│   │   ├── 04_02_logistic_regression_baseline.ipynb
│   │   ├── 04_03_random_forest_shap.ipynb
│   │   └── 04_04_scoring_function.ipynb
│   ├── phase_5_timeseries/         # Time series analysis notebooks
│   │   ├── 05_01_usage_trends.ipynb
│   │   └── 05_02_deprecation_detection.ipynb
│   ├── phase_6_agent/              # Agent assembly and evaluation notebooks
│   │   ├── 06_01_tool_testing.ipynb
│   │   ├── 06_02_agent_assembly.ipynb
│   │   └── 06_03_evaluation.ipynb
│   └── backtesting/                # One notebook per BT document
│       ├── BT01_eda_baseline.ipynb
│       ├── BT02_structural_analysis.ipynb
│       ├── BT03_cluster_completeness.ipynb
│       ├── BT04_temporal_validity.ipynb
│       ├── BT05_pipeline_attribution.ipynb
│       └── BT06_grading_rubric.ipynb
│
├── outputs/                        # All generated artefacts — sharable with NICE
│   ├── plots/                      # All matplotlib/plotly figures saved as PNG/HTML
│   ├── reports/                    # Written summary reports (Phase-level and condition-level)
│   ├── provenance/                 # PROV-O JSON-LD records — one folder per run_id
│   │   └── {run_id}/               # e.g. run_2026_03_14_obesity_t2dm_001/
│   │       └── {snomed_code}.jsonld
│   ├── xai_reports/                # Human-readable XAI audit reports (BT-06 format)
│   └── backtest_results/           # Final graded reports from the backtesting suite
│
├── tests/                          # Automated tests — run with pytest
│   ├── unit/                       # Unit tests for src/ modules
│   │   ├── test_tools.py           # Tests for every agent tool (required by Feature 6.1)
│   │   ├── test_features.py        # Tests for feature engineering functions
│   │   └── test_ingestion.py       # Tests for data loading functions
│   └── integration/                # End-to-end pipeline integration tests
│       ├── test_agent_pipeline.py  # Tests the full agent run for one condition
│       └── test_backtest_suite.py  # Tests the full backtesting run
│
├── mlruns/                         # MLflow experiment tracking — in .gitignore
│   └── (auto-generated by MLflow)
│
└── docs/                           # All project planning and reference documents
    ├── planning/                   # The main project series (00–10)
    │   ├── 00_project_brief_plain_english.md
    │   ├── 01_data_science_roadmap.md
    │   ├── 02_agentic_workflow_design.md
    │   ├── 03_understanding_qof.md
    │   ├── 04_rag_pipeline_deep_dive.md
    │   ├── 05_auditability_and_monitoring.md
    │   ├── 06_project_plan_and_features.md
    │   ├── 07_snomed_graph_architecture.md
    │   ├── 08_embeddings_and_semantic_search.md
    │   ├── 09_multi_agent_patterns.md
    │   └── 10_annotated_code_skeletons.md
    ├── backtesting/                # The BT series (BT-00–BT-06)
    │   ├── BT-00_backtesting_master_overview.md
    │   ├── BT-01_eda_usage_baseline_backtesting.md
    │   ├── BT-02_feature_engineering_backtesting.md
    │   ├── BT-03_clustering_multimorbidity_backtesting.md
    │   ├── BT-04_timeseries_drift_backtesting.md
    │   ├── BT-05_multiagent_self_diagnosis.md
    │   └── BT-06_evaluation_framework_grading_rubric.md
    └── repository/                 # The REPO series (REPO-00–REPO-03)
        ├── REPO-00_repository_structure.md   ← this document
        ├── REPO-01_data_ingestion_scripts.md
        ├── REPO-02_master_lookup_table.md
        └── REPO-03_project_readme.md
```

---

## Part 3 — File Content Templates for Config Files

These three configuration files are referenced throughout the codebase. Their content is defined here so that every team member initialises them identically.

### `config/data_source_versions.yaml`

```yaml
# data_source_versions.yaml
# Every agent run reads this file to stamp provenance records with version info.
# Update these values whenever a new data source release is downloaded.
# Do NOT change keys — only change values. These keys are referenced in src/audit/provenance.py.

qof_business_rules:
  version: "v49_2024-25"
  download_date: "2026-03-01"
  source_url: "https://digital.nhs.uk/data-and-information/data-collections-and-data-sets/data-collections/quality-and-outcomes-framework-qof/business-rules"

snomed_usage_primary_care:
  version: "2024-25"
  download_date: "2026-02-15"
  source_url: "https://digital.nhs.uk/data-and-information/publications/statistical/mi-snomed-code-usage-in-primary-care"

open_code_counts:
  version: "2025-Q3"
  download_date: "2026-01-20"
  source_url: "https://bennettoxford.github.io/opencodecounts/"

nhs_reference_sets:
  version: "UK_20241001"
  download_date: "2026-02-01"
  source_url: "https://digital.nhs.uk/services/terminology-and-classifications/snomed-ct/snomed-ct-uk-drug-extension-and-snomed-ct-uk-edition"

embedding_model:
  name: "pritamdeka/S-PubMedBert-MS-MARCO"
  version: "huggingface-latest"
  embedding_dim: 768
  vectorstore_built_date: "2026-03-10"
```

### `config/scoring_weights.yaml`

```yaml
# scoring_weights.yaml
# Composite scoring weights for the score_and_rank_candidates() tool.
# Weights must sum to 1.0. Tune these during Feature 4.4 using grid search
# against the NICE gold-standard lists. Current values are sensible defaults.
# Reference: 04_rag_pipeline_deep_dive.md, 06_project_plan_and_features.md Feature 4.4

weights:
  qof_membership:        0.40   # Highest: QOF codes are nationally mandated
  usage_frequency_pctl:  0.25   # High: reflects real-world clinical activity
  source_count:          0.15   # Medium: codes in more sources = more validated
  classifier_probability: 0.12  # Medium: supervised model trained on NICE lists
  semantic_similarity:   0.08   # Lower: exploratory signal, not authoritative

thresholds:
  high_confidence_min:   0.75   # Composite score >= this → HIGH tier
  medium_confidence_min: 0.50   # Composite score >= this → MEDIUM tier
  review_below:          0.50   # Composite score < this → REVIEW tier

usage_frequency_minimums:
  high_confidence:       100000  # Annual occurrences required for HIGH without QOF
  medium_confidence:     10000   # Annual occurrences required for MEDIUM
  review_trigger:        1000    # Below this triggers REVIEW flag regardless of other scores
```

### `config/agent_config.yaml`

```yaml
# agent_config.yaml
# Agent configuration — LLM settings and tool parameters.
# Reference: 09_multi_agent_patterns.md

llm:
  provider: "anthropic"
  model: "claude-opus-4-6"        # Use Opus for production; Sonnet for dev/testing
  temperature: 0                  # Always 0 for reproducibility in healthcare context
  max_tokens: 4096

agent:
  framework: "langgraph"          # Use LangGraph for production multi-agent flow
  max_iterations: 15              # Safety guard against infinite loops
  verbose: true                   # Log all reasoning steps (required for audit trail)
  return_intermediate_steps: true # Required for RunLogger and backtesting

tools:
  semantic_search_top_k: 25      # Number of candidates returned per semantic query
  hierarchy_max_depth: 6         # Max levels to traverse in SNOMED polyhierarchy
  qof_fuzzy_match_threshold: 80  # rapidfuzz score threshold for condition matching
```

---

## Part 4 — `.gitignore` and `.env.template`

### `.gitignore`

```gitignore
# Environment and secrets — never commit these
.env
*.pem
*.key

# Data files — too large for Git; use DVC or share via secure channel
data/raw/
data/vectorstore/
data/timeseries/
mlruns/

# Python artefacts
__pycache__/
*.pyc
*.pyo
.pytest_cache/
*.egg-info/

# Notebook checkpoints
.ipynb_checkpoints/

# OS artefacts
.DS_Store
Thumbs.db

# Trained model files — version these separately
*.pkl
*.joblib
*.pt
*.bin
```

### `.env.template`

```bash
# .env.template — copy this file to .env and fill in your values
# Never commit .env to Git
# Reference: 06_project_plan_and_features.md Feature 0.1

# LLM API Keys
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENAI_API_KEY=your_openai_api_key_here    # Optional fallback

# MLflow tracking server (optional — defaults to local mlruns/)
MLFLOW_TRACKING_URI=http://localhost:5000

# NHS API access (if using TRUD for Reference Set downloads)
NHS_TRUD_API_KEY=your_trud_api_key_here

# Data paths (override if not using default repo structure)
DATA_ROOT=./data
VECTORSTORE_PATH=./data/vectorstore/snomed_codes_pubmedbert
```

---

*This is the definitive structure. All other repository documents (REPO-01, REPO-02, REPO-03) reference these paths and file locations. If a path in any document conflicts with this one, this document takes precedence.*
