# NICE Clinical Code Automation: Analytics, Modelling, and Multi-Agent Evaluation Framework

> *Building defensible, explainable, and auditable AI-assisted clinical code lists for the National Institute for Health and Care Excellence.*

---

## What This Project Does

NICE analysts currently spend days or weeks manually searching through thousands of SNOMED CT clinical codes to define the patient cohorts needed for healthcare research and guidance development. This project builds an AI-assisted system that can receive a plain-English research question — such as "patients with obesity who also have type 2 diabetes" — and return a structured, sourced, and auditable list of the clinical codes that would identify that patient population in NHS primary care data.

The system is not a replacement for expert clinical judgement. It is a force multiplier: instead of an analyst spending a week building a code list from scratch, they spend an afternoon reviewing and approving a system-generated list that already cites its sources, flags uncertain cases, and explains why each code was included. Every recommendation is traceable to an authoritative source — the QOF Business Rules, the NHS England Reference Sets, or empirical usage data from 62 million patient records.

---

## Before You Begin: Reading Order

This repository has seventeen planning and reference documents in `docs/`. Reading them all before touching any code is the fastest path to understanding the system. The recommended order is as follows.

Start with `docs/planning/00_project_brief_plain_english.md` to understand what problem we are solving and why it matters for NICE. Then read `docs/planning/01_data_science_roadmap.md` to understand how the project is structured in six analytical phases, each building on the last. The documents `03_understanding_qof.md` and `07_snomed_graph_architecture.md` cover the two most important domain concepts — QOF and the SNOMED polyhierarchy — that every technique in the project depends on. Before writing any code, read `docs/repository/REPO-00_repository_structure.md` so you understand exactly where every file belongs.

---

## Quick Start: Setting Up Your Environment

Clone the repository and navigate into it. Copy `.env.template` to `.env` and fill in your API keys — the minimum required is `ANTHROPIC_API_KEY` for the agent to run. Install all Python dependencies using the requirements file. Run the post-install steps for spaCy and NLTK. Then verify the setup by running the environment test.

```bash
git clone https://github.com/your-org/nice-clinical-codes.git
cd nice-clinical-codes

cp .env.template .env
# Edit .env and add your API keys

pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
python -m nltk.downloader punkt stopwords wordnet

# Verify setup
python -c "import chromadb, langchain, sentence_transformers, networkx; print('All core packages OK')"
```

---

## Quick Start: Ingesting Data (Phase 0)

The project requires data from four external sources before any modelling can begin. Run the ingestion scripts in this exact order — each produces an output that the next depends on. Detailed documentation for each script is in `docs/repository/REPO-01_data_ingestion_scripts.md`.

```bash
# 1. Download NHS Digital SNOMED usage data
python scripts/download_snomed_usage.py --year 2024-25

# 2. Download OpenCodeCounts (Bennett Institute)
python scripts/download_open_code_counts.py

# 3. Download NHS Reference Sets (requires NHS TRUD account — see .env.template)
python scripts/download_nhs_refsets.py

# 4. Cache relevant OpenSAFELY codelists locally
python scripts/download_open_codelists.py

# 5. Place your DAAR_2025_004 files in data/gold_standard/ then validate them
python scripts/initialize_gold_standard.py

# 6. Parse QOF Business Rules Excel workbook
#    Download the v49 Excel file from:
#    https://digital.nhs.uk/data-and-information/data-collections-and-data-sets/
#    data-collections/quality-and-outcomes-framework-qof/business-rules
python scripts/parse_qof.py --input data/raw/qof_rules/qof_v49_2024_25/

# 7. Build the Master Lookup Table — the central artefact
python scripts/build_master_lookup_table.py
```

After these steps, `data/processed/merged/master_lookup_table.csv` exists and all notebooks can be opened. See `docs/repository/REPO-02_master_lookup_table.md` for the full field-by-field schema documentation.

---

## Quick Start: Running the Agent

Once Phase 0 through Phase 5 are complete and the vector store has been built, you can run the agent from the command line. The agent will write its output — a tiered code list with per-code rationale — to `outputs/` and archive the PROV-O provenance record to `outputs/provenance/{run_id}/`.

```bash
# Build the vector store (run once after Phase 2)
python scripts/build_vectorstore.py

# Run the agent with a research question
python scripts/run_agent.py \
    --question "Identify codes for patients with obesity who also have type 2 diabetes" \
    --output-dir outputs/
```

---

## Quick Start: Running the Backtesting Suite

The backtesting suite evaluates the agent's performance against the NICE gold-standard code lists and produces a graded report. Run it after any significant change to the agent's tools, scoring function, or prompts to check whether performance has improved or regressed.

```bash
python scripts/run_backtest.py --output-dir outputs/backtest_results/
```

The backtesting suite runs all six analytical modules (BT-01 through BT-06) and produces the following outputs in `outputs/backtest_results/`: a condition-level grading report, pipeline attribution tables showing which pipeline stage caused each error, cluster completeness heat maps, temporal validity classifications, and XAI audit reports for sample codes.

---

## Project Structure at a Glance

All folder and file locations are defined in `docs/repository/REPO-00_repository_structure.md`. The table below summarises the top-level directories for quick orientation.

`config/` holds the three YAML configuration files: `agent_config.yaml` (LLM settings), `data_source_versions.yaml` (version tracking for all data sources), and `scoring_weights.yaml` (the composite scoring weights tuned in Phase 4). These are the only files you should edit when adjusting system behaviour without changing code.

`data/` is the data lake. `data/raw/` holds original, unmodified source files — never edit these. `data/gold_standard/` holds the NICE example code lists that serve as ground truth. `data/processed/` holds cleaned and merged outputs. `data/vectorstore/` holds the ChromaDB persistent vector database. `data/backtest/` holds backtesting run logs and attribution reports.

`src/` is the project's Python library. It is structured into seven subpackages: `ingestion/` (data loading), `features/` (feature engineering), `models/` (ML models), `tools/` (agent tool functions), `agent/` (LangChain/LangGraph orchestration), `audit/` (provenance and PROV-O), and `backtest/` (the full BT-01 through BT-06 suite).

`scripts/` holds thin executable wrappers that call `src/` functions. Never put complex logic in a script — it should live in `src/` and be tested there.

`notebooks/` holds Jupyter notebooks organised by project phase. Each phase directory corresponds directly to a phase in `docs/planning/06_project_plan_and_features.md`.

`outputs/` holds all generated artefacts: plots, reports, provenance records, and backtesting results. This is the directory you share with NICE stakeholders.

`tests/` holds pytest unit and integration tests. Run `pytest tests/` before committing any changes to `src/`.

`docs/` holds all seventeen planning and reference documents, organised into three subdirectories: `planning/` (the main 00–10 series), `backtesting/` (the BT-00–BT-06 series), and `repository/` (the REPO-00–REPO-03 series).

---

## Key Design Decisions

Several design decisions are baked into this codebase that might not be obvious without the context of the planning documents. Understanding them will prevent you from accidentally working against them.

The embedding model name is stored in `config/data_source_versions.yaml` and in the ChromaDB collection name (`snomed_codes_pubmedbert` where `pubmedbert` reflects the model). This is intentional — if you change the embedding model, you must also rebuild the vector store, and naming the store after the model makes it immediately obvious which model was used to build it. Never use a generic name like `snomed_codes` for the ChromaDB collection.

The `deprecated_flag` in the Master Lookup Table is a soft flag, not a hard exclusion. Codes with `deprecated_flag = True` are moved to REVIEW tier but are still included in the agent's output. This is by design: clinical governance requires a human to confirm a deprecation, not an automated system to silently suppress a code.

Temperature is set to zero in `agent_config.yaml`. This is a hard requirement for this application, not a preference. Reproducibility is a governance requirement — the same query run twice must produce the same output. Do not increase temperature without a specific documented reason and sign-off from the team.

The `source_count` field in the MLT counts a maximum of four sources. This number is deliberately capped: adding more sources without re-tuning the scoring weights would silently change the composite score for thousands of codes. Whenever a new authoritative source is added (a new Reference Set, for example), the scoring weights in `config/scoring_weights.yaml` should be re-evaluated using the backtesting suite.

---

## Documentation Suite Summary

The full documentation suite contains twenty documents across three series, all in `docs/`. Every document is self-contained but cross-references others, so you can follow threads between topics. If you are new to the project, the recommended entry points are `00_project_brief_plain_english.md` (for context), `REPO-00_repository_structure.md` (for where things live), and `BT-00_backtesting_master_overview.md` (for how we evaluate the system).

---

## Contributing

All reusable logic belongs in `src/`. All executable entry points belong in `scripts/`. All exploratory work belongs in `notebooks/`. All tests belong in `tests/`. Before opening a pull request, run `pytest tests/unit/` and confirm that all existing tests pass. If you add a new agent tool, add a corresponding unit test in `tests/unit/test_tools.py`. If you change the MLT schema, update `docs/repository/REPO-02_master_lookup_table.md` before merging.

---

## Contact and Governance

This project is produced as part of a collaborative data science programme with NICE. All outputs — agent-generated code lists, provenance records, backtesting reports — are provisional and must undergo expert clinical review before use in any NICE analysis. Do not use system-generated code lists in production without explicit sign-off from a qualified clinical informatician.
