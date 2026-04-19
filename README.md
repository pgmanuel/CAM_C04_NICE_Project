# NICE Clinical Code Recommendation MVP

## Overview

This project is a working MVP for an AI-assisted clinical code recommendation pipeline. It takes a plain-English cohort query, retrieves relevant SNOMED candidates, groups them into analyst-facing buckets, and writes an audit trail for traceability.

It is designed as a **human-in-the-loop analyst support tool**, not an autonomous coding system.

## What it does

Given a query such as:

```text
Obesity, diabetes mellitus, and hypertension
```

the pipeline:

- decomposes the query into structured search components
- runs hybrid retrieval using BM25 and vector search
- enriches candidates with hierarchy metadata
- fuses candidates across search jobs
- applies relevance-gate and cross-encoder reranking
- applies the decision engine to group candidates into review buckets
- writes a JSON audit file for each run

Evaluation is available as an optional path and is not run by the production
runtime by default.

## Output buckets

The current output is grouped into:

- `include_candidates` — broad anchor concepts for first-pass review
- `review_candidates` — narrower but plausible concepts for analyst review
- `specific_variants` — lower-priority specialised related concepts
- `suppressed_candidates` — retained for traceability but hidden from first-pass review

## Design principles

This MVP is built to be:

- **explainable** — evidence fields are preserved
- **auditable** — each run writes a JSON audit artifact
- **deterministic** — grouping and confidence logic are rule-based
- **human-in-the-loop** — final judgement remains with the analyst

## Project structure

```text
backend/
├── main.py
├── config.py
├── user_settings.example.py
├── query_planning.py
├── retrieval_engine.py
├── hierarchy_enricher.py
├── fusion_engine.py
├── gate_reranker.py
├── ce_reranker.py
├── decision_engine.py
├── output_formatter.py
├── audit_logger.py
├── evaluation.py
├── eval_runner.py
├── requirements.txt
├── tests/
├── audit/
└── README.md
```

Current source of truth:

- `main.py` is the clean production runtime.
- `query_planning.py` owns decomposition and condition-specific search jobs.
- `retrieval_engine.py` owns hybrid BM25/vector retrieval policy.
- `fusion_engine.py` owns cross-job candidate fusion.
- `gate_reranker.py` owns the lightweight relevance gate.
- `ce_reranker.py` owns cross-encoder reranking.
- `decision_engine.py` owns final bucket assignment and presentation scoring.
- `audit_logger.py` owns run traces and audit JSON output.
- `evaluation.py` and `eval_runner.py` provide opt-in evaluation/RAGAS utilities outside the production path.

## Resource loading

The production runtime reads two required source files and builds reusable local
runtime assets from them:

- `snomed_master_v4.csv`
- `snomed_parent_child_edges_clean.csv`
- Chroma DB persistence directory
- embedding model cache directory
- audit output directory

The required source dataset files can be downloaded from this shared Drive
folder:

<https://drive.google.com/drive/folders/1y0JYAQf5lBf4HpyMZcW46ffDQu6KnaSg?usp=sharing>

By default, paths are resolved relative to this `backend/` directory:

- `../snomed_master_v4.csv`
- `../snomed_parent_child_edges_clean.csv`
- `../chroma_db_v4`
- `../embeddings`
- `audit/`

You can also override these with environment variables:

```bash
export NICE_SNOMED_PATH=/absolute/path/to/snomed_master_v4.csv
export NICE_EDGE_PATH=/absolute/path/to/snomed_parent_child_edges_clean.csv
export NICE_CHROMA_DIR=/absolute/path/to/chroma_db_v4
export NICE_EMBEDDINGS_DIR=/absolute/path/to/embeddings
export NICE_AUDIT_DIR=/absolute/path/to/audit
```

Behaviour:

- required source CSV files are validated at startup with a clear error if missing
- runtime directories for Chroma, embeddings, and audit output are created automatically
- if an existing Chroma collection is found and has records, it is reused
- if the Chroma collection is missing or empty, it is rebuilt automatically from `snomed_master_v4.csv`
- if the embedding model is missing, `sentence-transformers` downloads and caches it in the embeddings directory
- if `NICE_REBUILD_CHROMA=true`, the configured Chroma collection is deleted and rebuilt on startup

For a forced Chroma rebuild:

```bash
export NICE_REBUILD_CHROMA=true
export NICE_CHROMA_REBUILD_BATCH_SIZE=1000
./.venv/bin/python main.py
```

Unset the rebuild flag afterwards so normal runs reuse the persisted index:

```bash
unset NICE_REBUILD_CHROMA
```

## Configuration Setup

To customise paths, models, or testing queries without modifying core files:

1. Copy `user_settings.example.py` to `user_settings.py`.
2. Edit `user_settings.py` to set your local customisations.
   *(Note: `user_settings.py` is safely ignored by git).*

**⚠️ CRITICAL**: Do NOT store API keys in `user_settings.py`. Any secret credentials must remain purely within your environment variables.

## Environment setup

Python 3.11 is the tested runtime for this repository.

From the `backend/` directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

First run requirements:

- local Ollama is optional for query decomposition; if unavailable, the planner uses fallback decomposition
- Hugging Face access is required the first time embedding and reranking models are downloaded
- the Chroma rebuild can take a while because it embeds all SNOMED rows

## How to run

From the project root:

```bash
python main.py
```

Or with the virtual environment:

```bash
./.venv/bin/python main.py
```

To run optional evaluation without changing production runtime:

```bash
./.venv/bin/python eval_runner.py "Obesity, diabetes mellitus, and hypertension" --custom-eval
```

Evaluation uses Nebius for judge-model calls. Set `NEBIUS_API_KEY` only when you
want to run `eval_runner.py` with `--custom-eval`, `--ragas`, or `--all-evals`:

```bash
export NEBIUS_API_KEY=...
```

## Regression checks

To run the lightweight regression checks:

```bash
./.venv/bin/python tests/run_regression_checks.py
```

Or, run the programmatic baseline suite via pytest:

```bash
./.venv/bin/python -m pytest tests/test_regression_baseline.py -v
```

## Audit output

Each run writes a JSON audit file into:

```text
audit/
```

The audit captures the key pipeline steps and config snapshot used for that run.

## Troubleshooting

If startup fails with a missing source-file error, set:

```bash
export NICE_SNOMED_PATH=/absolute/path/to/snomed_master_v4.csv
export NICE_EDGE_PATH=/absolute/path/to/snomed_parent_child_edges_clean.csv
```

If retrieval returns empty results on a fresh machine, force a Chroma rebuild:

```bash
export NICE_REBUILD_CHROMA=true
./.venv/bin/python main.py
unset NICE_REBUILD_CHROMA
```

If model download fails, confirm internet access to Hugging Face or pre-populate
the directory set by `NICE_EMBEDDINGS_DIR`.

If optional evaluation reports missing Nebius credentials, set `NEBIUS_API_KEY`.
Do not put API keys in `user_settings.py` or commit them to audit artifacts.

## Current limitations

This is still an MVP.

- not clinically validated
- not production-ready
- requires human review
- retrieval quality depends on local data and cache state
- grouping and confidence are heuristic, not final clinical logic

## Positioning

This project should be described as:

- an analyst-support pipeline
- explainable and auditable by design
- a practical candidate-generation MVP

It should **not** be described as:

- a clinically validated system
- a production deployment
- an autonomous clinical coding agent
- a replacement for analyst judgement
