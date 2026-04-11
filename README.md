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
- merges and ranks evidence across search jobs
- groups candidates into review buckets
- writes a JSON audit file for each run

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
CAM_C04_NICE_Project/
├── main.py
├── config.py
├── query_planning.py
├── retrieval_engine.py
├── fusion_engine.py
├── ranking_engine.py
├── scoring_rules.py
├── decision_engine.py
├── output_formatter.py
├── audit_logger.py
├── tests/
├── audit/
└── README.md
```

## Resource loading
The project supports reusable local resources for:

- **Chroma DB**
- **embedding model cache**

By default, it looks for sibling folders outside the project root:

- `../chroma_db`
- `../embeddings`

You can also override these with environment variables:

```bash
export INTEGRATED_AGENT_CHROMA_DIR=/absolute/path/to/chroma_db
export INTEGRATED_AGENT_EMBEDDINGS_DIR=/absolute/path/to/embeddings
```

Behaviour:

- if an existing Chroma DB is found, it is reused
- if a cached embedding model is found, it is reused
- if resources are missing, they are created or downloaded automatically

## Configuration Setup
To customise paths, models, or testing queries without modifying core files:
1. Copy `user_settings.example.py` to `user_settings.py`.
2. Edit `user_settings.py` to set your local customisations.
   *(Note: `user_settings.py` is safely ignored by git).*

**⚠️ CRITICAL**: Do NOT store API keys in `user_settings.py`. Any secret credentials must remain purely within your environment variables.

## How to run
From the project root:

```bash
python main.py
```

Or with the virtual environment:

```bash
./.venv/bin/python main.py
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
