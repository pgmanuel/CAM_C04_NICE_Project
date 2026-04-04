# NICE Clinical Code Recommendation MVP

## Project Overview
This project is a working MVP for a NICE-style clinical code recommendation pipeline. It takes a plain-English cohort query such as `Obesity, diabetes mellitus, and hypertension` and returns grouped SNOMED code candidates for human review.

The system is designed as an analyst-assist tool, not an autonomous coding system. Its purpose is to help a reviewer get to sensible candidate concepts faster while preserving enough evidence and traceability to explain why those concepts were surfaced.

## Current Goal
The current goal is to support a human-in-the-loop code list workflow:

- interpret a cohort query into clinically useful search components
- retrieve a broad but relevant candidate pool
- rank and group candidates into analyst-facing buckets
- attach evidence and confidence information
- write an audit trail for each run

Manual review is still required.

## High-Level Pipeline
The current end-to-end flow is:

1. Query normalization
2. Structured query decomposition
3. Search-job planning
4. Hybrid retrieval
5. Cross-query fusion
6. Deterministic decisioning and grouping
7. Deterministic output formatting
8. Audit trail export

In plain English:

- the user writes a cohort query
- the pipeline identifies a primary condition and any secondary conditions
- multiple search jobs are generated from that structure
- retrieval combines semantic search and BM25 lexical search
- evidence is merged across search jobs
- deterministic rules assign confidence and candidate role
- the output is grouped for analyst review rather than flattened into one long ranked list
- a JSON audit file is written for traceability

## Design Principles
The MVP follows a few practical rules:

- deterministic logic is used for retrieval, ranking, grouping, and confidence assignment
- evidence fields such as `in_qof`, `in_opencodelists`, and `usage_count_nhs` are preserved and exposed
- the grouped output is designed to match analyst review workflow better than a single flat leaderboard
- the audit trail is treated as a core feature, not an add-on
- the system is a candidate-generation workbench, not a final decision-maker

This project is not clinically validated, not production-ready, and should not be used for live patient care.

## Codebase Structure
The current MVP is split into focused modules.

### Main pipeline
- [main.py](main.py)
  Main orchestration layer. Builds config, resolves shared paths, runs the pipeline stages, and writes the final audit artifact.

### Shared paths
- [project_paths.py](project_paths.py)
  Central path resolver for shared project assets such as `snomed_master_v3.csv`, `chroma_db/`, `audit/`, and `tests/regression_cases.json`.

### Query planning
- [query_planning.py](query_planning.py)
  Handles query normalization, structured decomposition, and generation of multiple retrieval jobs from the decomposed query.

### Retrieval and resource loading
- [retrieval_engine.py](retrieval_engine.py)
  Loads SNOMED data, initializes BM25, connects to Chroma, loads the embedding model, and runs hybrid retrieval per search job.

### Ranking and grouping
- [ranking_engine.py](ranking_engine.py)
  Thin compatibility layer that re-exports the ranking components used by the runner.
- [fusion_engine.py](fusion_engine.py)
  Handles cross-query candidate fusion and retrieval aggregation.
- [scoring_rules.py](scoring_rules.py)
  Holds ranking helpers, suppression rules, and scoring components used by deterministic decisioning.
- [decision_engine.py](decision_engine.py)
  Applies confidence assignment, candidate-role assignment, bucket placement, and final grouped decisions.

### Output formatting
- [output_formatter.py](output_formatter.py)
  Shapes the final grouped output and adds deterministic rationale text. The LLM formatter is intentionally disabled at the current stage.

### Audit trail
- [audit_logger.py](audit_logger.py)
  Captures pipeline traces and writes one JSON audit file per run into `audit/`.

### Supporting notebooks and earlier prototypes
- `Pod 1`, `Pod 2`, and `Pod 3`
  Earlier notebook-based work that informed retrieval, query decomposition, and audit/provenance ideas.

## Output Schema
The current output is intentionally grouped into analyst-facing buckets.

### `include_candidates`
Broad anchor concepts most suitable for first-pass analyst review.

Typical examples:
- `Obesity (disorder)`
- `Diabetes mellitus (disorder)`
- `Essential hypertension (disorder)`

### `review_candidates`
Narrower but still plausible concepts worth analyst consideration.

### `specific_variants`
More specialised related concepts kept for lower-priority review depth.

### `suppressed_candidates`
Concepts intentionally kept out of first-pass review but retained for traceability.

These usually include query-misaligned or context-specific concepts such as inactive states or irrelevant contextual variants.

## Audit Trail
Every pipeline run writes a JSON audit artifact under `audit/`.

The audit currently captures:

- config snapshot for the run
- normalized query
- structured query
- generated search jobs
- retrieval summaries
- top candidates per search job
- fused candidate preview
- fused candidates before final decisioning
- grouped final decisions
- explanation or formatter mode

This matters because it allows a reviewer to inspect what the system did at each stage instead of treating the final output as a black box.

## Reproducibility
The current MVP is set up to be rerun more consistently than earlier notebook-only versions.

- `requirements.txt`
  Pins the main Python dependencies used by the pipeline.
- [project_paths.py](project_paths.py)
  Resolves one project root and derives shared paths from it.
- `config_snapshot`
  Is written into every audit JSON so runs can be compared against the exact data paths, model settings, bucket caps, suppression rules, and retrieval weights that produced them.
- [regression_cases.json](tests/regression_cases.json)
  Freezes a small set of property-based regression expectations for the current MVP.
- [test_regression_baseline.py](tests/test_regression_baseline.py)
  Provides a minimal pytest baseline that checks anchor-concept and clutter-suppression behavior without requiring exact full ranking matches.
- [run_regression_checks.py](tests/run_regression_checks.py)
  Provides a tiny pass/fail regression runner that uses `regression_cases.json` directly.

## Current Demo Behavior
For the demo query `Obesity, diabetes mellitus, and hypertension`, the current system decomposes the query into:

- primary condition: `Obesity`
- secondary conditions: `Diabetes Mellitus`, `Hypertension`

The grouped output is intended to behave roughly like this:

- `include_candidates`
  Broad anchor concepts such as obesity, diabetes mellitus, and essential hypertension
- `review_candidates`
  A short list of narrower but still plausible concepts
- `specific_variants`
  More specialised related concepts, especially hypertension variants
- `suppressed_candidates`
  Misaligned or context-heavy concepts retained only for traceability

This grouped output is intentional. It reflects a review workflow more closely than a single flat top-k ranking.

## Current Limitations
This is still an MVP and has important limitations.

- it is not clinically validated
- it is not production-ready
- manual review is still required
- the ranking and grouping are practical heuristics, not perfect clinical logic
- some variant selection is still being tuned, especially around obesity refinements and specialised hypertension concepts
- the formatter is currently deterministic by design; LLM-based rationale generation is disabled
- retrieval quality still depends on local data availability, Chroma state, and embedding model availability

## How to Run
The current entry point is:

- [main.py](main.py)

Example:

```bash
python CAM_Project/main.py
```

To run the lightweight regression checks:

```bash
python CAM_Project/tests/run_regression_checks.py
```

The script currently uses this demo query by default:

```text
Obesity, diabetes mellitus, and hypertension
```

The pipeline expects shared project assets such as:

- `snomed_master_v3.csv`
- `chroma_db/`

It writes audit artifacts to:

- `audit/`

## Practical Positioning
This MVP should be described as:

- a working analyst-support pipeline
- explainable and auditable by design
- useful for structured candidate generation
- suitable for demonstrating an end-to-end Assignment 2 workflow

It should not be described as:

- clinically validated
- production-ready
- an autonomous clinical coding agent
- a replacement for analyst judgement
