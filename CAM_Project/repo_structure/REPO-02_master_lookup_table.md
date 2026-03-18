# REPO-02 — The Master Lookup Table: Schema, Construction, and Downstream Usage

> **Why this table is the single most important artefact in the project.** Every analytical technique in this project — correlation analysis, KMeans clustering, the supervised classifier, the agent's scoring function, the backtesting suite — reads from the Master Lookup Table. It is not a temporary intermediate file; it is the persistent, version-controlled record of everything known about every SNOMED code, assembled from all data sources, updated as new pipeline stages complete. This document is its complete specification.

---

## Part 1 — Why a Centralised Table Matters

Without a Master Lookup Table, the project faces a fragmentation problem. The QOF parser produces one DataFrame. The usage frequency downloader produces another. The NICE gold-standard loader produces a third. The graph feature extractor produces a fourth. If each notebook or script loads and joins these sources independently, you get four things that are all problematic: inconsistent join logic between team members, column name collisions that produce silent `_x` and `_y` suffixes, no single place to check what information is available for a given code, and no way to version-control the data together as a coherent snapshot.

The MLT solves all of this. It is built once — by `scripts/build_master_lookup_table.py`, documented in REPO-01 — and then updated incrementally as each pipeline stage completes. Phase 2 adds graph features. Phase 3 adds cluster IDs. Phase 5 adds deprecation flags. Every downstream script simply reads one file, selects the columns it needs, and works from a consistent, validated, version-stamped base.

---

## Part 2 — The Complete Field Specification

The table below documents every field in `data/processed/merged/master_lookup_table.csv`, explaining where it comes from, what it means, and which downstream components use it. The original suggestion proposed four fields; the full schema has twenty-four. The extra twenty fields are not gold-plating — each one is required by at least one planning or backtesting document.

### Identity and Description Fields

**`snomed_code`** is the primary key — the SNOMED CT concept identifier, stored as a string (not an integer, because leading zeros are meaningful in some older code formats and integer conversion would silently corrupt them). Every join in the project uses this field. It is populated from the QOF parser, the OpenCodeCounts download, and the NICE gold-standard files, merged via outer join so that codes from any source are represented even if absent from others.

**`description`** is the preferred term for the code — the official SNOMED CT description as it appears in the UK Edition. This is the text that gets embedded to produce the semantic search vectors. It comes from the OpenCodeCounts dataset (which includes descriptions) and is supplemented from the NHS Data Dictionary for codes where OpenCodeCounts does not carry the description. Do not use the QOF `Term` column as the primary description source because QOF occasionally uses abbreviated or modified versions of the preferred term.

### Source Provenance Fields

**`in_qof`** is a boolean indicating whether this code is mandated by any indicator in QOF Business Rules v49 2024-25. This is the highest-authority flag in the entire system — it determines the initial confidence tier in the agent's output and carries the largest positive weight in the composite scoring function. Set to `True` by the QOF parser; defaults to `False` for all other codes.

**`qof_indicators`** is a pipe-separated string of all QOF indicator IDs that reference this code (e.g., `DM012|DM014|OB001`). This field is what populates the "Source: QOF Indicator X" line in the PROV-O provenance records. Storing it as a pipe-separated string rather than a list means it can be written to CSV without serialisation issues, while still being easily split when needed.

**`qof_clusters`** is a pipe-separated string of all QOF cluster IDs referencing this code (e.g., `HYPERTENSION_DAT|BPCOMP_QUAL`). QOF clusters are groupings of related indicators used in the business rules. This field is distinct from `qof_indicators` — a single code can appear in multiple clusters across different indicators.

**`in_nice_gold`** is a boolean indicating whether this code appears in any of the DAAR_2025_004 gold-standard files. This is the training label for the supervised classifier — Phase 4 uses this field as `y` (the target variable). It is also the primary comparison column in the backtesting suite.

**`nice_conditions`** is a pipe-separated string of all condition categories from the gold-standard files that include this code (e.g., `hypertension|obesity`). A code can appear in multiple gold-standard lists — the obesity and hypertension lists share several cardiovascular risk codes. This field captures that multi-membership rather than arbitrarily assigning a code to a single condition.

**`in_open_codelists`** is a boolean indicating whether this code appears in any of the OpenSAFELY OpenCodelists cached in `data/raw/open_codelists/api_cache/`. OpenCodelists are community-validated code lists produced by clinical informatics experts at the Bennett Institute. Codes validated there are more likely to be correct than codes found only through semantic search.

**`open_codelists_source`** is a pipe-separated string of the OpenCodelists codelist slugs that include this code. Like `qof_indicators`, this populates the source citation in provenance records.

### Usage Frequency Fields

**`usage_count`** is the annual occurrence count from the OpenCodeCounts dataset — the number of times this code was recorded in English primary care patient records in the most recent reporting year. This is the primary frequency metric because OpenCodeCounts covers over 62 million patient records, making it the most representative available source. May be zero for codes that are theoretically valid but never or rarely used.

**`nhs_digital_usage_count`** is the corresponding count from the NHS Digital SNOMED Code Usage in Primary Care publication. Both are kept separately because they come from different data sources, may differ slightly (due to different population coverage and data extraction methods), and should both be cited in provenance records when both are available.

**`log_usage`** is `log(1 + usage_count)`. It is a derived field, computed during MLT construction rather than in the feature engineering step, because it is used by nearly every downstream component — the correlation analysis, the supervised classifier, the composite scoring function, and the EDA notebooks. Computing it once here avoids repeated, potentially inconsistent implementations across the codebase. The `log1p()` transform is used rather than `log()` to handle zero-usage codes safely.

### Source Count and Validation Fields

**`source_count`** is an integer counting how many distinct authoritative sources reference this code — specifically: `in_qof` (1 if true), `in_nice_gold` (1 if true), `in_open_codelists` (1 if true), and whether the code appears in the NHS Digital SNOMED publication (1 if not null). The maximum is 4. This field carries the second-highest weight in the composite scoring function because it operationalises the principle that a code validated by multiple independent sources is more likely to be correct than one found in only one.

### Graph Feature Fields

These four fields are populated by `src/features/graph_features.py` during Phase 2 and default to `None` until that step runs. Their meaning is fully explained in `07_snomed_graph_architecture.md`.

**`in_degree`** is the number of parent concepts in the SNOMED polyhierarchy. A value of 1 means the code sits in a simple hierarchy. A value of 2 or more means it is a bridging concept — a child of multiple parent categories simultaneously. The polyhierarchy recall analysis in BT-02 specifically uses this field to stratify agent performance.

**`out_degree`** is the number of direct child concepts. A value of 0 means the code is a leaf node — the most specific available concept in that branch. A high value means the code is a broad category with many sub-types.

**`subtree_size`** is the total number of descendants (all levels, not just direct children). A large subtree indicates a broad concept; a subtree size of 0 means a leaf node. This is used as a proxy for hierarchical specificity in the structural depth analysis in BT-02.

**`degree_centrality`** is the NetworkX degree centrality score — a normalised measure of how many connections a code has relative to all other nodes in the graph. High-centrality codes are clinical "hubs" relevant across many conditions and are disproportionately important in comorbidity code lists.

### Temporal Analysis Fields

These two fields are populated by `src/models/timeseries.py` during Phase 5 and default to `False` / `"unknown"` until that step runs.

**`deprecated_flag`** is a boolean set to `True` when the time series analysis in Phase 5 detects a structural break in the code's usage time series followed by a sustained period of near-zero usage. The ruptures change-point detection algorithm produces this flag. Any code with this flag set to `True` is automatically moved to REVIEW tier by the agent's scoring function, regardless of its other scores. The temporal backtesting in BT-04 specifically tests whether agents correctly avoid recommending deprecated codes.

**`usage_trend`** is a string classification — one of `"growing"`, `"stable"`, or `"declining"` — produced by the Prophet trend component in Phase 5. Growing codes are increasingly mainstream; declining codes may be approaching deprecation. This field enriches the agent's rationale text ("usage trend indicates growing clinical adoption") and is checked during the drift monitoring workflow.

### Embedding and Clustering Fields

**`embedding_vector_id`** is a string pointer to the code's record in the ChromaDB vector store — specifically, the ID value used when the code's embedding was added to the collection. This allows the agent's hierarchy explorer and other tools to retrieve the pre-computed embedding for a specific code without re-embedding it on the fly, which would be slow.

**`cluster_id`** is an integer cluster assignment from the KMeans model trained in Phase 3. This field is central to the multimorbidity clustering analysis in BT-03 — it is how the backtesting suite checks whether the agent's code list draws from the right cluster families for the target condition. It defaults to `None` until the clustering step runs.

### Version Tracking Fields

These three fields record exactly which version of each data source was used when the MLT row was last updated. They feed directly into the PROV-O provenance records, satisfying the auditability requirement by making every code recommendation traceable not just to a source but to a specific version of that source.

**`data_source_version_qof`** stores the QOF version string, e.g., `"v49_2024-25"`.

**`data_source_version_usage`** stores the NHS Digital usage publication version, e.g., `"2024-25"`.

**`data_source_version_occ`** stores the OpenCodeCounts version, e.g., `"2025-Q3"`.

---

## Part 3 — Update Schedule: When to Rebuild the MLT

The MLT is not a one-time artefact. It should be rebuilt whenever any of the following events occur, because stale data in the MLT will produce wrong agent recommendations and will not be caught until backtesting.

When the QOF Business Rules are updated (typically each April), re-run `scripts/parse_qof.py` followed by `scripts/build_master_lookup_table.py`. Update `config/data_source_versions.yaml` first so the new version is stamped into every row.

When a new NHS Digital SNOMED usage publication is released (also typically annual), re-run `scripts/download_snomed_usage.py` followed by `scripts/build_master_lookup_table.py`.

When new SNOMED CT codes are added in a UK Edition release, re-run the full ingestion pipeline — download, parse, MLT build, and then `scripts/build_vectorstore.py` to add the new codes' embeddings to the vector store.

When the Phase 3 clustering step is re-run with a different K or embedding model, the `cluster_id` field needs to be updated. Re-run `src/models/clustering.py` and merge the new cluster assignments back into the MLT using the MLT update pattern below.

---

## Part 4 — The MLT Update Pattern

Later pipeline stages (Phase 2, Phase 3, Phase 5) produce new fields that need to be merged into the MLT. The standard pattern for doing this safely — without risking column name collisions or accidental overwrites — is shown below. Every `src/` module that updates the MLT should follow this exact pattern.

```python
import pandas as pd
from pathlib import Path
from loguru import logger

MLT_PATH = Path("data/processed/merged/master_lookup_table.csv")

def update_mlt_with_new_fields(new_fields_df: pd.DataFrame,
                                join_key: str = "snomed_code",
                                update_description: str = "new fields") -> None:
    """
    Safely merge new columns into the Master Lookup Table.
    
    new_fields_df must contain the join_key column plus any new columns.
    If any of the new columns already exist in the MLT (e.g., from a previous
    run of the same pipeline step), the existing values are overwritten —
    this is correct behaviour for a re-run, not a data corruption risk.
    
    The MLT is read from disk, updated in memory, and written back atomically
    (written to a temp file first, then renamed) to prevent partial writes
    from corrupting the MLT if the process is interrupted.
    """
    mlt = pd.read_csv(MLT_PATH, dtype={join_key: str})
    new_fields_df[join_key] = new_fields_df[join_key].astype(str)
    
    # Drop any columns in new_fields_df that already exist in the MLT
    # so the merge doesn't create _x/_y duplicates
    overlap_cols = [c for c in new_fields_df.columns
                    if c in mlt.columns and c != join_key]
    if overlap_cols:
        logger.info(f"Overwriting existing columns: {overlap_cols}")
        mlt = mlt.drop(columns=overlap_cols)
    
    updated_mlt = mlt.merge(new_fields_df, on=join_key, how="left")
    
    # Atomic write: write to temp, then rename
    temp_path = MLT_PATH.with_suffix(".tmp.csv")
    updated_mlt.to_csv(temp_path, index=False)
    temp_path.rename(MLT_PATH)
    
    logger.success(f"MLT updated with {update_description}: "
                   f"{len(updated_mlt)} rows, {len(updated_mlt.columns)} columns")
```

---

*This document is the authoritative schema reference. Any discrepancy between this document and the column names in actual code should be resolved in favour of this document — update the code, not the spec.*
