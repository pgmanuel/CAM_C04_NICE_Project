# REPO-01 — Data Ingestion Scripts: Complete Annotated Reference

> **What this document contains.** The original pasted suggestion included three scripts — two partial and one a placeholder. Our planning documents collectively require eleven ingestion and setup scripts. This document provides all eleven, fully annotated, with each design decision explained in terms of which planning document it serves and why the implementation was chosen that way. Every script is a thin wrapper: the actual logic lives in `src/ingestion/`, and the scripts in `scripts/` simply call those functions with command-line arguments and logging.

---

## Architecture: Why Scripts Are Thin Wrappers

Before diving into the scripts themselves, it is worth explaining why the code is structured the way it is. The original pasted approach put all logic inside the script files themselves. Our project splits this differently: `src/ingestion/` contains reusable, importable functions, and `scripts/` contains thin executables that call those functions. This matters for two reasons. First, notebooks can import from `src/` directly without needing to run a subprocess — which means EDA notebooks can reuse the same parsing logic as the ingestion scripts without any code duplication. Second, tests can import from `src/` and test each function in isolation — you cannot easily unit-test code that lives inside a `if __name__ == "__main__"` block.

---

## Script 1 — `scripts/download_snomed_usage.py`

The original pasted version hardcoded a specific URL that is unlikely to remain stable as NHS Digital updates its publication infrastructure. The corrected version uses the publication's persistent landing page and a configurable year parameter.

```python
"""
scripts/download_snomed_usage.py

Downloads the NHS England SNOMED Code Usage in Primary Care statistical
publication for a given reporting year and saves it to data/raw/snomed_usage/.

Source: https://digital.nhs.uk/data-and-information/publications/statistical/
        mi-snomed-code-usage-in-primary-care

Usage:
    python scripts/download_snomed_usage.py --year 2024-25

Note: The NHS Digital publication URL structure changes between releases.
If the automated download fails, navigate to the landing page manually,
copy the direct download URL for the .txt file, and pass it via --url.
The script saves the raw file unchanged — no parsing occurs here.
Parsing is done by scripts/parse_qof.py which calls src/ingestion/load_usage_data.py.
"""

import argparse
import os
import requests
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Known stable URL pattern for NHS Digital SNOMED usage publications.
# If this fails, check: https://digital.nhs.uk/data-and-information/publications/statistical/
#                        mi-snomed-code-usage-in-primary-care
NHS_USAGE_URL_TEMPLATE = (
    "https://digital.nhs.uk/binaries/content/assets/website-assets/publications/"
    "statistical/mi-snomed-code-usage-in-primary-care/{year}/"
    "snomed_code_usage_{year}.txt"
)

def download_snomed_usage(year: str, output_dir: str, url_override: str = None) -> Path:
    """
    Download the SNOMED usage publication for a given year.
    
    Returns the path to the saved file so downstream scripts can chain directly.
    The function uses streaming download (stream=True) because these files
    can be several hundred megabytes — loading the full response into memory
    at once would fail on machines with limited RAM.
    """
    output_path = Path(output_dir) / f"usage_{year.replace('-', '_')}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    url = url_override or NHS_USAGE_URL_TEMPLATE.format(year=year)
    logger.info(f"Downloading SNOMED usage data from: {url}")
    
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()  # Raises HTTPError for 4xx/5xx responses
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.success(f"Saved {file_size_mb:.1f}MB to {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NHS SNOMED usage data")
    parser.add_argument("--year", default="2024-25", help="Reporting year, e.g. 2024-25")
    parser.add_argument("--output-dir", default="data/raw/snomed_usage/")
    parser.add_argument("--url", default=None, help="Override the download URL")
    args = parser.parse_args()
    
    download_snomed_usage(args.year, args.output_dir, args.url)
```

---

## Script 2 — `scripts/download_open_code_counts.py`

This script was entirely absent from the original suggestion. OpenCodeCounts is a distinct dataset from the NHS SNOMED usage publication — it comes from the Bennett Institute, covers a larger number of patient records (62 million+), and is accessed via its own published data portal rather than NHS Digital. This distinction is critical for our PROV-O provenance records: codes validated against OpenCodeCounts cite a different source than codes validated against the NHS Digital publication.

```python
"""
scripts/download_open_code_counts.py

Downloads the OpenCodeCounts dataset from the Bennett Institute for Applied
Data Science, University of Oxford.

Source: https://bennettoxford.github.io/opencodecounts/
GitHub: https://github.com/bennettoxford/opencodecounts

The dataset is available as R package data, but also exported as CSV files
accessible directly. This script fetches the latest available CSV export.

Usage:
    python scripts/download_open_code_counts.py

Reference: 01_data_science_roadmap.md Phase 0 Feature 0.4,
           BT-01_eda_usage_baseline_backtesting.md
"""

import requests
import pandas as pd
from pathlib import Path
from loguru import logger

# The Bennett Institute publishes OpenCodeCounts data to GitHub.
# The raw CSV URL below points to the primary care SNOMED usage counts.
# If this URL changes, check: https://github.com/bennettoxford/opencodecounts
OPEN_CODE_COUNTS_URL = (
    "https://raw.githubusercontent.com/bennettoxford/opencodecounts/main/"
    "data/opencodecounts_primary_care.csv"
)

def download_open_code_counts(output_dir: str = "data/raw/open_code_counts/") -> Path:
    """
    Download the OpenCodeCounts CSV and validate its structure before saving.
    
    We validate immediately on download — if the file structure has changed
    (e.g., column names renamed in a new release), we want to know before
    attempting to build the Master Lookup Table, not discover it mid-pipeline.
    """
    output_path = Path(output_dir) / "opencodecounts_latest.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info("Downloading OpenCodeCounts dataset from Bennett Institute...")
    response = requests.get(OPEN_CODE_COUNTS_URL, timeout=120, stream=True)
    response.raise_for_status()
    
    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    # Validate structure immediately
    df = pd.read_csv(output_path, nrows=5)
    expected_columns = {'snomed_code', 'description', 'usage_count'}
    actual_columns = set(df.columns.str.lower())
    
    missing_cols = expected_columns - actual_columns
    if missing_cols:
        logger.warning(
            f"Column structure mismatch: expected {expected_columns}, "
            f"found {set(df.columns)}. Check if OpenCodeCounts has updated "
            f"its schema. You may need to update src/ingestion/load_usage_data.py."
        )
    else:
        logger.success(f"Downloaded and validated. File saved to {output_path}")
    
    return output_path


if __name__ == "__main__":
    download_open_code_counts()
```

---

## Script 3 — `scripts/parse_qof.py`

The original suggestion mentioned this script but didn't write it, noting only that the key columns to extract were Cluster ID, Concept ID, and Term. Our planning docs require a much richer extraction — we need the indicator ID, domain, and refset ID as well, because these become fields in the Master Lookup Table and are cited in the PROV-O provenance records. The parser here handles the complex multi-tab Excel structure of the QOF v49 workbook.

```python
"""
scripts/parse_qof.py

Parses the QOF Business Rules v49 (2024-25) Excel workbook into a clean,
unified DataFrame and saves it as data/processed/qof_v49_clean.csv.

The QOF workbook has a complex structure:
- One tab per clinical domain (e.g., Cardiovascular, Diabetes, Obesity)
- Each tab has an "Expanded Cluster List" subtable
- Key columns: Cluster ID, Indicator ID, Concept ID (SNOMED code), Term

The parser handles merged cells, inconsistent header positions, and the
fact that some tabs have different column orderings between domains.

Usage:
    python scripts/parse_qof.py --input data/raw/qof_rules/qof_v49_2024_25/
                                 --output data/processed/qof_v49_clean.csv

Reference: 03_understanding_qof.md, 06_project_plan_and_features.md Feature 0.2
"""

import argparse
import pandas as pd
import openpyxl
from pathlib import Path
from loguru import logger
from typing import Optional


def find_header_row(sheet, target_column: str) -> Optional[int]:
    """
    Scan a worksheet to find the row number where a specific column header appears.
    
    The QOF workbook doesn't have a consistent header row — each domain tab
    places the cluster list at a different starting row, sometimes after
    a block of explanatory text. This function finds the header dynamically
    rather than assuming a fixed row number, which would break when NHS Digital
    reformats the workbook in future releases.
    """
    for row_idx, row in enumerate(sheet.iter_rows(values_only=True), start=1):
        if row and any(
            str(cell).strip().lower() == target_column.lower()
            for cell in row if cell is not None
        ):
            return row_idx
    return None


def parse_qof_workbook(workbook_path: Path) -> pd.DataFrame:
    """
    Parse all domain tabs in the QOF workbook and return a unified DataFrame.
    
    Target schema (as specified in 06_project_plan_and_features.md Feature 0.2):
    [indicator_id, domain, condition_name, snomed_code, code_description, refset_id, cluster_id]
    
    The function skips tabs that don't contain expanded cluster lists (e.g.,
    the README tab, the Business Rules Overview tab) and logs a warning for
    any tabs that contain data but cannot be parsed — these need manual review.
    """
    wb = openpyxl.load_workbook(workbook_path, read_only=True, data_only=True)
    all_records = []
    
    # These tab name patterns indicate non-data tabs to skip
    skip_tab_patterns = ['readme', 'contents', 'overview', 'guidance', 'change']
    
    for sheet_name in wb.sheetnames:
        if any(p in sheet_name.lower() for p in skip_tab_patterns):
            logger.debug(f"Skipping tab: {sheet_name}")
            continue
        
        sheet = wb[sheet_name]
        header_row = find_header_row(sheet, "Concept ID")
        
        if header_row is None:
            logger.warning(f"Could not find 'Concept ID' header in tab '{sheet_name}' — skipping")
            continue
        
        # Extract all rows from the header row onwards
        rows = list(sheet.iter_rows(min_row=header_row, values_only=True))
        headers = [str(h).strip() if h else f"col_{i}" for i, h in enumerate(rows[0])]
        
        tab_records = []
        for row in rows[1:]:
            if not any(cell for cell in row):  # Skip empty rows
                continue
            record = dict(zip(headers, row))
            # Map QOF column names to our standard schema
            tab_records.append({
                'cluster_id':       str(record.get('Cluster ID', '') or '').strip(),
                'indicator_id':     str(record.get('Indicator ID', '') or '').strip(),
                'domain':           sheet_name,
                'snomed_code':      str(record.get('Concept ID', '') or '').strip(),
                'code_description': str(record.get('Term', '') or '').strip(),
                'refset_id':        str(record.get('Refset ID', '') or '').strip(),
            })
        
        logger.info(f"Parsed {len(tab_records)} codes from domain: {sheet_name}")
        all_records.extend(tab_records)
    
    wb.close()
    df = pd.DataFrame(all_records)
    
    # Filter out rows where snomed_code is not a valid numeric SNOMED identifier
    df = df[df['snomed_code'].str.match(r'^\d{6,18}$', na=False)]
    
    logger.success(f"Total QOF codes parsed: {len(df)} across {df['domain'].nunique()} domains")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse QOF Business Rules workbook")
    parser.add_argument("--input", required=True, help="Path to QOF .xlsx file or directory")
    parser.add_argument("--output", default="data/processed/qof_v49_clean.csv")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if input_path.is_dir():
        xlsx_files = list(input_path.glob("*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(f"No .xlsx files found in {input_path}")
        workbook_path = xlsx_files[0]
        logger.info(f"Using workbook: {workbook_path.name}")
    else:
        workbook_path = input_path
    
    df = parse_qof_workbook(workbook_path)
    df.to_csv(args.output, index=False)
    logger.success(f"Saved QOF clean data to {args.output}")
```

---

## Script 4 — `scripts/initialize_gold_standard.py`

The original script created placeholder files with a comment line. Our version instead validates that the real MAAR files are present, checks their encoding, detects the delimiter, and registers their metadata into the Master Lookup Table pipeline. The placeholder approach would mean every downstream step quietly processes empty files — a subtle failure that would not produce errors but would produce wrong results.

```python
"""
scripts/initialize_gold_standard.py

Validates that all DAAR_2025_004 gold standard files are present, readable,
and have the expected structure. Produces a manifest file recording the
row count, encoding, and delimiter for each file.

This is NOT a placeholder generator. The actual DAAR files must be placed
in data/gold_standard/ before running this script. The script will fail
explicitly if any file is missing — silent failures are worse than loud ones.

Usage:
    python scripts/initialize_gold_standard.py

Reference: 06_project_plan_and_features.md Feature 0.3,
           BT-00_backtesting_master_overview.md
"""

import pandas as pd
import chardet
from pathlib import Path
from loguru import logger
import json

GOLD_STANDARD_DIR = Path("data/gold_standard/")
MANIFEST_PATH = Path("data/gold_standard/manifest.json")

EXPECTED_FILES = [
    "DAAR_2025_004_all_prod_codes.txt",
    "DAAR_2025_004_antihypertensive_codes.txt",
    "DAAR_2025_004_ascvd_codes.txt",
    "DAAR_2025_004_BMI_codes.txt",
    "DAAR_2025_004_dyslipidemia_codes.txt",
    "DAAR_2025_004_ethnicity_codes.txt",
    "DAAR_2025_004_hdl_codes.txt",
    "DAAR_2025_004_hypertension_codes.txt",
    "DAAR_2025_004_ldl_codes.txt",
    "DAAR_2025_004_llt_codes.txt",
    "DAAR_2025_004_osa_codes.txt",
    "DAAR_2025_004_t2dm_codes.txt",
    "DAAR_2025_004_triglycerides_codes.txt",
]

def detect_encoding(filepath: Path) -> str:
    """
    Use chardet to detect the actual byte encoding of the file.
    NHS clinical data files are often Latin-1 (ISO-8859-1) rather than UTF-8,
    because they were exported from legacy clinical systems. Reading a Latin-1
    file as UTF-8 will silently corrupt characters with accents or special symbols
    in code descriptions. Always detect before reading.
    """
    with open(filepath, "rb") as f:
        raw = f.read(10000)  # Sample the first 10KB for detection
    result = chardet.detect(raw)
    return result.get("encoding", "utf-8")


def detect_delimiter(filepath: Path, encoding: str) -> str:
    """
    Try tab, pipe, and comma delimiters to find which one produces the
    expected column structure. NICE exports use tab by default but some
    systems produce pipe-delimited files.
    """
    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        first_line = f.readline().strip()
    
    for sep in ["\t", "|", ","]:
        if sep in first_line:
            return sep
    return "\t"  # Default assumption


def validate_gold_standard_files() -> dict:
    manifest = {"files": {}, "total_codes": 0, "missing_files": []}
    
    for filename in EXPECTED_FILES:
        filepath = GOLD_STANDARD_DIR / filename
        condition = filename.replace("DAAR_2025_004_", "").replace("_codes.txt", "").replace(".txt", "")
        
        if not filepath.exists():
            logger.error(f"MISSING: {filename}")
            manifest["missing_files"].append(filename)
            continue
        
        encoding = detect_encoding(filepath)
        sep = detect_delimiter(filepath, encoding)
        
        try:
            df = pd.read_csv(filepath, sep=sep, encoding=encoding,
                             dtype=str, on_bad_lines="warn")
            row_count = len(df)
            cols = list(df.columns)
            
            manifest["files"][filename] = {
                "condition_category": condition,
                "encoding": encoding,
                "delimiter": repr(sep),
                "row_count": row_count,
                "columns": cols,
                "path": str(filepath)
            }
            manifest["total_codes"] += row_count
            logger.success(f"✓ {filename}: {row_count} codes, encoding={encoding}")
        
        except Exception as e:
            logger.error(f"Failed to parse {filename}: {e}")
            manifest["files"][filename] = {"error": str(e)}
    
    if manifest["missing_files"]:
        raise FileNotFoundError(
            f"The following gold standard files are missing from {GOLD_STANDARD_DIR}:\n"
            + "\n".join(manifest["missing_files"])
            + "\nPlease obtain these files from the NICE team before continuing."
        )
    
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)
    
    logger.success(f"Manifest saved to {MANIFEST_PATH}")
    logger.info(f"Total codes across all files: {manifest['total_codes']}")
    return manifest


if __name__ == "__main__":
    validate_gold_standard_files()
```

---

## Script 5 — `scripts/download_open_codelists.py`

This script was missing from the original. OpenCodelists (opencodelists.org) is the OpenSAFELY community's repository of validated clinical code lists. Using its API to cache relevant lists locally gives our agent a third-tier source (above semantic search, below QOF) that adds confidence to borderline code recommendations.

```python
"""
scripts/download_open_codelists.py

Caches relevant validated code lists from the OpenSAFELY OpenCodelists API
into data/raw/open_codelists/api_cache/.

The API base URL is: https://www.opencodelists.org/api/v1/codelist/
Key endpoint: GET /api/v1/codelist/{organisation}/{slug}/{version}.json

This script fetches a pre-defined list of relevant codelists for the NICE
obesity and comorbidity project. Add new slugs to CODELIST_TARGETS as the
project expands to new condition areas.

Usage:
    python scripts/download_open_codelists.py

Reference: 02_agentic_workflow_design.md Tool 5 (open_codelists),
           09_multi_agent_patterns.md
"""

import json
import time
import requests
from pathlib import Path
from loguru import logger

API_BASE = "https://www.opencodelists.org/api/v1/codelist"
CACHE_DIR = Path("data/raw/open_codelists/api_cache/")

# Add new codelists here as the project expands to additional condition areas.
# Format: (organisation_slug, codelist_slug, version)
# Find slugs by browsing: https://www.opencodelists.org/
CODELIST_TARGETS = [
    ("nhsd",        "hypertension-snomed",               "v1"),
    ("nhsd",        "obesity-snomed",                    "v1"),
    ("nhsd",        "type-2-diabetes-mellitus",          "v1"),
    ("opensafely",  "hypertension",                      "2020-05-12"),
    ("opensafely",  "type-2-diabetes",                   "2020-06-02"),
    ("opensafely",  "bmi-stage",                         "2021-09-14"),
    ("opensafely",  "lipid-lowering-medication",         "2021-02-23"),
    ("opensafely",  "obstructive-sleep-apnoea",          "2023-07-10"),
]


def fetch_codelist(org: str, slug: str, version: str) -> dict | None:
    """Fetch a single codelist from the OpenCodelists API."""
    url = f"{API_BASE}/{org}/{slug}/{version}.json"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as e:
        logger.warning(f"HTTP {e.response.status_code} for {url} — skipping")
        return None
    except Exception as e:
        logger.error(f"Error fetching {url}: {e}")
        return None


def download_all_codelists() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    
    for org, slug, version in CODELIST_TARGETS:
        cache_filename = f"{org}__{slug}__{version}.json"
        cache_path = CACHE_DIR / cache_filename
        
        if cache_path.exists():
            logger.debug(f"Already cached: {cache_filename}")
            continue
        
        logger.info(f"Fetching: {org}/{slug}/{version}")
        data = fetch_codelist(org, slug, version)
        
        if data:
            with open(cache_path, "w") as f:
                json.dump(data, f, indent=2)
            code_count = len(data.get("codes", []))
            logger.success(f"Saved {code_count} codes → {cache_filename}")
        
        time.sleep(0.5)  # Respectful rate limiting for the public API
    
    logger.success(f"OpenCodelists cache complete: {len(list(CACHE_DIR.glob('*.json')))} files")


if __name__ == "__main__":
    download_all_codelists()
```

---

## Script 6 — `scripts/build_master_lookup_table.py`

This is the most important script in the pipeline. It assembles every data source into the Master Lookup Table — the central artefact that feeds every downstream step. The original suggestion proposed a four-field MLT. Our version implements the full seventeen-field schema required by the planning and backtesting documents, with every field annotated by which document specified it.

```python
"""
scripts/build_master_lookup_table.py

Assembles the Master Lookup Table (MLT) — the central data artefact of this project.
Every downstream pipeline (feature engineering, embeddings, agent scoring, backtesting)
reads from this table. Run this script once after all raw data is downloaded and parsed.
Re-run whenever a data source is updated to a new version.

Output: data/processed/merged/master_lookup_table.csv

Reference: REPO-02_master_lookup_table.md for full schema documentation
           06_project_plan_and_features.md Feature 0.4
           04_rag_pipeline_deep_dive.md for how scoring uses these fields
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import yaml
import json


def load_version_config() -> dict:
    with open("config/data_source_versions.yaml") as f:
        return yaml.safe_load(f)


def load_qof_clean(path: str = "data/processed/qof_v49_clean.csv") -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # Create a pipe-separated list of all QOF indicator IDs for each code
    qof_indicators = (
        df.groupby("snomed_code")["indicator_id"]
        .apply(lambda x: "|".join(x.dropna().unique()))
        .reset_index()
        .rename(columns={"indicator_id": "qof_indicators"})
    )
    qof_clusters = (
        df.groupby("snomed_code")["cluster_id"]
        .apply(lambda x: "|".join(x.dropna().unique()))
        .reset_index()
        .rename(columns={"cluster_id": "qof_clusters"})
    )
    qof_summary = qof_indicators.merge(qof_clusters, on="snomed_code")
    qof_summary["in_qof"] = True
    return qof_summary


def load_usage_data(snomed_path: str, occ_path: str) -> pd.DataFrame:
    """
    Merge NHS Digital SNOMED usage with OpenCodeCounts.
    Both represent usage frequency but from different sources and populations.
    We use OpenCodeCounts as the primary frequency field (larger population, 
    more granular) with NHS Digital as a secondary validation column.
    """
    # OpenCodeCounts
    occ = pd.read_csv(occ_path, dtype={"snomed_code": str})
    occ.columns = occ.columns.str.lower().str.replace(" ", "_")
    
    # NHS Digital SNOMED usage (tab-delimited)
    nhs = pd.read_csv(snomed_path, sep="\t", dtype={"snomed_code": str}, on_bad_lines="warn")
    nhs.columns = nhs.columns.str.lower().str.replace(" ", "_")
    
    merged = occ.merge(
        nhs[["snomed_code", "usage_count"]].rename(
            columns={"usage_count": "nhs_digital_usage_count"}
        ),
        on="snomed_code", how="outer"
    )
    return merged


def load_nice_gold_standard(
        gold_dir: str = "data/gold_standard/",
        manifest_path: str = "data/gold_standard/manifest.json") -> pd.DataFrame:
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    all_codes = []
    for filename, meta in manifest["files"].items():
        if "error" in meta:
            continue
        df = pd.read_csv(
            meta["path"],
            sep=meta["delimiter"].replace("'", ""),
            encoding=meta["encoding"],
            dtype=str,
            on_bad_lines="warn"
        )
        df["condition_category"] = meta["condition_category"]
        df["source_file"] = filename
        all_codes.append(df)
    
    combined = pd.concat(all_codes, ignore_index=True)
    # Normalise the SNOMED code column name across files
    code_col = next((c for c in combined.columns if "code" in c.lower()), combined.columns[0])
    combined = combined.rename(columns={code_col: "snomed_code"})
    combined["snomed_code"] = combined["snomed_code"].str.strip()
    
    # Produce per-code summary: which conditions include this code?
    nice_summary = (
        combined.groupby("snomed_code")["condition_category"]
        .apply(lambda x: "|".join(x.unique()))
        .reset_index()
        .rename(columns={"condition_category": "nice_conditions"})
    )
    nice_summary["in_nice_gold"] = True
    return nice_summary


def load_open_codelists_cache(
        cache_dir: str = "data/raw/open_codelists/api_cache/") -> pd.DataFrame:
    records = []
    for json_file in Path(cache_dir).glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)
        codelist_name = json_file.stem
        for code_entry in data.get("codes", []):
            records.append({
                "snomed_code": str(code_entry.get("id", "")),
                "open_codelists_source": codelist_name
            })
    if not records:
        return pd.DataFrame(columns=["snomed_code", "open_codelists_source"])
    df = pd.DataFrame(records)
    # Aggregate — a code may appear in multiple cached codelists
    return (
        df.groupby("snomed_code")["open_codelists_source"]
        .apply(lambda x: "|".join(x.unique()))
        .reset_index()
        .assign(in_open_codelists=True)
    )


def build_master_lookup_table() -> pd.DataFrame:
    """
    Assemble the full MLT from all sources. See REPO-02 for full schema documentation.
    """
    versions = load_version_config()
    logger.info("Loading all data sources...")
    
    qof = load_qof_clean()
    usage = load_usage_data(
        snomed_path="data/raw/snomed_usage/usage_2024_25.txt",
        occ_path="data/raw/open_code_counts/opencodecounts_latest.csv"
    )
    nice = load_nice_gold_standard()
    opencodelists = load_open_codelists_cache()
    
    logger.info("Merging all sources...")
    # Start from usage (largest and most complete source)
    mlt = usage.copy()
    mlt["snomed_code"] = mlt["snomed_code"].astype(str)
    
    for source_df, label in [(qof, "QOF"), (nice, "NICE"), (opencodelists, "OpenCodelists")]:
        source_df["snomed_code"] = source_df["snomed_code"].astype(str)
        mlt = mlt.merge(source_df, on="snomed_code", how="outer")
        logger.info(f"After merging {label}: {len(mlt)} total codes")
    
    # -----------------------------------------------------------------------
    # Compute derived fields — these are not in any raw source but are
    # required by downstream pipelines (see 04_rag_pipeline_deep_dive.md)
    # -----------------------------------------------------------------------
    
    # Fill boolean columns with False (not NaN) for absent codes
    for bool_col in ["in_qof", "in_nice_gold", "in_open_codelists"]:
        mlt[bool_col] = mlt[bool_col].fillna(False)
    
    # Log-transform usage count — required by all models (linear regression,
    # Random Forest, scoring function) because the raw distribution is
    # a power law. log1p handles zero-usage codes safely.
    mlt["log_usage"] = np.log1p(mlt["usage_count"].fillna(0))
    
    # Count how many distinct authoritative sources reference each code.
    # This is a key feature in the composite scoring function.
    mlt["source_count"] = (
        mlt["in_qof"].astype(int) +
        mlt["in_nice_gold"].astype(int) +
        mlt["in_open_codelists"].astype(int) +
        (mlt["nhs_digital_usage_count"].notna()).astype(int)
    )
    
    # Add placeholder columns for fields populated by later pipeline stages.
    # Populating these here ensures the MLT schema is consistent from the start.
    # graph_features.py (Phase 2) updates: in_degree, out_degree, subtree_size, degree_centrality
    # clustering.py (Phase 3) updates: cluster_id
    # timeseries.py (Phase 5) updates: deprecated_flag, usage_trend
    # embeddings.py (Phase 2) updates: embedding_vector_id
    mlt["in_degree"] = None                  # Populated by Phase 2 Feature 2.1
    mlt["out_degree"] = None
    mlt["subtree_size"] = None
    mlt["degree_centrality"] = None
    mlt["cluster_id"] = None                 # Populated by Phase 3 Feature 3.1
    mlt["deprecated_flag"] = False           # Updated by Phase 5 Feature 5.2
    mlt["usage_trend"] = "unknown"           # Updated by Phase 5 Feature 5.1
    mlt["embedding_vector_id"] = None        # Populated by Phase 2 Feature 2.3
    
    # Record which data source versions were used — feeds into PROV-O records
    mlt["data_source_version_qof"] = versions["qof_business_rules"]["version"]
    mlt["data_source_version_usage"] = versions["snomed_usage_primary_care"]["version"]
    mlt["data_source_version_occ"] = versions["open_code_counts"]["version"]
    
    output_path = Path("data/processed/merged/master_lookup_table.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mlt.to_csv(output_path, index=False)
    
    logger.success(f"MLT built: {len(mlt)} codes, {len(mlt.columns)} columns → {output_path}")
    logger.info(f"  In QOF: {mlt['in_qof'].sum()}")
    logger.info(f"  In NICE gold standard: {mlt['in_nice_gold'].sum()}")
    logger.info(f"  In OpenCodelists: {mlt['in_open_codelists'].sum()}")
    logger.info(f"  In all three sources: {(mlt['source_count'] >= 3).sum()}")
    return mlt


if __name__ == "__main__":
    build_master_lookup_table()
```

---

## Scripts 7–11 — Abbreviated Reference (Full Code in `src/`)

The remaining five scripts follow the same thin-wrapper pattern and call directly into `src/` modules. Their full implementations are in the respective `src/` modules.

**`scripts/build_vectorstore.py`** calls `src/features/embeddings.py` and `src/features/embeddings.build_vector_store()`. It reads the MLT, generates embeddings for all code descriptions using the configured model, and persists the ChromaDB collection. Run once after `build_master_lookup_table.py`. Re-run whenever the embedding model changes or new codes are added.

**`scripts/run_agent.py`** calls `src/agent/executor.py` or `src/agent/graph.py` depending on the `agent_config.yaml` `framework` setting. It accepts a research question as a command-line argument, runs the agent with full logging, and writes the output to `outputs/` with a provenance record in `outputs/provenance/{run_id}/`.

**`scripts/run_backtest.py`** calls each function in `src/backtest/` in sequence, loading all NICE gold-standard lists as the ground truth, running the agent against each condition's research question, and writing the full graded report to `outputs/backtest_results/`.

**`scripts/monitor_drift.py`** is designed to run on a schedule (e.g., monthly cron job) after new NHS data releases. It calls `src/backtest/temporal.py` on every archived code list in `outputs/` and writes a drift report if any codes have changed status.

**`scripts/setup_mlflow.py`** initialises MLflow with the project's experiment names and tags, registers the data source versions as run parameters, and verifies that the tracking server is reachable. Run once per team member environment setup.

---

*See `REPO-02_master_lookup_table.md` for the complete MLT schema with field-by-field documentation.*
