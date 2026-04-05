"""
app_audit.py
------------
Audit, Tracing, and Backtesting for the NICE Clinical Code Assistant.

PURPOSE
━━━━━━━
This file does three things that matter for clinical governance and
for understanding how well the system is working:

  1. AUDIT TRAIL — every time the chatbot makes a recommendation,
     this module records what happened: which model was used, which
     ranker was used, what codes were found, and why each one was
     included. This creates an evidence trail that a clinical reviewer
     can read and challenge.

  2. RUN COMPARISON — you can load two saved runs and ask "what
     changed?". This answers questions like "did switching from Pod 1
     to Pod 2 ranking change which codes appeared?" or "did the new
     QOF data change the output for diabetes queries?".

  3. BACKTESTING — given a set of queries where you already know the
     correct answer (from the NICE gold-standard code lists), this
     module runs the system on those queries and scores how well the
     system did. It produces a clear report explaining the results
     in plain English.

WHY BACKTESTING MATTERS HERE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
The NICE DAAR_2025_004 files are lists of codes that a real NICE
analyst already validated. They are our "known correct answers".
If we run the system on the same clinical questions and compare
the output to those lists, we can measure:

  - How many correct codes did we find? (recall)
  - Of the codes we recommended, how many were actually correct? (precision)
  - How well did our ranking order match what a human expert would choose?

These numbers tell us whether changes to the model, prompt, or ranking
algorithm are actually improvements or regressions.

HOW TO USE THIS FILE
━━━━━━━━━━━━━━━━━━━━
Quick audit of one chat message:
    from app_audit import AuditLogger
    logger = AuditLogger()
    run_id = logger.start_run("obesity with type 2 diabetes", "llama3.2:1b", "Pod 1")
    # ... run your chat function ...
    logger.finish_run(run_id, codes_returned, tool_calls_made)
    logger.save(run_id)

Run a full backtest against the gold-standard files:
    from app_audit import run_backtest
    run_backtest(gold_standard_dir="data/gold_standard/", output_dir="outputs/backtest/")

Compare two saved runs:
    from app_audit import compare_runs
    compare_runs("outputs/run_logs/run_001.json", "outputs/run_logs/run_002.json")
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — DATA STRUCTURES
#
# These are the shapes of the data we store. Using dataclasses
# means every field is named and typed, which makes the saved JSON
# files self-documenting — anyone can open them and understand
# what each value means without reading the code.
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CodeProvenance:
    """
    Records why a single clinical code was recommended.

    Every code in a run gets one of these. Together they form the
    evidence trail — the document a clinical reviewer would read to
    understand and challenge any recommendation.

    Fields
    ------
    snomed_code       The code identifier (e.g. "44054006")
    term              Human-readable name (e.g. "Type 2 diabetes mellitus")
    confidence_score  The ranker's score for this code, 0.0 to 1.0
    rank              Position in the ranked output (1 = most relevant)
    ranked_by         Which ranker produced this rank (e.g. "pod1_tfidf")
    match_score       Initial match score from the CSV search (0.0 to 1.0)
    explanation       The LLM's plain-English reason for inclusion
    source_type       Where the code came from: "csv_match" or "llm_only"
    is_hallucinated   True if this code did NOT come from the CSV — a red flag
    """
    snomed_code:      str
    term:             str
    confidence_score: float
    rank:             int
    ranked_by:        str
    match_score:      float
    explanation:      str   = ""
    source_type:      str   = "csv_match"
    is_hallucinated:  bool  = False


@dataclass
class RunRecord:
    """
    The complete record of one chat message being processed.

    One RunRecord is created every time the user submits a query.
    It records everything about the configuration and the output
    so that the run can be reproduced, compared, or challenged.

    Fields
    ------
    run_id          Unique identifier (timestamp + short hash of query)
    query           The user's exact query string
    model_name      The LLM model used (e.g. "llama3.2:1b")
    ranking_model   The ranker used (e.g. "Pod 1")
    started_at      ISO timestamp when the run began
    completed_at    ISO timestamp when the run finished
    codes           List of CodeProvenance records, one per recommended code
    validation_flags List of issues found during automatic validation
    query_hash      Short hash of the query — used to match runs for backtesting
    """
    run_id:           str
    query:            str
    model_name:       str
    ranking_model:    str
    started_at:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at:     str = ""
    codes:            list = field(default_factory=list)      # list[CodeProvenance]
    validation_flags: list = field(default_factory=list)      # list[dict]
    query_hash:       str  = ""

    def __post_init__(self):
        # Compute a short hash of the query so we can group runs
        # that answered the same clinical question across different
        # model / ranker configurations
        self.query_hash = hashlib.md5(self.query.lower().strip().encode()).hexdigest()[:8]


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — AUDIT LOGGER
#
# The AuditLogger is what app_v2.py calls to record each run.
# It is designed to be non-blocking — if logging fails for any
# reason (disk full, permissions issue), the chat still works.
# ═══════════════════════════════════════════════════════════════════

class AuditLogger:
    """
    Records every chat run as a structured JSON file.

    Usage
    -----
    Call start_run() before the pipeline runs.
    Call finish_run() after the pipeline completes.
    Call save() to write the record to disk.

    The saved files go in outputs/run_logs/ and are named
    by their run_id so they are easy to find and sort.
    """

    def __init__(self, output_dir: str = "outputs/run_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Active runs stored in memory while they are in progress
        self._active: dict[str, RunRecord] = {}

    def start_run(self, query: str, model_name: str, ranking_model: str) -> str:
        """
        Begin tracking a new run.

        Call this at the start of the chat() function in app_v2.py,
        before _extract_candidates() is called.

        Parameters
        ----------
        query          The user's query string
        model_name     Value from the AI Model dropdown
        ranking_model  Value from the Clinical Ranking dropdown

        Returns
        -------
        str  A run_id string — pass this to finish_run() and save()
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        query_short = re.sub(r"[^a-z0-9]", "", query.lower())[:12]
        run_id = f"run_{timestamp}_{query_short}"

        self._active[run_id] = RunRecord(
            run_id=run_id,
            query=query,
            model_name=model_name,
            ranking_model=ranking_model,
        )

        print(f"[audit] started run: {run_id}")
        return run_id

    def finish_run(self, run_id: str, report_items: list[dict]) -> None:
        """
        Complete a run and run automatic validation checks.

        Call this after _format_response() returns, passing the
        report["items"] list so the audit can record each code.

        Parameters
        ----------
        run_id        The string returned by start_run()
        report_items  The final list of code dicts from the pipeline
        """
        rec = self._active.get(run_id)
        if not rec:
            print(f"[audit] WARNING: no active run found for {run_id}")
            return

        rec.completed_at = datetime.now(timezone.utc).isoformat()

        # Record each recommended code as a CodeProvenance object
        for item in report_items:
            cp = CodeProvenance(
                snomed_code=      item.get("code", "UNKNOWN"),
                term=             item.get("term", ""),
                confidence_score= float(item.get("confidence_score", item.get("score", 0.0))),
                rank=             int(item.get("rank", item.get("priority", 0))),
                ranked_by=        item.get("ranked_by", rec.ranking_model),
                match_score=      float(item.get("score", 0.0)),
                explanation=      item.get("explanation", ""),
                source_type=      "csv_match",
            )
            rec.codes.append(cp)

        # Run automatic validation checks
        self._validate(rec)
        print(f"[audit] finished run: {run_id} | codes: {len(rec.codes)} | flags: {len(rec.validation_flags)}")

    def _validate(self, rec: RunRecord) -> None:
        """
        Automatic checks run after every completed run.

        These catch common problems before a human reviewer sees them.
        Problems are recorded as validation_flags rather than raised
        as exceptions so the system can log all issues, not just the first.

        Checks performed:
          - Missing source citations (code appeared with no explanation)
          - Possible hallucination (code not from CSV — UNKNOWN source)
          - Overconfidence (code marked HIGH confidence but score < 0.5)
        """

        # Check 1: any codes with no explanation
        for cp in rec.codes:
            if not cp.explanation or cp.explanation.strip() == "":
                rec.validation_flags.append({
                    "severity": "MEDIUM",
                    "type":     "MISSING_EXPLANATION",
                    "code":     cp.snomed_code,
                    "message":  f"Code {cp.snomed_code} ({cp.term}) has no explanation."
                })

        # Check 2: codes where the source is unknown (possible hallucination)
        for cp in rec.codes:
            if cp.snomed_code == "UNKNOWN" or cp.source_type == "llm_only":
                rec.validation_flags.append({
                    "severity": "CRITICAL",
                    "type":     "POSSIBLE_HALLUCINATION",
                    "code":     cp.snomed_code,
                    "message":  (
                        f"Code '{cp.snomed_code}' could not be matched to the CSV data. "
                        "This code may have been generated by the LLM rather than "
                        "retrieved from the verified dataset."
                    )
                })

        # Check 3: overconfidence — high score but low match
        for cp in rec.codes:
            if cp.confidence_score >= 0.9 and cp.match_score < 0.5:
                rec.validation_flags.append({
                    "severity": "LOW",
                    "type":     "CONFIDENCE_MISMATCH",
                    "code":     cp.snomed_code,
                    "message":  (
                        f"Code {cp.snomed_code} has high confidence ({cp.confidence_score:.0%}) "
                        f"but low initial match score ({cp.match_score:.0%}). "
                        "Verify the ranking model's reasoning."
                    )
                })

    def save(self, run_id: str) -> Path | None:
        """
        Write the run record to disk as a JSON file.

        The file is saved to outputs/run_logs/{run_id}.json.
        Returns the file path, or None if saving failed.
        """
        rec = self._active.get(run_id)
        if not rec:
            print(f"[audit] WARNING: nothing to save for run_id={run_id}")
            return None

        filepath = self.output_dir / f"{run_id}.json"
        data = asdict(rec)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            print(f"[audit] saved: {filepath}")
            return filepath
        except Exception as e:
            print(f"[audit] ERROR: could not save {filepath}: {e}")
            return None

    def print_summary(self, run_id: str) -> None:
        """Print a human-readable summary of a completed run to the terminal."""
        rec = self._active.get(run_id)
        if not rec:
            print(f"[audit] No record found for run_id={run_id}")
            return

        print(f"\n{'='*60}")
        print(f"AUDIT SUMMARY: {run_id}")
        print(f"{'='*60}")
        print(f"Query:          {rec.query}")
        print(f"Model:          {rec.model_name}")
        print(f"Ranker:         {rec.ranking_model}")
        print(f"Codes returned: {len(rec.codes)}")

        if rec.validation_flags:
            print(f"\nValidation flags ({len(rec.validation_flags)}):")
            for flag in rec.validation_flags:
                icons = {"CRITICAL": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
                icon = icons.get(flag["severity"], "⚪")
                print(f"  {icon} [{flag['severity']}] {flag['type']}: {flag['message']}")
        else:
            print("\n✓ No validation issues found.")


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — RUN COMPARATOR
#
# Loads two saved run JSON files and explains what changed between
# them. Useful for understanding the effect of switching models,
# rankers, or data sources.
# ═══════════════════════════════════════════════════════════════════

def compare_runs(run_a_path: str, run_b_path: str) -> dict:
    """
    Compare two saved run records and explain what changed.

    This answers questions like:
      - "Did switching from Pod 1 to Pod 2 ranking change the output?"
      - "Did updating the LLM model change which codes appeared?"
      - "Did the QOF data update affect the hypertension query?"

    Parameters
    ----------
    run_a_path  Path to the first run's JSON file (the "before")
    run_b_path  Path to the second run's JSON file (the "after")

    Returns
    -------
    dict  A structured summary of what changed, including:
          - which configuration settings changed
          - which codes were added, removed, or changed rank
          - whether validation issues appeared or disappeared
    """

    with open(run_a_path, encoding="utf-8") as f:
        run_a = json.load(f)
    with open(run_b_path, encoding="utf-8") as f:
        run_b = json.load(f)

    # Build code lookup dicts: {snomed_code: code_record}
    codes_a = {c["snomed_code"]: c for c in run_a.get("codes", [])}
    codes_b = {c["snomed_code"]: c for c in run_b.get("codes", [])}
    all_codes = set(codes_a) | set(codes_b)

    # Identify configuration changes (what was different between runs)
    config_changes = {}
    for field in ["model_name", "ranking_model"]:
        va, vb = run_a.get(field), run_b.get(field)
        if va != vb:
            config_changes[field] = {"run_a": va, "run_b": vb}

    # Classify each code by what happened to it
    added, removed, rank_changed, unchanged = [], [], [], []
    for code in all_codes:
        in_a, in_b = code in codes_a, code in codes_b
        if in_a and not in_b:
            removed.append({"code": code, "term": codes_a[code].get("term",""), "was_rank": codes_a[code].get("rank")})
        elif in_b and not in_a:
            added.append({"code": code, "term": codes_b[code].get("term",""), "now_rank": codes_b[code].get("rank")})
        elif codes_a[code].get("rank") != codes_b[code].get("rank"):
            rank_changed.append({
                "code": code,
                "term": codes_a[code].get("term",""),
                "rank_a": codes_a[code].get("rank"),
                "rank_b": codes_b[code].get("rank"),
            })
        else:
            unchanged.append(code)

    result = {
        "run_a_id":         run_a.get("run_id"),
        "run_b_id":         run_b.get("run_id"),
        "query":            run_a.get("query"),
        "config_changes":   config_changes,
        "codes_added":      added,
        "codes_removed":    removed,
        "rank_changed":     rank_changed,
        "unchanged_count":  len(unchanged),
        "flags_a":          len(run_a.get("validation_flags", [])),
        "flags_b":          len(run_b.get("validation_flags", [])),
    }

    # Print readable summary
    print(f"\n{'='*60}")
    print("RUN COMPARISON")
    print(f"{'='*60}")
    print(f"Query: {result['query']}")
    if config_changes:
        print("\nConfiguration changes (why results may differ):")
        for k, v in config_changes.items():
            print(f"  {k}: {v['run_a']} → {v['run_b']}")
    else:
        print("\nNo configuration changes — identical settings.")

    print(f"\nCode changes:")
    print(f"  Added:         {len(added)}")
    print(f"  Removed:       {len(removed)}")
    print(f"  Rank changed:  {len(rank_changed)}")
    print(f"  Unchanged:     {len(unchanged)}")
    print(f"\nValidation flags: {result['flags_a']} → {result['flags_b']}")

    if added:
        print("\nCodes added in Run B:")
        for c in added:
            print(f"  + {c['code']} | {c['term']} (rank {c['now_rank']})")
    if removed:
        print("\nCodes removed from Run B:")
        for c in removed:
            print(f"  - {c['code']} | {c['term']} (was rank {c['was_rank']})")

    return result


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — BACKTESTING
#
# Backtesting is the process of running the system on queries where
# we already know the correct answer and measuring how well it did.
#
# WHY IT MATTERS FOR UNDERSTANDING MODEL REASONING:
# The NICE gold-standard files (DAAR_2025_004) contain code lists
# that a real clinical analyst validated. They represent expert
# human judgement. When we run the system on the same conditions
# and compare the output, we are measuring how closely the system
# reproduces that expert judgement.
#
# A high recall score (close to 1.0) means the system found most
# of the codes the expert chose. A low recall means it missed many.
# Precision measures whether the codes it recommended were mostly
# correct, or whether it added a lot of irrelevant ones.
#
# Rank correlation measures whether the system ranked the codes in
# a similar order to how an expert would prioritise them — this is
# particularly relevant for evaluating Pod 1 vs Pod 2 ranking.
# ═══════════════════════════════════════════════════════════════════

def score_against_gold_standard(
    recommended_codes: list[str],
    gold_standard_codes: set[str],
    usage_counts: dict[str, int] | None = None,
) -> dict:
    """
    Score a set of recommended codes against a gold-standard list.

    This is the core backtesting measurement. It computes three metrics:

    RECALL
      "Of all the codes the expert chose, how many did we find?"
      Formula: true_positives / total_gold_standard_codes
      A recall of 0.85 means we found 85% of the expert's codes.
      Higher is better. Missing codes is a serious problem in clinical
      coding because a missed code means a patient is not counted.

    PRECISION
      "Of the codes we recommended, how many were actually correct?"
      Formula: true_positives / total_recommended_codes
      A precision of 0.70 means 70% of our recommendations were good.
      Lower precision means we recommended many irrelevant codes.

    PATIENT-WEIGHTED RECALL
      The same as recall but weighted by how many patients each missed
      code represents (using national usage counts). Missing a code
      used by 500,000 patients is much more serious than missing one
      used by 50 patients. If usage_counts is not provided, this is
      set to the same value as regular recall.

    Parameters
    ----------
    recommended_codes    List of code strings the system returned
    gold_standard_codes  Set of code strings from the expert-validated list
    usage_counts         Optional dict mapping code string to annual count

    Returns
    -------
    dict with:
        recall, precision, f1, patient_weighted_recall,
        true_positives, false_negatives, false_positives (as lists)
    """
    recommended_set = set(recommended_codes)

    true_positives  = recommended_set & gold_standard_codes
    false_negatives = gold_standard_codes - recommended_set
    false_positives = recommended_set - gold_standard_codes

    recall    = len(true_positives) / max(len(gold_standard_codes), 1)
    precision = len(true_positives) / max(len(recommended_set), 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-9)

    # Patient-weighted recall: weights each missed code by its national usage count.
    # This reflects clinical impact — missing a high-frequency code is worse.
    patient_weighted_recall = recall  # default if no usage data
    if usage_counts:
        total_gold_usage  = sum(usage_counts.get(c, 0) for c in gold_standard_codes)
        missed_usage      = sum(usage_counts.get(c, 0) for c in false_negatives)
        if total_gold_usage > 0:
            patient_weighted_recall = 1.0 - (missed_usage / total_gold_usage)

    return {
        "recall":                  round(recall, 4),
        "precision":               round(precision, 4),
        "f1":                      round(f1, 4),
        "patient_weighted_recall": round(patient_weighted_recall, 4),
        "true_positives":          sorted(true_positives),
        "false_negatives":         sorted(false_negatives),
        "false_positives":         sorted(false_positives),
        "n_recommended":           len(recommended_set),
        "n_gold_standard":         len(gold_standard_codes),
    }


def load_gold_standard_file(filepath: str) -> set[str]:
    """
    Load a NICE gold-standard code list file and return the set of codes.

    These are the DAAR_2025_004 files provided by NICE.
    They are tab-delimited text files where one column contains the code.
    This function handles both UTF-8 and UTF-8-with-BOM encoding.

    Parameters
    ----------
    filepath  Path to a gold standard .txt file

    Returns
    -------
    set[str]  The set of code strings from the file
    """
    import csv
    codes: set[str] = set()
    path = Path(filepath)

    if not path.exists():
        print(f"[backtest] WARNING: gold standard file not found: {filepath}")
        return codes

    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        headers = [h.lower().strip() for h in (reader.fieldnames or [])]

        # Find the column that contains the code
        code_col = next(
            (h for h in reader.fieldnames or []
             if h.lower().strip() in {"code", "medcodeid", "snomedctconceptid"}),
            None
        )
        if not code_col:
            # If no header matched, try the first column
            code_col = (reader.fieldnames or [None])[0]

        for row in reader:
            val = str(row.get(code_col, "")).strip()
            if val and val != "code":
                codes.add(val)

    print(f"[backtest] loaded {len(codes)} codes from {path.name}")
    return codes


def run_backtest(
    chat_fn,
    gold_standard_dir: str,
    model_name: str,
    ranking_model: str,
    output_dir:    str = "outputs/backtest/",
) -> dict:
    """
    Run a full backtest against all gold-standard files in a directory.

    This is the main function to call when you want to measure how
    well a particular model + ranker combination performs overall.

    For each gold-standard condition file, it:
      1. Extracts the condition name from the filename
      2. Runs the chat function with that condition as the query
      3. Compares the output codes to the gold-standard codes
      4. Records recall, precision, and F1 for that condition

    Then it aggregates results across all conditions.

    Parameters
    ----------
    chat_fn            The chat() function from app_v2.py
    gold_standard_dir  Path to the folder containing DAAR_2025_004 files
    model_name         The LLM model to use for this backtest run
    ranking_model      "Pod 1" or "Pod 2"
    output_dir         Where to save the backtest report

    Returns
    -------
    dict  Full results, one entry per condition plus an aggregate summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    gold_dir  = Path(gold_standard_dir)
    results   = {}
    all_recall, all_precision, all_f1 = [], [], []

    gold_files = list(gold_dir.glob("DAAR_2025_004_*.txt"))
    if not gold_files:
        print(f"[backtest] No gold standard files found in {gold_dir}")
        return {}

    print(f"\n[backtest] Starting backtest: model={model_name} | ranker={ranking_model}")
    print(f"[backtest] {len(gold_files)} conditions to test\n")

    for gold_file in sorted(gold_files):
        # Extract condition name from filename
        # e.g. "DAAR_2025_004_hypertension_codes.txt" → "hypertension"
        condition = gold_file.stem.replace("DAAR_2025_004_", "").replace("_codes", "")

        gold_codes = load_gold_standard_file(str(gold_file))
        if not gold_codes:
            continue

        # Run the chat function as if a user had typed the condition name
        try:
            response = chat_fn(
                message=condition.replace("_", " "),
                history=[],
                model_choice=model_name,
                ranking_choice=ranking_model,
            )
        except Exception as e:
            print(f"[backtest] ERROR running {condition}: {e}")
            results[condition] = {"error": str(e)}
            continue

        # Extract code identifiers from the response text
        # The response is Markdown — codes appear in backticks as `{code}`
        found_codes = set(re.findall(r"`(\d{6,18})`", response))

        scores = score_against_gold_standard(list(found_codes), gold_codes)
        results[condition] = {
            "condition":   condition,
            "model":       model_name,
            "ranker":      ranking_model,
            "gold_count":  len(gold_codes),
            **scores,
        }

        all_recall.append(scores["recall"])
        all_precision.append(scores["precision"])
        all_f1.append(scores["f1"])

        status_icon = "✓" if scores["recall"] >= 0.6 else "✗"
        print(
            f"  {status_icon} {condition:<30} "
            f"recall={scores['recall']:.2f}  "
            f"precision={scores['precision']:.2f}  "
            f"f1={scores['f1']:.2f}"
        )

    # Aggregate summary
    if all_recall:
        summary = {
            "model":            model_name,
            "ranker":           ranking_model,
            "conditions_tested": len(results),
            "mean_recall":      round(sum(all_recall) / len(all_recall), 4),
            "mean_precision":   round(sum(all_precision) / len(all_precision), 4),
            "mean_f1":          round(sum(all_f1) / len(all_f1), 4),
            "pass_count":       sum(1 for r in all_recall if r >= 0.6),
        }
        results["_summary"] = summary

        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY: {model_name} / {ranking_model}")
        print(f"{'='*60}")
        print(f"Conditions tested:  {summary['conditions_tested']}")
        print(f"Mean recall:        {summary['mean_recall']:.1%}")
        print(f"Mean precision:     {summary['mean_precision']:.1%}")
        print(f"Mean F1:            {summary['mean_f1']:.3f}")
        print(f"Passed (recall≥60%): {summary['pass_count']} / {summary['conditions_tested']}")

    # Save the full results report
    report_path = output_path / f"backtest_{model_name.replace(':','_')}_{ranking_model.replace(' ','')}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n[backtest] Report saved: {report_path}")

    return results


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — HOW TO INTEGRATE WITH app_v2.py
#
# This section shows example code you can copy into app_v2.py
# when you are ready to turn on the audit trail. No changes are
# needed to any other files.
# ═══════════════════════════════════════════════════════════════════

INTEGRATION_EXAMPLE = '''
# ── Add this near the top of app_v2.py, after the imports ─────────
from app_audit import AuditLogger
_audit = AuditLogger(output_dir="outputs/run_logs")


# ── Replace the chat() function body in Section F with this ───────
def chat(message, history, model_choice, ranking_choice):
    if not message.strip():
        return "Please enter a clinical query."

    _apply_model_choice(model_choice)

    # Start the audit trail for this run
    run_id = _audit.start_run(message, model_choice, ranking_choice)

    try:
        candidates = _extract_candidates(message)
        report     = _build_report_from_candidates(candidates)

        if not report["items"]:
            return "No matching codes found."

        ranked = _apply_ranking(
            [{"MedCodeId":i["code"],"term":i["term"],"score":i["score"]} for i in report["items"]],
            message, ranking_choice,
        )
        rm = {c["MedCodeId"]: c for c in ranked}
        for item in report["items"]:
            r = rm.get(item["code"], {})
            item["rank"]             = r.get("rank", item["priority"])
            item["confidence_score"] = r.get("confidence_score", item["score"])
            item["ranked_by"]        = r.get("ranked_by", ranking_choice)
        report["items"].sort(key=lambda x: x.get("rank", 999))

        report = _add_llm_explanations(report, message)

        # Finish the audit trail and save it
        _audit.finish_run(run_id, report["items"])
        _audit.save(run_id)

        return _format_response(report, model_choice, ranking_choice)

    except Exception as e:
        return f"Error: {e}"


# ── To run a backtest from the command line ───────────────────────
# python -c "
# from app_v2 import chat
# from app_audit import run_backtest
# run_backtest(chat, 'data/gold_standard/', 'llama3.2:1b', 'Pod 1')
# "
'''


if __name__ == "__main__":
    # Quick self-test — demonstrates each component with mock data
    print("app_audit.py self-test\n")

    # 1. Test the audit logger
    logger = AuditLogger(output_dir="/tmp/test_audit_logs")
    run_id = logger.start_run("obesity with type 2 diabetes", "llama3.2:1b", "Pod 1")

    mock_items = [
        {"code": "44054006", "term": "Type 2 diabetes mellitus",
         "rank": 1, "confidence_score": 0.91, "score": 0.8,
         "ranked_by": "pod1_placeholder", "explanation": "Primary diagnosis code for T2DM."},
        {"code": "414916001", "term": "Obesity (BMI 30+)",
         "rank": 2, "confidence_score": 0.85, "score": 0.8,
         "ranked_by": "pod1_placeholder", "explanation": "Obesity classification code."},
        {"code": "UNKNOWN", "term": "mystery code",
         "rank": 3, "confidence_score": 0.3, "score": 0.1,
         "ranked_by": "pod1_placeholder", "explanation": ""},
    ]

    logger.finish_run(run_id, mock_items)
    logger.print_summary(run_id)
    logger.save(run_id)

    # 2. Test the scoring function
    print("\nScoring test:")
    scores = score_against_gold_standard(
        recommended_codes=["44054006", "414916001", "99999999"],
        gold_standard_codes={"44054006", "414916001", "73211009"},
    )
    print(f"  Recall:    {scores['recall']:.1%}")
    print(f"  Precision: {scores['precision']:.1%}")
    print(f"  F1:        {scores['f1']:.3f}")
    print(f"  Missed:    {scores['false_negatives']}")
    print(f"  Extra:     {scores['false_positives']}")

    print("\nSelf-test complete.")
