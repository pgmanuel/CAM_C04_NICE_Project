"""
feedback_hitl.py
----------------
Human-in-the-Loop (HITL) Feedback Collection.

PURPOSE
-------
Every AI recommendation should be reviewable by a human. This module
provides the mechanism for clinical analysts to record whether a set
of recommended codes was useful, and for that feedback to be stored
in a structured, queryable format.

The feedback has two uses:
  1. QUALITY MONITORING — over time, the thumbs-up/down ratio tells
     you whether the system is improving or degrading as the models,
     data, or prompts change.
  2. TRAINING SIGNAL — accumulated feedback can be used to fine-tune
     the hybrid scoring weights or to build a classifier that learns
     from analyst preferences.

WHAT GETS RECORDED
Each feedback event stores:
  - run_id: which system run produced the recommendation
  - query: what the analyst asked
  - recommended_codes: the codes that were shown
  - vote: "thumbs_up" or "thumbs_down"
  - note: optional free-text from the analyst
  - model_name: which LLM was used
  - timestamp: when the feedback was submitted

All feedback is saved as JSON files in outputs/feedback/.
A summary CSV is maintained for easy analysis.

HOW IT INTEGRATES WITH THE UI
app.py renders two buttons (👍 / 👎) below each assistant response.
When clicked, they call record_feedback() with the run_id of the
response that was just shown. The buttons are disabled after one click
per response to prevent duplicate votes.

Usage
-----
    from feedback_hitl import record_feedback, load_feedback_summary

    run_id = "run_20260410_143022_obesity"

    # Record a positive vote
    record_feedback(
        run_id=run_id,
        query="obesity with type 2 diabetes",
        recommended_codes=["44054006", "414916001"],
        vote="thumbs_up",
        model_name="llama3.2:1b",
        note="Codes are correct and QOF flags are accurate"
    )

    # Get aggregated statistics
    summary = load_feedback_summary()
    print(f"Positive rate: {summary['positive_rate']:.0%}")
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ─── Storage location ─────────────────────────────────────────────
# Relative to this script so it works regardless of working directory
_FEEDBACK_DIR     = Path(__file__).parent / "outputs" / "feedback"
_FEEDBACK_CSV     = _FEEDBACK_DIR / "feedback_summary.csv"
_FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
# DATA STRUCTURE
# ─────────────────────────────────────────────────────────────────

@dataclass
class FeedbackRecord:
    """
    A single feedback event.

    Every field is included in both the JSON file (full detail) and
    the summary CSV (for analysis). The dataclass ensures every field
    is named and typed, which makes the saved files self-documenting.
    """
    run_id:              str
    query:               str
    vote:                str            # "thumbs_up" or "thumbs_down"
    recommended_codes:   list = field(default_factory=list)  # list of code strings
    model_name:          str = ""
    note:                str = ""       # Optional analyst comment
    n_codes:             int = 0        # Derived from recommended_codes
    n_qof_codes:         int = 0        # How many were QOF-mandated
    timestamp:           str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.n_codes:
            self.n_codes = len(self.recommended_codes)


# ─────────────────────────────────────────────────────────────────
# RECORD FEEDBACK
# ─────────────────────────────────────────────────────────────────

def record_feedback(
    run_id:            str,
    query:             str,
    vote:              str,
    recommended_codes: list[str]   = None,
    model_name:        str         = "",
    note:              str         = "",
    n_qof_codes:       int         = 0,
) -> dict:
    """
    Save a feedback event to disk and update the summary CSV.

    Parameters
    ----------
    run_id             The run_id from the pipeline response
    query              The original query string
    vote               "thumbs_up" or "thumbs_down"
    recommended_codes  List of SNOMED code strings that were shown
    model_name         Which LLM model produced the response
    note               Optional analyst comment (free text)
    n_qof_codes        How many of the returned codes were QOF-mandated

    Returns
    -------
    dict  The saved FeedbackRecord as a dict, for UI confirmation
    """
    vote = vote.lower().strip()
    if vote not in ("thumbs_up", "thumbs_down"):
        raise ValueError(f"vote must be 'thumbs_up' or 'thumbs_down', got '{vote}'")

    rec = FeedbackRecord(
        run_id=run_id,
        query=query,
        vote=vote,
        recommended_codes=recommended_codes or [],
        model_name=model_name,
        note=note,
        n_qof_codes=n_qof_codes,
    )

    # ── Save individual JSON file ──────────────────────────────────
    safe_id = run_id.replace(":", "_").replace("/", "_")
    json_path = _FEEDBACK_DIR / f"{safe_id}_{vote[:2]}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(asdict(rec), f, indent=2, default=str)

    # ── Update summary CSV ─────────────────────────────────────────
    _append_to_csv(rec)

    vote_label = "👍 Positive" if vote == "thumbs_up" else "👎 Negative"
    print(f"[feedback] Recorded {vote_label} for run_id={run_id}")

    return asdict(rec)


def _append_to_csv(rec: FeedbackRecord) -> None:
    """Append one row to the summary CSV for easy spreadsheet analysis."""
    row = {
        "timestamp":    rec.timestamp,
        "run_id":       rec.run_id,
        "vote":         rec.vote,
        "query":        rec.query[:100],
        "n_codes":      rec.n_codes,
        "n_qof_codes":  rec.n_qof_codes,
        "model_name":   rec.model_name,
        "note":         rec.note[:200] if rec.note else "",
    }
    write_header = not _FEEDBACK_CSV.exists()
    with open(_FEEDBACK_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────
# QUERY FEEDBACK
# ─────────────────────────────────────────────────────────────────

def load_feedback_summary() -> dict:
    """
    Load and aggregate all feedback records.

    Returns
    -------
    dict with keys:
        total           int    — total feedback events recorded
        thumbs_up       int    — positive votes
        thumbs_down     int    — negative votes
        positive_rate   float  — thumbs_up / total (0.0 to 1.0)
        recent          list   — last 10 feedback events as dicts
        by_model        dict   — positive_rate grouped by model name
    """
    if not _FEEDBACK_CSV.exists():
        return {
            "total": 0, "thumbs_up": 0, "thumbs_down": 0,
            "positive_rate": 0.0, "recent": [], "by_model": {}
        }

    rows = []
    with open(_FEEDBACK_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total       = len(rows)
    thumbs_up   = sum(1 for r in rows if r.get("vote") == "thumbs_up")
    thumbs_down = total - thumbs_up

    by_model: dict[str, dict] = {}
    for row in rows:
        model = row.get("model_name", "unknown") or "unknown"
        if model not in by_model:
            by_model[model] = {"total": 0, "thumbs_up": 0}
        by_model[model]["total"] += 1
        if row.get("vote") == "thumbs_up":
            by_model[model]["thumbs_up"] += 1
    for model_data in by_model.values():
        t = model_data["total"]
        model_data["positive_rate"] = round(model_data["thumbs_up"] / t, 3) if t else 0.0

    return {
        "total":         total,
        "thumbs_up":     thumbs_up,
        "thumbs_down":   thumbs_down,
        "positive_rate": round(thumbs_up / total, 3) if total else 0.0,
        "recent":        rows[-10:],
        "by_model":      by_model,
    }


def get_feedback_for_run(run_id: str) -> dict | None:
    """Return the feedback record for a specific run_id, or None."""
    if not _FEEDBACK_CSV.exists():
        return None
    with open(_FEEDBACK_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("run_id") == run_id:
                return row
    return None


def format_feedback_summary_panel() -> str:
    """
    Format the feedback summary as Markdown for the UI dashboard panel.

    Called by app.py to populate the HITL Summary section.
    """
    s = load_feedback_summary()
    if s["total"] == 0:
        return (
            "### 👍👎 Analyst Feedback Summary\n\n"
            "*No feedback recorded yet. Use the thumbs buttons below each "
            "response to rate recommendations.*"
        )

    rate_icon = "🟢" if s["positive_rate"] >= 0.7 else "🟡" if s["positive_rate"] >= 0.5 else "🔴"
    lines = [
        "### 👍👎 Analyst Feedback Summary\n",
        f"| | Count |",
        f"|---|---|",
        f"| 👍 Positive | **{s['thumbs_up']}** |",
        f"| 👎 Negative | **{s['thumbs_down']}** |",
        f"| **Positive rate** | {rate_icon} **{s['positive_rate']:.0%}** |",
        "",
    ]
    if s["by_model"]:
        lines.append("**By model:**")
        for model, data in s["by_model"].items():
            icon = "🟢" if data["positive_rate"] >= 0.7 else "🟡"
            lines.append(
                f"- `{model}`: {icon} {data['positive_rate']:.0%} positive "
                f"({data['thumbs_up']}/{data['total']})"
            )
    return "\n".join(lines)
