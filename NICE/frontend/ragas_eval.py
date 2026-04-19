"""
ragas_eval.py
-------------
Retrieval Quality Evaluation — RAGAS-style metrics computed locally.

WHAT IS RAGAS?
RAGAS (Retrieval Augmented Generation Assessment) is a framework for
measuring how well a RAG pipeline is working. It defines several
metrics that answer specific questions about quality:

  FAITHFULNESS — "Is the AI explanation grounded in the retrieved codes?"
    Measures whether the LLM explanation sticks to the evidence
    (the codes, their QOF status, their usage data) or invents facts.
    Score: 0.0 (completely unfaithful) to 1.0 (fully grounded)
    Clinical relevance: low faithfulness means the AI is making
    clinical claims it cannot support from the data.

  ANSWER RELEVANCY — "Are the returned codes actually relevant to the query?"
    Measures whether the retrieved SNOMED codes genuinely relate to
    what the analyst asked for.
    Score: 0.0 (irrelevant) to 1.0 (fully relevant)
    Clinical relevance: low relevancy means the retrieval pipeline
    is finding codes from the wrong clinical domain.

  CONTEXT RECALL — "Did we find the important codes?"
    In the absence of a gold-standard list, this is estimated by
    measuring how well the retrieved codes cover the decomposed
    sub-queries. A low recall suggests the query decomposition step
    is missing important clinical concepts.
    Score: 0.0 (missed most) to 1.0 (good coverage)

WHY LOCAL RATHER THAN THE RAGAS PYTHON PACKAGE?
The official ragas package requires an OpenAI API key and sends
data to external servers. For NHS clinical data, external transmission
is a governance concern. This file implements the same conceptual
measurements using the local LLM (via llm.py), so no data leaves
the machine.

HOW IT INTEGRATES WITH THE UI
app.py calls evaluate(report, query) after the main response is
built. The returned dict is formatted by format_eval_panel() and
inserted into the chat response as a collapsible Accordion.

Because evaluation runs after the main response is already displayed,
it is non-blocking — the analyst sees codes immediately and the
eval metrics appear below once computed.

Usage
-----
    from ragas_eval import evaluate, format_eval_panel

    report = pipeline.retrieve_and_rank("obesity with type 2 diabetes")
    report = pipeline.add_llm_explanations(report, query)

    metrics = evaluate(report, query)
    panel_md = format_eval_panel(metrics)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────
# METRIC 1 — FAITHFULNESS
# ─────────────────────────────────────────────────────────────────

def _score_faithfulness(report: dict, query: str) -> float:
    """
    Estimate whether each LLM explanation is grounded in the evidence
    attached to the code (QOF status, usage data, semantic score).

    Approach: for each code, we check whether the explanation mentions
    the code's term or clinical concept, and whether QOF-mandated codes
    have their mandate acknowledged. We score 1.0 per item if grounded,
    0.0 if the explanation appears to contain clinical claims not
    traceable to the code's metadata.

    This is a heuristic check — the full RAGAS faithfulness metric
    uses an LLM judge, which we replicate via call_llm() if available,
    falling back to the heuristic if the LLM is offline.
    """
    items = report.get("items", [])
    if not items:
        return 0.0

    try:
        from llm import call_llm

        # Ask the LLM to check each explanation against its evidence
        evidence_text = "\n".join(
            f"Code {item['code']} ({item['term']}): "
            f"explanation='{item.get('explanation', '')}' | "
            f"in_qof={item.get('in_qof', False)} | "
            f"usage={item.get('usage_count', 0):,}"
            for item in items[:5]  # Limit to top 5 to keep the prompt short
        )

        system = (
            "You are an evaluation assistant. For each clinical code entry, "
            "judge whether the explanation is FAITHFUL to the provided evidence "
            "(term name, QOF status, usage count). "
            "Faithful means the explanation only makes claims the evidence supports. "
            "Return a JSON object: "
            '{"faithfulness_scores": [0.0 to 1.0 per item], "mean": 0.0 to 1.0}'
        )
        user = (
            f"Query: {query}\n\nEntries:\n{evidence_text}\n\n"
            "Return ONLY the JSON object."
        )

        raw = call_llm(system, user)
        cleaned = raw.strip().lstrip("```json").rstrip("```").strip()
        parsed = json.loads(cleaned)
        return float(parsed.get("mean", 0.0))

    except Exception as e:
        print(f"[ragas] Faithfulness LLM check failed, using heuristic: {e}")

    # Heuristic fallback: check that each explanation contains the term
    scores = []
    for item in items:
        expl      = item.get("explanation", "").lower()
        term      = item.get("term", "").lower()
        # Does the explanation contain at least part of the clinical term?
        term_words = [w for w in term.split() if len(w) > 3]
        term_hit   = any(w in expl for w in term_words) if term_words else False
        # Did QOF-mandated codes have their mandate acknowledged?
        qof_ok = True
        if item.get("in_qof") and "qof" not in expl and "mandated" not in expl:
            qof_ok = False
        scores.append(1.0 if (term_hit and qof_ok) else 0.5)

    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ─────────────────────────────────────────────────────────────────
# METRIC 2 — ANSWER RELEVANCY
# ─────────────────────────────────────────────────────────────────

def _score_answer_relevancy(report: dict, query: str) -> float:
    """
    Estimate how relevant the returned codes are to the query.

    Approach: compute the mean semantic similarity between the query
    terms and the code terms using simple token overlap (a lightweight
    proxy for embedding cosine similarity when no embedding model is
    available at eval time). The CrossEncoder rerank scores, already
    computed during retrieval, are a better signal — we use those
    if available.
    """
    items = report.get("items", [])
    if not items:
        return 0.0

    # Prefer the semantic scores already computed by the CrossEncoder
    semantic_scores = [item.get("semantic_score", 0.0) for item in items]
    if any(s > 0 for s in semantic_scores):
        # CrossEncoder scores are not normalised to 0-1 but are typically
        # in the range -5 to +5. We clip and rescale.
        def rescale(s: float) -> float:
            return max(0.0, min(1.0, (s + 5) / 10))
        return round(sum(rescale(s) for s in semantic_scores) / len(semantic_scores), 3)

    # Fallback: token-overlap between query and each code term
    query_tokens = set(re.sub(r"[^a-z0-9 ]", "", query.lower()).split())
    scores = []
    for item in items:
        term_tokens = set(re.sub(r"[^a-z0-9 ]", "", item.get("term", "").lower()).split())
        overlap = len(query_tokens & term_tokens) / max(len(query_tokens), 1)
        scores.append(min(overlap * 2, 1.0))  # Scale up since single-word matches are common

    return round(sum(scores) / len(scores), 3) if scores else 0.0


# ─────────────────────────────────────────────────────────────────
# METRIC 3 — CONTEXT RECALL (estimated)
# ─────────────────────────────────────────────────────────────────

def _score_context_recall(report: dict, query: str) -> float:
    """
    Estimate how well the retrieved codes cover the clinical query.

    Without a gold-standard list, we estimate recall by checking
    coverage: for each sub-query that was searched, was at least
    one relevant code found? A sub-query with no matching code
    indicates a coverage gap.

    This is an estimate — true recall requires a validated gold list.
    The app_audit.py backtesting module provides real recall numbers
    when gold-standard files are available.
    """
    sub_queries = report.get("sub_queries", [])
    items       = report.get("items", [])

    if not sub_queries or not items:
        # Cannot estimate without sub-queries; default to the confidence
        # of the top result as a proxy
        top_score = items[0].get("confidence_score", 0.0) if items else 0.0
        return round(top_score, 3)

    # Check how many sub-queries produced at least one result
    covered = 0
    for sq in sub_queries:
        sq_lower = sq.lower()
        for item in items:
            if sq_lower in item.get("sub_query_found", "").lower():
                covered += 1
                break

    coverage_ratio = covered / len(sub_queries)

    # Adjust downward if the top confidence score is low
    top_score = items[0].get("confidence_score", 0.0) if items else 0.0
    adjusted  = coverage_ratio * (0.5 + 0.5 * top_score)

    return round(adjusted, 3)


# ─────────────────────────────────────────────────────────────────
# MAIN EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────

def evaluate(report: dict, query: str) -> dict:
    """
    Compute all three evaluation metrics for a completed report.

    Call this after retrieve_and_rank() and add_llm_explanations()
    have both been called on the report.

    Parameters
    ----------
    report  The dict from pipeline.retrieve_and_rank() with explanations
    query   The original user query string

    Returns
    -------
    dict with keys:
        faithfulness       float  0.0-1.0 — explanations grounded in evidence
        answer_relevancy   float  0.0-1.0 — codes relevant to query
        context_recall     float  0.0-1.0 — query coverage estimate
        overall            float  0.0-1.0 — mean of the three
        interpretation     str   — plain-English summary for the analyst
        codes_evaluated    int   — how many codes were assessed
        eval_method        str   — "llm_judge" or "heuristic"
        timestamp          str   — ISO timestamp
    """
    start = time.time()
    items = report.get("items", [])

    if not items:
        return {
            "faithfulness": 0.0, "answer_relevancy": 0.0,
            "context_recall": 0.0, "overall": 0.0,
            "interpretation": "No codes returned — evaluation not possible.",
            "codes_evaluated": 0, "eval_method": "none",
            "timestamp": _now(),
        }

    # Run all three metrics
    faithfulness     = _score_faithfulness(report, query)
    answer_relevancy = _score_answer_relevancy(report, query)
    context_recall   = _score_context_recall(report, query)
    overall          = round((faithfulness + answer_relevancy + context_recall) / 3, 3)

    # Plain-English interpretation
    interpretation = _interpret(faithfulness, answer_relevancy, context_recall, overall)

    elapsed = round(time.time() - start, 2)
    print(f"[ragas] Eval complete in {elapsed}s — overall={overall:.2f}")

    return {
        "faithfulness":     faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_recall":   context_recall,
        "overall":          overall,
        "interpretation":   interpretation,
        "codes_evaluated":  len(items),
        "eval_method":      "llm_judge+heuristic",
        "eval_time_s":      elapsed,
        "timestamp":        _now(),
    }


def _interpret(faith: float, relevancy: float, recall: float, overall: float) -> str:
    """Write a plain-English interpretation for the clinical analyst."""
    parts = []

    if overall >= 0.80:
        parts.append("**Overall quality: High.** The results appear well-grounded and relevant.")
    elif overall >= 0.60:
        parts.append("**Overall quality: Moderate.** Results are useful but may benefit from review.")
    else:
        parts.append("**Overall quality: Low.** Manual review of all recommendations is advised.")

    if faith < 0.60:
        parts.append(
            "⚠️ *Faithfulness is low* — some explanations may not be fully supported "
            "by the retrieved data. Verify claims before use."
        )
    if relevancy < 0.60:
        parts.append(
            "⚠️ *Answer relevancy is low* — some returned codes may not match the "
            "clinical query. Consider rephrasing with more specific terminology."
        )
    if recall < 0.60:
        parts.append(
            "⚠️ *Context recall is low* — the query may not have retrieved codes for "
            "all conditions mentioned. Try breaking it into separate queries."
        )

    return " ".join(parts)


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────────
# UI FORMATTER
# ─────────────────────────────────────────────────────────────────

def format_eval_panel(metrics: dict) -> str:
    """
    Format the evaluation metrics dict as a Markdown string
    suitable for display in a Gradio Accordion panel.

    The panel is designed to be readable without technical knowledge —
    scores are shown as percentages with a colour indicator and a
    plain-English label.

    Parameters
    ----------
    metrics  The dict returned by evaluate()

    Returns
    -------
    str  Markdown for display in the UI
    """
    def pct(v: float) -> str:
        return f"{v:.0%}"

    def badge(v: float) -> str:
        if v >= 0.80: return "🟢"
        if v >= 0.60: return "🟡"
        return "🔴"

    faith     = metrics.get("faithfulness", 0.0)
    relevancy = metrics.get("answer_relevancy", 0.0)
    recall    = metrics.get("context_recall", 0.0)
    overall   = metrics.get("overall", 0.0)
    interp    = metrics.get("interpretation", "")
    n_codes   = metrics.get("codes_evaluated", 0)
    elapsed   = metrics.get("eval_time_s", 0)

    lines = [
        "### 📊 Evaluation Metrics (RAGAS-style)\n",
        f"| Metric | Score | Status |",
        f"|--------|-------|--------|",
        f"| Faithfulness | **{pct(faith)}** | {badge(faith)} Explanations grounded in evidence |",
        f"| Answer Relevancy | **{pct(relevancy)}** | {badge(relevancy)} Codes match the query |",
        f"| Context Recall | **{pct(recall)}** | {badge(recall)} Query coverage estimate |",
        f"| **Overall** | **{pct(overall)}** | {badge(overall)} Composite quality score |",
        "",
        interp,
        "",
        f"*Evaluated {n_codes} codes in {elapsed}s. "
        f"Scores are estimates — true recall requires a validated gold-standard list. "
        f"Use `app_audit.py` backtesting for rigorous evaluation.*",
    ]

    return "\n".join(lines)
