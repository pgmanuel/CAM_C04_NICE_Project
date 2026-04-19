"""
reasoning_eval.py
-----------------
Reasoning Trace Generator — step-by-step audit of why each code appeared.

PURPOSE
-------
Every recommendation needs to be explainable. This module produces a
structured, step-by-step account of how the pipeline reached each result:

  STEP 1 — What the query was decomposed into
  STEP 2 — Which sub-queries were searched and why
  STEP 3 — What signals contributed to each code's score
  STEP 4 — How the hybrid formula weighted those signals
  STEP 5 — What the LLM said, and whether it stuck to the evidence

This trace serves two audiences:
  - Clinical analysts who want to understand why a specific code appeared
  - The governance/audit function who need a traceable decision record

HOW IT INTEGRATES WITH THE UI
app.py calls generate_reasoning_trace(report, query) and displays
the result in a collapsible Accordion panel labelled "🔍 Reasoning Trace".
The analyst can expand it to see the full trace, or ignore it if they
only need the ranked codes.

The trace is also saved by app_audit.py when audit logging is enabled,
providing a persistent record alongside the run log JSON.

Usage
-----
    from reasoning_eval import generate_reasoning_trace, generate_score_breakdown

    report = pipeline.retrieve_and_rank("obesity with type 2 diabetes")
    trace  = generate_reasoning_trace(report, "obesity with type 2 diabetes")
    print(trace)

    breakdown = generate_score_breakdown(report["items"][0])
    print(breakdown)
"""

from __future__ import annotations

from datetime import datetime, timezone


# ─────────────────────────────────────────────────────────────────
# MAIN TRACE FUNCTION
# ─────────────────────────────────────────────────────────────────

def generate_reasoning_trace(report: dict, query: str) -> str:
    """
    Produce a step-by-step Markdown trace of the full pipeline run.

    This is the document a clinical reviewer reads to understand
    the complete chain of decisions from query to recommendation.

    Parameters
    ----------
    report  The dict returned by pipeline.retrieve_and_rank() —
            must include items, sub_queries, primary_condition,
            comorbidities
    query   The original user query string

    Returns
    -------
    str  Multi-section Markdown document
    """
    sections = []

    # ── Header ────────────────────────────────────────────────────
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    sections.append(
        f"## 🔍 Reasoning Trace\n"
        f"*Generated {ts} for query: \"{query}\"*\n"
    )

    # ── Step 1: Query Decomposition ───────────────────────────────
    primary      = report.get("primary_condition", query)
    comorbids    = report.get("comorbidities", [])
    sub_queries  = report.get("sub_queries", [])

    sections.append("### Step 1 — Query Decomposition")
    sections.append(
        "The query was parsed by a local language model (phi4:mini) to "
        "identify distinct clinical concepts. Searching each concept "
        "separately improves recall compared to searching the full query "
        "as a single string."
    )
    sections.append(f"- **Primary condition:** {primary}")
    if comorbids:
        sections.append(f"- **Comorbidities identified:** {', '.join(comorbids)}")
    else:
        sections.append("- **Comorbidities:** none identified")
    sections.append("")

    # ── Step 2: Sub-query Search ───────────────────────────────────
    sections.append("### Step 2 — Semantic Search (ChromaDB)")
    sections.append(
        "Each sub-query was converted into a 384-dimensional vector by the "
        "BGE embedding model (BAAI/bge-small-en), then used to search the "
        "ChromaDB SNOMED index. The 15 nearest codes per sub-query were "
        "retrieved by cosine distance. Duplicates were removed, keeping the "
        "closest match for each code."
    )
    if sub_queries:
        for sq in sub_queries:
            # Count how many items were found by this sub-query
            found = [
                item for item in report.get("items", [])
                if item.get("sub_query_found", "").lower() == sq.lower()
            ]
            sections.append(f"- `{sq}` → {len(found)} code(s) in final output")
    sections.append("")

    # ── Step 3: Cross-Encoder Reranking ───────────────────────────
    sections.append("### Step 3 — Cross-Encoder Reranking")
    sections.append(
        "The merged candidate list was reranked using a CrossEncoder model "
        "(BAAI/bge-reranker-base). Unlike the embedding search (which scores "
        "query and code independently), the CrossEncoder reads both together, "
        "giving a more accurate relevance score at the cost of speed. "
        "This score becomes the semantic signal in the hybrid formula."
    )
    items = report.get("items", [])
    if items:
        top = items[0]
        sections.append(
            f"\nHighest CrossEncoder score: **{top.get('semantic_score', 0):.3f}** "
            f"— {top.get('term', '')} ({top.get('code', '')})"
        )
    sections.append("")

    # ── Step 4: Hybrid Scoring Formula ────────────────────────────
    sections.append("### Step 4 — Hybrid Scoring")
    sections.append(
        "Each code's final score combines three signals:\n\n"
        "```\n"
        "hybrid_score = (0.7 × semantic_score)\n"
        "             + (0.2 × usage_count / max_usage_in_results)\n"
        "             + (0.1 × qof_bonus)  ← 0.1 if QOF-mandated, else 0\n"
        "```\n\n"
        "The semantic weight (70%) is highest because clinical relevance "
        "to the query is the primary selection criterion. Usage frequency (20%) "
        "ensures mainstream clinical practice is represented. The QOF bonus "
        "(10%) prioritises codes that NHS policy requires GPs to record."
    )
    sections.append("\n**Score breakdown for top results:**\n")
    sections.append("| Rank | Code | Term | Semantic | Usage | QOF | Hybrid |")
    sections.append("|------|------|------|----------|-------|-----|--------|")
    for item in items[:5]:
        sem    = item.get("semantic_score", 0.0)
        hybrid = item.get("confidence_score", 0.0)
        qof    = "✓ 10%" if item.get("in_qof") else "—"
        usage  = f"{item.get('usage_count', 0):,}" if item.get("usage_count") else "—"
        sections.append(
            f"| #{item.get('rank', '?')} "
            f"| `{item.get('code', '')}` "
            f"| {item.get('term', '')[:35]} "
            f"| {sem:.3f} "
            f"| {usage} "
            f"| {qof} "
            f"| **{hybrid:.3f}** |"
        )
    sections.append("")

    # ── Step 5: LLM Explanation Layer ─────────────────────────────
    sections.append("### Step 5 — LLM Explanation")
    sections.append(
        "After ranking, the selected LLM wrote a plain-English explanation "
        "for each code. The LLM was explicitly instructed not to add new "
        "codes or change rankings — its only role is to translate the "
        "clinical significance of each code into language an analyst can "
        "read and act on."
    )
    if items:
        sections.append("\n**Sample explanations:**\n")
        for item in items[:3]:
            expl = item.get("explanation", "No explanation generated.")
            sections.append(f"- **{item.get('term', '')}:** {expl}")
    sections.append("")

    # ── Summary ───────────────────────────────────────────────────
    n_qof = sum(1 for item in items if item.get("in_qof"))
    sections.append("### Summary")
    sections.append(
        f"- **{len(items)} codes** returned from ChromaDB\n"
        f"- **{n_qof}** are QOF-mandated (highest clinical authority)\n"
        f"- **{len(sub_queries)} sub-queries** searched\n"
        f"- Top hybrid score: **{items[0].get('confidence_score', 0):.1%}** "
        f"({items[0].get('term', '')})" if items else "- No results"
    )

    return "\n".join(sections)


# ─────────────────────────────────────────────────────────────────
# PER-CODE SCORE BREAKDOWN
# ─────────────────────────────────────────────────────────────────

def generate_score_breakdown(item: dict) -> str:
    """
    Generate a single-code score breakdown as Markdown.

    Used when the analyst wants to inspect exactly why one specific
    code scored the way it did.

    Parameters
    ----------
    item  One item dict from report["items"]

    Returns
    -------
    str  Short Markdown explanation of the scoring
    """
    code    = item.get("code", "?")
    term    = item.get("term", "Unknown")
    sem     = item.get("semantic_score", 0.0)
    hybrid  = item.get("confidence_score", 0.0)
    in_qof  = item.get("in_qof", False)
    usage   = item.get("usage_count", 0)
    sub_q   = item.get("sub_query_found", "unknown")

    lines = [
        f"**`{code}` — {term}**\n",
        f"- Found by sub-query: `{sub_q}`",
        f"- CrossEncoder semantic score: `{sem:.3f}`",
        f"- NHS annual usage: {usage:,}" if usage else "- NHS annual usage: not recorded",
        f"- QOF mandated: {'Yes (10% bonus applied)' if in_qof else 'No'}",
        f"- **Final hybrid score: {hybrid:.3f}** ({hybrid:.0%})",
    ]

    return "\n".join(lines)
