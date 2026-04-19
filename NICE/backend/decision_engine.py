"""decision_engine.py — CE-first, per-condition anchor decisioning.

Source: Playground.ipynb, Section 3 (cells PRbLMmcY4fCI + BYT92I_AClpo).
Logic is unchanged from the notebook.

FIX 2: Anchor selection enforces final_rerank_score >= 0.2.
FIX 3: Hard CE safety filter (ce_score < 0.2 → skip) as secondary belt-and-braces
        after ce_reranker.py has already applied the primary filter.
NOTE:   specificity_score is NOT computed here — it is metadata from HierarchyEnricher.
"""

from __future__ import annotations

import math
import re
from typing import Any


# Semantic tags that must never be selected as a broad anchor concept.
BLOCKED_CORE_TAGS = {
    "situation",
    "context-dependent category",
    "finding",
}


# ── Shared helpers (also imported by gate_reranker and ce_reranker) ───────────

def major_conditions(structured_query: dict[str, Any]) -> list[str]:
    """Return the ordered list of major conditions from a structured query."""
    raw = [structured_query.get("primary_condition", "")] + structured_query.get("secondary_conditions", [])
    seen: set[str] = set()
    result = []
    for c in raw:
        t = _normalize_query(c)
        if t and t.lower() not in seen:
            seen.add(t.lower())
            result.append(t)
    return result


def _normalize_query(text: str) -> str:
    return " ".join(str(text).strip().split())


# ── Confidence + authority helpers ────────────────────────────────────────────

def assign_confidence(candidate: dict[str, Any]) -> str:
    in_qof = bool(candidate.get("in_qof", False))
    in_oc = bool(candidate.get("in_opencodelists", False))
    usage = float(candidate.get("usage_count_nhs", 0.0))

    if in_qof:
        return "HIGH"
    if in_oc and usage >= 29097:
        return "HIGH"
    if in_oc or usage >= 6730:
        return "MEDIUM"
    return "REVIEW"


def compute_bounded_authority(candidate: dict[str, Any]) -> float:
    usage = float(candidate.get("usage_count_nhs", 0.0))
    in_qof = bool(candidate.get("in_qof", False))
    in_oc = bool(candidate.get("in_opencodelists", False))

    return min(
        1.0,
        0.40 * float(in_qof) +
        0.25 * float(in_oc) +
        min(0.35, math.log10(usage + 1.0) / 8.0)
    )


def is_bad_core_pattern(term: str, condition: str) -> bool:
    term_lower = term.lower()
    cond_lower = condition.lower()

    bad_phrases = [
        "due to",
        "caused by",
        "secondary to",
        "fear of",
        "history of",
        "suspected",
        "risk of",
        "exposure to",
        "family history",
    ]
    for phrase in bad_phrases:
        if phrase in term_lower and phrase not in cond_lower:
            return True

    if cond_lower == "hypertension" and "pulmonary" in term_lower:
        return True
    if cond_lower == "hypertension" and "portal" in term_lower:
        return True
    if cond_lower == "hypertension" and "ocular" in term_lower:
        return True

    return False


def is_weak_suppress_pattern(term: str, condition: str) -> bool:
    term_lower = term.lower()
    cond_lower = condition.lower()

    weak_phrases = ["fear of", "worried about", "anxiety about"]
    for phrase in weak_phrases:
        if phrase in term_lower and phrase not in cond_lower:
            return True
    return False


def compute_anchor_score(candidate: dict[str, Any], condition: str) -> float:
    """Score how well this concept serves as a broad canonical anchor."""
    term = str(candidate.get("term", "")).lower()
    cond = condition.lower()

    term_tokens = set(re.findall(r"\b\w+\b", term))
    cond_tokens = set(re.findall(r"\b\w+\b", cond))

    if not cond_tokens:
        return 0.0

    stop_words = {"disorder", "finding", "disease", "syndrome"}
    clean_term_tokens = term_tokens - stop_words

    if clean_term_tokens == cond_tokens:
        return 1.0

    extra_tokens = clean_term_tokens - cond_tokens
    specificity_penalty = len(extra_tokens) * 0.15
    overlap = len(clean_term_tokens & cond_tokens) / len(cond_tokens)

    score = overlap - specificity_penalty
    return max(0.0, min(1.0, score))


# ── Output serializer ─────────────────────────────────────────────────────────

def serialize_candidate_output(row: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    payload = {
        "presentation_score": float(row.get("presentation_score", 0.0)),
        "snomed_code": row["snomed_code"],
        "term": row["term"],
        "semantic_tag": row.get("semantic_tag", ""),
        "confidence_tier": row["confidence_tier"],
        "candidate_role": row.get("candidate_role"),
        "evidence": {
            "in_qof": bool(row["in_qof"]),
            "in_opencodelists": bool(row["in_opencodelists"]),
            "usage_count_nhs": float(row["usage_count_nhs"]),
        },
        "retrieval_features": {
            "fusion_score": float(row.get("fusion_score", 0.0)),
            "rrf_score": float(row.get("rrf_score", row.get("fusion_score", 0.0))),
            "semantic_score_max": float(row.get("semantic_score_max", row.get("semantic_score", 0.0))),
            "bm25_score_max": float(row.get("bm25_score_max", row.get("bm25_score", 0.0))),
            "lexical_overlap_max": float(row.get("lexical_overlap_max", row.get("lexical_overlap", 0.0))),
            "term_precision_max": float(row.get("term_precision_max", row.get("term_precision", 0.0))),
            "specificity_score_max": float(row.get("specificity_score_max", row.get("specificity_score", 0.0))),
            "query_coverage_count": int(row.get("query_coverage_count", 0)),
        },
        "reranker_features": {
            "rerank_score": float(row.get("rerank_score", 0.0)),
            "relevance_score": float(row.get("relevance_score", 0.0)),
            "condition_relevance": row.get("condition_relevance", {}),
            "ce_score": float(row.get("ce_score", 0.0)),
            "ce_score_by_condition": row.get("ce_score_by_condition", {}),
            "matched_conditions_from_ce": row.get("matched_conditions_from_ce", []),
            "dominant_condition_from_ce": row.get("dominant_condition_from_ce"),
        },
        "matched_conditions": row.get("matched_conditions", []),
        "dominant_condition": row.get("dominant_condition", "unassigned"),
        "ranking_components": row.get("ranking_components", {}),
        "retrieval_method": row.get("retrieval_method", "direct"),
        "hierarchy_parent_code": row.get("hierarchy_parent_code"),
        "hierarchy_parent_term": row.get("hierarchy_parent_term"),
        "retrieval_trace": row.get("retrieval_trace", []),
    }

    if index is not None:
        payload["presentation_rank"] = index

    return payload


# ── Decision Engine ───────────────────────────────────────────────────────────

class DecisionEngine:
    def assign_final_decisions(
        self,
        reranked_candidates: list[dict[str, Any]],
        structured_query: dict[str, Any],
        top_k: int,
    ) -> dict[str, list[dict[str, Any]]]:

        conditions = major_conditions(structured_query)
        multimorbidity = len(conditions) > 1

        # ── Stage 1: scoring ─────────────────────────────────────────────────
        enriched = []

        for candidate in reranked_candidates:
            ce_score = float(candidate.get("ce_score", 0.0))
            rerank_score = float(candidate.get("rerank_score", 0.0))
            relevance_score = float(candidate.get("relevance_score", 0.0))

            final_rerank_score = (
                0.70 * ce_score + 0.30 * rerank_score
            ) if ce_score > 0 else rerank_score

            # FIX 3: HARD CE QUALITY FILTER — secondary safeguard
            # (primary filter already applied in ce_reranker.py)
            if ce_score < 0.2:
                continue

            if final_rerank_score <= 0:
                continue

            matched = list(candidate.get("matched_conditions_from_ce", [])) or list(
                candidate.get("matched_conditions_from_gate", [])
            )

            if not matched:
                continue

            dominant = (
                candidate.get("dominant_condition_from_ce")
                or candidate.get("dominant_condition_from_gate")
                or "unassigned"
            )

            authority = compute_bounded_authority(candidate)

            # scoring components
            coverage_bonus = (
                min(0.20, 0.08 * len(matched))
                if multimorbidity else
                min(0.10, 0.05 * len(matched))
            )

            specificity = float(candidate.get("specificity_score_max", 0.0))
            anchor_bonus = (
                0.10 if specificity >= 0.70 else
                0.05 if specificity >= 0.50 else
                0.0
            )

            presentation_score = (
                0.65 * final_rerank_score +
                0.20 * authority +
                0.10 * coverage_bonus +
                0.05 * anchor_bonus
            )

            ranking_components = {
                "ce_score": round(ce_score, 6),
                "rerank_score": round(rerank_score, 6),
                "final_rerank_score": round(final_rerank_score, 6),
                "relevance_score": round(relevance_score, 6),
                "authority_component": round(authority, 6),
                "coverage_component": round(coverage_bonus, 6),
                "anchor_bonus": round(anchor_bonus, 6),
            }

            d = dict(candidate)
            d["confidence_tier"] = assign_confidence(candidate)
            d["matched_conditions"] = matched
            d["dominant_condition"] = dominant
            d["final_rerank_score"] = final_rerank_score
            d["authority_score"] = authority
            d["presentation_score"] = round(presentation_score, 6)
            d["ranking_components"] = ranking_components

            enriched.append(d)

        # ── Stage 2: group per condition ──────────────────────────────────────
        by_condition: dict[str, list[dict[str, Any]]] = {c: [] for c in conditions}

        for c in enriched:
            for cond in c["matched_conditions"]:
                if cond in by_condition:
                    by_condition[cond].append(c)

        include, review, specific, suppressed = [], [], [], []
        selected_codes: set[str] = set()

        # ── Stage 3: strict anchor selection (1 per condition) ────────────────
        for condition in conditions:
            candidates = by_condition.get(condition, [])
            if not candidates:
                continue

            for c in candidates:
                c["_anchor_score"] = compute_anchor_score(c, condition)
                c["_is_bad_core"] = is_bad_core_pattern(c.get("term", ""), condition)
                c["_is_weak_suppress"] = is_weak_suppress_pattern(c.get("term", ""), condition)

            candidates = sorted(
                candidates,
                key=lambda x: (
                    -x.get("_anchor_score", 0.0),
                    -x["ranking_components"]["ce_score"],
                    -x["authority_score"],
                )
            )

            anchor = None

            for c in candidates:
                tag = str(c.get("semantic_tag", "")).lower()

                # FIX 2: anchor constraint — blocked tags + final_rerank_score >= 0.2
                if (
                    tag not in BLOCKED_CORE_TAGS and
                    c["final_rerank_score"] >= 0.2 and
                    not c.get("_is_bad_core", False) and
                    not c.get("_is_weak_suppress", False)
                ):
                    anchor = c
                    break

            if anchor:
                anchor["candidate_role"] = "core"
                include.append(anchor)
                selected_codes.add(anchor["snomed_code"])

            remaining = sorted(
                [c for c in candidates if c["snomed_code"] not in selected_codes],
                key=lambda x: -x["presentation_score"],
            )

            for c in remaining:
                if c["snomed_code"] in selected_codes:
                    continue

                if c.get("_is_weak_suppress", False):
                    c["candidate_role"] = "suppress"
                    suppressed.append(c)
                    selected_codes.add(c["snomed_code"])
                    continue

                tag = str(c.get("semantic_tag", "")).lower()

                if tag in BLOCKED_CORE_TAGS or c.get("_is_bad_core", False):
                    c["candidate_role"] = "specific"
                    specific.append(c)
                else:
                    c["candidate_role"] = "variant"
                    review.append(c)

                selected_codes.add(c["snomed_code"])

        # ── Stage 4: deduplicate + cap ─────────────────────────────────────────
        def serialize_list(rows: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
            out = []
            seen: set[str] = set()

            rows = sorted(rows, key=lambda x: -x["presentation_score"])

            for i, r in enumerate(rows, start=1):
                code = r["snomed_code"]
                if code in seen:
                    continue

                seen.add(code)
                out.append(serialize_candidate_output(r, index=i))

                if len(out) >= cap:
                    break

            return out

        return {
            "include_candidates": serialize_list(include, len(conditions)),
            "review_candidates": serialize_list(review, 5),
            "specific_variants": serialize_list(specific, 5),
            "suppressed_candidates": serialize_list(suppressed, 5),
        }
