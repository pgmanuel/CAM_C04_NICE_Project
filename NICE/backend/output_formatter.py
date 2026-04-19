"""output_formatter.py — deterministic candidate formatter (v4 aligned).

Source: Playground.ipynb, Section 3 (cell LYfhl8463wNk).
Logic is unchanged from the notebook.

Key changes from original:
- Role descriptions updated to match v4 CE-first pipeline.
- Rationale now surfaces ce_score and rerank_score (in that priority).
- LLM call removed — output is purely deterministic.
"""

from __future__ import annotations

from typing import Any

BUCKET_ORDER = [
    "include_candidates",
    "review_candidates",
    "specific_variants",
    "suppressed_candidates",
]


class LLMExplanationFormatter:
    def __init__(self, config: Any):
        self.config = config

    @staticmethod
    def _deterministic_rationale(candidate: dict[str, Any]) -> str:
        evidence = candidate["evidence"]
        role     = candidate.get("candidate_role", "")
        ranking  = candidate.get("ranking_components", {})

        reasons = []

        # Role-aware explanation aligned with revised pipeline
        if role == "core":
            reasons.append("broad anchor concept selected after relevance filtering and reranking")
        elif role == "variant":
            reasons.append("relevant condition-linked concept kept for analyst review")
        elif role == "specific":
            reasons.append("more specific lower-priority concept retained as depth")
        elif role == "suppress":
            reasons.append("filtered from first-pass review but retained for traceability")

        # Reranker-aware signal — CE score takes priority over gate rerank score
        ce_score     = ranking.get("ce_score")
        rerank_score = ranking.get("rerank_score")

        if ce_score is not None and float(ce_score) > 0:
            reasons.append(f"cross-encoder score {float(ce_score):.3f}")
        elif rerank_score is not None and float(rerank_score) > 0:
            reasons.append(f"rerank score {float(rerank_score):.3f}")

        # Broadness / anchor signal
        anchor_bonus = ranking.get("anchor_bonus")
        if anchor_bonus is not None and float(anchor_bonus) > 0:
            reasons.append("favoured as a broader anchor candidate")

        # Evidence signal
        if evidence["in_qof"]:
            reasons.append("QOF-supported code")
        if evidence["in_opencodelists"]:
            reasons.append("present in OpenCodelists")
        if evidence["usage_count_nhs"] > 0:
            reasons.append(f"NHS usage count {int(evidence['usage_count_nhs'])}")

        if not reasons:
            reasons.append("retained after relevance filtering with limited supporting evidence")

        return "; ".join(reasons)

    def _serialize_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        return {
            "presentation_score": candidate.get("presentation_score"),
            "snomed_code":         candidate["snomed_code"],
            "term":                candidate["term"],
            "confidence_tier":     candidate["confidence_tier"],
            "candidate_role":      candidate.get("candidate_role"),
            "rationale":           self._deterministic_rationale(candidate),
            "evidence": {
                "in_qof":            candidate["evidence"]["in_qof"],
                "in_opencodelists":  candidate["evidence"]["in_opencodelists"],
                "usage_count_nhs":   candidate["evidence"]["usage_count_nhs"],
            },
        }

    def format_candidates(
        self,
        user_query: str,
        candidate_groups: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, Any]]:
        del user_query  # not used — deterministic mode

        grouped_output = {bucket: [] for bucket in BUCKET_ORDER}
        for bucket in BUCKET_ORDER:
            for c in candidate_groups.get(bucket, []):
                grouped_output[bucket].append(self._serialize_candidate(c))

        debug_info = {
            "formatter_enabled":        False,
            "mode":                     "deterministic_revised_formatter",
            "schema_validation_error":  None,
        }

        return grouped_output, "deterministic_revised_formatter", debug_info
