"""gate_reranker.py — per-condition relevance gate reranker.

Source: Playground.ipynb, Section 3 (cell ufRJRflO3wC2).
Logic is unchanged from the notebook.
Gate threshold: 0.20 (corrected from notebook's 0.10 per v2 plan approval).
"""

from __future__ import annotations

from typing import Any


def _normalize_query(text: str) -> str:
    return " ".join(str(text).strip().split())


class RelevanceGateReranker:
    def __init__(
        self,
        relevance_threshold: float = 0.20,  # corrected: v2 plan sets this to 0.20
        top_n_per_condition: int = 50,
    ):
        self.relevance_threshold = relevance_threshold
        self.top_n_per_condition = top_n_per_condition

    @staticmethod
    def _norm(text: str) -> str:
        return _normalize_query(text).lower()

    def _per_condition_scores(
        self,
        candidate: dict[str, Any],
        condition: str,
    ) -> dict[str, float]:
        """Score one candidate against one bare condition only.
        Relevance stays primary. Specificity only scales it modestly.
        """
        condition_n = self._norm(condition)

        semantic = 0.0
        bm25 = 0.0
        term_precision = 0.0
        specificity = float(candidate.get("specificity_score_max", 0.0) or 0.0)

        for trace in candidate.get("retrieval_trace", []):
            focus = self._norm(trace.get("clinical_focus", ""))
            if focus != condition_n:
                continue

            semantic = max(semantic, float(trace.get("semantic_score", 0.0) or 0.0))
            bm25 = max(bm25, float(trace.get("bm25_score", 0.0) or 0.0))
            term_precision = max(term_precision, float(trace.get("term_precision", 0.0) or 0.0))
            specificity = max(specificity, float(trace.get("specificity_score", 0.0) or 0.0))

        # Base relevance: semantic + BM25 + term precision
        relevance = (
            0.50 * semantic +
            0.30 * bm25 +
            0.20 * term_precision
        )

        # Specificity is a bounded modifier, not a free bonus
        rerank_score = relevance * (0.60 + 0.40 * specificity)

        return {
            "semantic": round(semantic, 6),
            "bm25": round(bm25, 6),
            "term_precision": round(term_precision, 6),
            "specificity_score": round(specificity, 6),
            "relevance_score": round(relevance, 6),
            "rerank_score": round(rerank_score, 6),
        }

    def rerank(
        self,
        fused_candidates: list[dict[str, Any]],
        structured_query: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Returns a flat reranked shortlist, but builds it per condition first."""
        from decision_engine import major_conditions
        conditions = major_conditions(structured_query)
        if not conditions:
            return []

        scored: list[dict[str, Any]] = []

        # Score every candidate against every condition
        for candidate in fused_candidates:
            per_condition = {
                condition: self._per_condition_scores(candidate, condition)
                for condition in conditions
            }

            matched_conditions = [
                condition
                for condition, scores in per_condition.items()
                if scores["relevance_score"] >= self.relevance_threshold
            ]

            if not matched_conditions:
                continue

            dominant_condition = max(
                matched_conditions,
                key=lambda c: per_condition[c]["rerank_score"]
            )

            d = dict(candidate)
            d["condition_relevance"] = per_condition
            d["matched_conditions_from_gate"] = matched_conditions
            d["dominant_condition_from_gate"] = dominant_condition
            d["relevance_score"] = per_condition[dominant_condition]["relevance_score"]
            d["rerank_score"] = per_condition[dominant_condition]["rerank_score"]
            scored.append(d)

        # Build shortlist per condition first
        shortlisted_by_condition: dict[str, list[dict[str, Any]]] = {}
        for condition in conditions:
            branch = [
                c for c in scored
                if condition in c["matched_conditions_from_gate"]
            ]
            branch.sort(
                key=lambda x: (
                    x["condition_relevance"][condition]["rerank_score"],
                    x["condition_relevance"][condition]["relevance_score"],
                    x.get("fusion_score", 0.0),
                ),
                reverse=True,
            )
            shortlisted_by_condition[condition] = branch[:self.top_n_per_condition]

        # Flatten after per-condition ranking
        # Keep best version of each code by rerank score
        best_by_code: dict[str, dict[str, Any]] = {}
        for condition in conditions:
            for candidate in shortlisted_by_condition[condition]:
                code = str(candidate["snomed_code"])
                if (
                    code not in best_by_code
                    or candidate["rerank_score"] > best_by_code[code]["rerank_score"]
                ):
                    best_by_code[code] = candidate

        final_candidates = list(best_by_code.values())
        final_candidates.sort(
            key=lambda x: (
                x["rerank_score"],
                x["relevance_score"],
                x.get("fusion_score", 0.0),
                x.get("specificity_score_max", 0.0),
            ),
            reverse=True,
        )

        return final_candidates
