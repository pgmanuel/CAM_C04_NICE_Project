"""ce_reranker.py — neural cross-encoder reranker using BAAI/bge-reranker-v2-m3.

Source: Playground.ipynb, Section 3 (cell xQFdKRjRsHnC).
Logic is unchanged from the notebook.
FIX 1: Hard CE filter (ce_condition_score < 0.2 → skip) is the PRIMARY enforcement
        point — candidates are dropped here before reaching DecisionEngine.
"""

from __future__ import annotations

from typing import Any

import torch
from sentence_transformers import CrossEncoder


def _normalize_query(text: str) -> str:
    return " ".join(str(text).strip().split())


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        top_n_per_condition: int = 20,
        batch_size: int = 16,
        max_length: int = 512,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.top_n_per_condition = top_n_per_condition
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                device=self.device,
                trust_remote_code=True,
            )
        return self._model

    @staticmethod
    def _norm(text: str) -> str:
        return _normalize_query(text).lower()

    @staticmethod
    def _candidate_text(candidate: dict[str, Any]) -> str:
        """Minimal, clean clinical text for CE.
        No evidence, no usage, no admin noise.
        """
        term = str(candidate.get("term", "")).strip()
        semantic_tag = str(candidate.get("semantic_tag", "")).strip()
        return f"{term} ({semantic_tag})".strip()

    def _score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        model = self._get_model()
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def rerank(
        self,
        gate_candidates: list[dict[str, Any]],
        structured_query: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Input: output of RelevanceGateReranker.
        Output: reranked candidates using CE per condition ONLY.
        """
        from decision_engine import major_conditions
        conditions = major_conditions(structured_query)

        if not conditions or not gate_candidates:
            return gate_candidates

        # ── Build per-condition candidate pools ──────────────────────────────
        candidates_by_condition: dict[str, list[dict[str, Any]]] = {}

        for condition in conditions:
            branch = [
                c for c in gate_candidates
                if condition in c.get("matched_conditions_from_gate", [])
            ]

            branch.sort(
                key=lambda x: (
                    x.get("condition_relevance", {}).get(condition, {}).get("rerank_score", 0.0),
                    x.get("condition_relevance", {}).get(condition, {}).get("relevance_score", 0.0),
                    x.get("fusion_score", 0.0),
                ),
                reverse=True,
            )

            candidates_by_condition[condition] = branch[:self.top_n_per_condition]

        # ── Cross-encoder scoring (STRICTLY per condition) ───────────────────
        scored_by_code: dict[str, dict[str, Any]] = {}

        for condition in conditions:
            branch = candidates_by_condition.get(condition, [])
            if not branch:
                continue

            candidate_texts = [self._candidate_text(c) for c in branch]
            condition_pairs = [(condition, text) for text in candidate_texts]
            condition_scores = self._score_pairs(condition_pairs)

            for candidate, ce_condition_score in zip(branch, condition_scores):
                # FIX 1: HARD CE FILTER — primary enforcement point (before DecisionEngine)
                if ce_condition_score < 0.2:
                    continue

                code = str(candidate["snomed_code"])
                enriched = dict(candidate)

                ce_score_by_condition = dict(enriched.get("ce_score_by_condition", {}))
                ce_score_by_condition[condition] = {
                    "ce_condition_score": round(ce_condition_score, 6),
                    "ce_score": round(ce_condition_score, 6),
                }

                enriched["ce_score_by_condition"] = ce_score_by_condition

                # Keep best CE view per code
                prev = scored_by_code.get(code)

                if prev is None:
                    scored_by_code[code] = enriched
                else:
                    prev_best = float(prev.get("ce_score", float("-inf")))

                    if ce_condition_score > prev_best:
                        merged = dict(enriched)
                        merged_scores = dict(prev.get("ce_score_by_condition", {}))
                        merged_scores.update(ce_score_by_condition)
                        merged["ce_score_by_condition"] = merged_scores
                        scored_by_code[code] = merged
                    else:
                        prev_scores = dict(prev.get("ce_score_by_condition", {}))
                        prev_scores.update(ce_score_by_condition)
                        prev["ce_score_by_condition"] = prev_scores

        # ── Final aggregation ────────────────────────────────────────────────
        final_candidates: list[dict[str, Any]] = []

        for code, candidate in scored_by_code.items():
            ce_score_by_condition = candidate.get("ce_score_by_condition", {})

            matched_conditions_from_ce = [
                condition for condition, scores in ce_score_by_condition.items()
                if float(scores.get("ce_score", 0.0)) >= 0.2
            ]

            if not matched_conditions_from_ce:
                matched_conditions_from_ce = list(candidate.get("matched_conditions_from_gate", []))

            if matched_conditions_from_ce:
                dominant_condition_from_ce = max(
                    matched_conditions_from_ce,
                    key=lambda c: float(ce_score_by_condition.get(c, {}).get("ce_score", float("-inf")))
                )
            else:
                dominant_condition_from_ce = candidate.get("dominant_condition_from_gate", "unassigned")

            best_ce = max(
                [float(v.get("ce_score", float("-inf"))) for v in ce_score_by_condition.values()],
                default=0.0,
            )

            enriched = dict(candidate)
            enriched["ce_score"] = round(best_ce, 6)
            enriched["matched_conditions_from_ce"] = matched_conditions_from_ce
            enriched["dominant_condition_from_ce"] = dominant_condition_from_ce

            final_candidates.append(enriched)

        # ── Final sort ───────────────────────────────────────────────────────
        final_candidates.sort(
            key=lambda x: (
                x.get("ce_score", 0.0),
                x.get("rerank_score", 0.0),
                x.get("relevance_score", 0.0),
                x.get("fusion_score", 0.0),
                x.get("specificity_score_max", 0.0),
            ),
            reverse=True,
        )

        return final_candidates
