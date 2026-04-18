"""hierarchy_enricher.py — zero-trust child-concept recall via SNOMED hierarchy edges.

Source: Playground.ipynb, Section 3 (cell 1Par1LLIZLxC).
Logic is unchanged from the notebook.
"""

from __future__ import annotations

import math
import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger("IntegratedPipeline")


class HierarchyEnricher:
    def __init__(self, snomed_path: str | Path, edge_path: str | Path):
        self.snomed_path = Path(snomed_path)
        self.edge_path = Path(edge_path)
        self._master = None
        self._master_by_code = None
        self._edge = None

    def _load(self) -> None:
        if self._master is None:
            master = pd.read_csv(self.snomed_path, dtype=str)
            master["snomed_code"] = master["snomed_code"].astype(str).str.strip()
            master["term"] = master["term"].astype(str).str.strip()
            master["semantic_tag"] = master.get("semantic_tag", "").fillna("").astype(str)

            for col in ["num_parents", "num_children"]:
                if col in master.columns:
                    master[col] = pd.to_numeric(master[col], errors="coerce").fillna(0).astype(int)
                else:
                    master[col] = 0

            for col in ["in_qof", "in_opencodelists"]:
                if col in master.columns:
                    master[col] = master[col].map(
                        {"True": True, "False": False, True: True, False: False}
                    ).fillna(False)
                else:
                    master[col] = False

            if "usage_count_nhs" in master.columns:
                master["usage_count_nhs"] = pd.to_numeric(
                    master["usage_count_nhs"], errors="coerce"
                ).fillna(0.0)
            else:
                master["usage_count_nhs"] = 0.0

            self._master = master
            self._master_by_code = master.set_index("snomed_code", drop=False)

        if self._edge is None:
            edge = pd.read_csv(self.edge_path, dtype=str)
            edge["child_code"] = edge["child_code"].astype(str).str.strip()
            edge["parent_code"] = edge["parent_code"].astype(str).str.strip()
            edge["child_term"] = edge["child_term"].astype(str).str.strip()
            edge["parent_term"] = edge["parent_term"].astype(str).str.strip()
            self._edge = edge

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(str(text).strip().split())

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        import re
        stop = {
            "a", "an", "and", "are", "as", "at", "by", "for", "from",
            "in", "of", "on", "or", "the", "to", "with",
        }
        return {
            t for t in re.findall(r"[a-z0-9]+", str(text).lower())
            if t and t not in stop and len(t) > 1
        }

    @classmethod
    def _lexical_overlap(cls, query_text: str, candidate_term: str) -> float:
        q = cls._tokenize(query_text)
        c = cls._tokenize(candidate_term)
        if not q or not c:
            return 0.0
        return len(q & c) / max(1, len(q))

    @classmethod
    def _term_precision(cls, query_text: str, candidate_term: str) -> float:
        q = cls._tokenize(query_text)
        c = cls._tokenize(candidate_term)
        if not q or not c:
            return 0.0
        return len(q & c) / max(1, len(c))

    @classmethod
    def _specificity_score(
        cls,
        candidate_term: str,
        condition_str: str,
        num_parents: int,
        num_children: int,
    ) -> float:
        stop_tokens = {"disorder", "finding", "observable", "entity", "situation"}
        extra_tokens = cls._tokenize(candidate_term) - cls._tokenize(condition_str) - stop_tokens
        depth_penalty = min(1.0, num_parents / 8.0)
        extra_penalty = min(1.0, len(extra_tokens) / 5.0)
        breadth_bonus = min(0.3, math.log1p(num_children) / 10.0)
        score = 1.0 - 0.45 * depth_penalty - 0.55 * extra_penalty + breadth_bonus
        return max(0.0, min(1.0, score))

    def get_concept_info(self, code: str) -> dict[str, Any] | None:
        self._load()
        code = str(code).strip()
        if code not in self._master_by_code.index:
            return None
        row = self._master_by_code.loc[code]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row.to_dict()

    def get_children(self, parent_code: str) -> list[dict[str, Any]]:
        self._load()
        children = self._edge.loc[
            self._edge["parent_code"] == str(parent_code).strip(),
            ["child_code", "child_term", "parent_code", "parent_term"],
        ].drop_duplicates()
        return children.to_dict(orient="records")

    def enrich_batch(
        self,
        batch: list[dict[str, Any]],
        top_n_parents_per_focus: int = 1,
        max_children_per_parent: int = 3,
        min_child_overlap: float = 0.10,
    ) -> list[dict[str, Any]]:
        """Adds child candidates as zero-trust recall candidates.
        They do not inherit parent score.
        They must pass a small lexical check against clinical_focus.
        """
        self._load()

        if not batch:
            return batch

        enriched = list(batch)
        seen_codes = {str(c["snomed_code"]).strip() for c in batch}

        # group by clinical_focus so multimorbidity does not get hijacked by one condition
        by_focus: dict[str, list[dict[str, Any]]] = {}
        for c in batch:
            focus = self._normalize(c.get("clinical_focus", ""))
            by_focus.setdefault(focus, []).append(c)

        for focus, focus_batch in by_focus.items():
            # direct retrieved candidates only
            direct_candidates = [
                c for c in focus_batch
                if c.get("retrieval_method") != "hierarchy_child"
            ]

            ranked_parents = sorted(
                direct_candidates,
                key=lambda x: float(x.get("adjusted_retrieval_score", 0.0)),
                reverse=True,
            )

            parents_taken = 0
            for parent_candidate in ranked_parents:
                if parents_taken >= top_n_parents_per_focus:
                    break

                parent_code = str(parent_candidate["snomed_code"]).strip()
                parent_info = self.get_concept_info(parent_code)
                if not parent_info:
                    continue

                if int(parent_info.get("num_children", 0) or 0) <= 0:
                    continue

                parents_taken += 1
                children = self.get_children(parent_code)[:max_children_per_parent]

                for child in children:
                    child_code = str(child["child_code"]).strip()
                    if child_code in seen_codes:
                        continue

                    child_info = self.get_concept_info(child_code)
                    if not child_info:
                        continue

                    child_term = str(child_info.get("term", child["child_term"]))
                    lex = self._lexical_overlap(focus, child_term)
                    if lex < min_child_overlap:
                        continue

                    prec = self._term_precision(focus, child_term)
                    spec = self._specificity_score(
                        child_term,
                        focus,
                        int(child_info.get("num_parents", 0) or 0),
                        int(child_info.get("num_children", 0) or 0),
                    )

                    enriched.append({
                        "snomed_code": child_code,
                        "term": child_term,
                        "semantic_tag": str(child_info.get("semantic_tag", "")),
                        "in_qof": bool(child_info.get("in_qof", False)),
                        "in_opencodelists": bool(child_info.get("in_opencodelists", False)),
                        "usage_count_nhs": float(child_info.get("usage_count_nhs", 0.0) or 0.0),
                        "num_parents": int(child_info.get("num_parents", 0) or 0),
                        "num_children": int(child_info.get("num_children", 0) or 0),
                        "query_text": parent_candidate["query_text"],
                        "query_type": parent_candidate["query_type"],
                        "clinical_focus": focus,
                        "query_weight": parent_candidate.get("query_weight", 1.0),

                        # zero inherited trust
                        "semantic_score": 0.0,
                        "bm25_score": 0.0,
                        "retrieval_score": 0.0,
                        "adjusted_retrieval_score": 0.0,
                        "weighted_retrieval_score": 0.0,

                        # useful metadata only
                        "lexical_overlap": float(lex),
                        "term_precision": float(prec),
                        "specificity_score": float(spec),
                        "semantic_tag_weight": 0.0,

                        "retrieval_method": "hierarchy_child",
                        "hierarchy_parent_code": parent_code,
                        "hierarchy_parent_term": parent_candidate["term"],
                    })
                    seen_codes.add(child_code)

        return enriched
