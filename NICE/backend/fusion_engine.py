"""fusion_engine.py — RRF-based candidate fusion.

Source: Playground.ipynb, Section 3 (cell cE_h1jvEZU9S).
Logic is unchanged from the notebook.

Key changes from original:
- Replaced manual score blending with RRF (Rank Reciprocal Fusion).
- Richer fused item fields: num_parents/num_children, has_direct/hierarchy flags,
  specificity_score_max, retrieval_trace.
"""

from __future__ import annotations

from typing import Any


class CandidateFusionEngine:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(self, retrieval_batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        for batch_index, batch in enumerate(retrieval_batches):
            if not batch:
                continue

            for rank, c in enumerate(batch, start=1):
                code = str(c["snomed_code"]).strip()
                item = fused.setdefault(code, {
                    "snomed_code": code,
                    "term": c["term"],
                    "semantic_tag": c.get("semantic_tag", ""),
                    "in_qof": c.get("in_qof", False),
                    "in_opencodelists": c.get("in_opencodelists", False),
                    "usage_count_nhs": c.get("usage_count_nhs", 0.0),
                    "num_parents": c.get("num_parents", 0),
                    "num_children": c.get("num_children", 0),

                    "query_types_hit": set(),
                    "query_texts_hit": set(),
                    "clinical_focuses_hit": set(),
                    "retrieval_trace": [],

                    "semantic_score_max": 0.0,
                    "bm25_score_max": 0.0,
                    "lexical_overlap_max": 0.0,
                    "term_precision_max": 0.0,
                    "specificity_score_max": 0.0,
                    "best_adjusted_retrieval_score": 0.0,

                    "rrf_score": 0.0,
                    "has_direct_retrieval": False,
                    "has_hierarchy_retrieval": False,
                    "hierarchy_parent_code": None,
                    "hierarchy_parent_term": None,
                })

                item["query_types_hit"].add(c.get("query_type"))
                item["query_texts_hit"].add(c.get("query_text"))
                item["clinical_focuses_hit"].add(c.get("clinical_focus"))

                item["semantic_score_max"] = max(item["semantic_score_max"], float(c.get("semantic_score", 0.0)))
                item["bm25_score_max"] = max(item["bm25_score_max"], float(c.get("bm25_score", 0.0)))
                item["lexical_overlap_max"] = max(item["lexical_overlap_max"], float(c.get("lexical_overlap", 0.0)))
                item["term_precision_max"] = max(item["term_precision_max"], float(c.get("term_precision", 0.0)))
                item["specificity_score_max"] = max(item["specificity_score_max"], float(c.get("specificity_score", 0.0)))
                item["best_adjusted_retrieval_score"] = max(
                    item["best_adjusted_retrieval_score"],
                    float(c.get("adjusted_retrieval_score", 0.0)),
                )

                retrieval_method = c.get("retrieval_method", "direct")
                if retrieval_method == "hierarchy_child":
                    item["has_hierarchy_retrieval"] = True
                    item["hierarchy_parent_code"] = c.get("hierarchy_parent_code")
                    item["hierarchy_parent_term"] = c.get("hierarchy_parent_term")
                else:
                    item["has_direct_retrieval"] = True

                # rank-based fusion
                item["rrf_score"] += 1.0 / (self.rrf_k + rank)

                item["retrieval_trace"].append({
                    "batch_index": batch_index,
                    "rank_in_batch": rank,
                    "query_text": c.get("query_text"),
                    "query_type": c.get("query_type"),
                    "clinical_focus": c.get("clinical_focus"),
                    "retrieval_method": retrieval_method,
                    "semantic_score": round(float(c.get("semantic_score", 0.0)), 6),
                    "bm25_score": round(float(c.get("bm25_score", 0.0)), 6),
                    "lexical_overlap": round(float(c.get("lexical_overlap", 0.0)), 6),
                    "term_precision": round(float(c.get("term_precision", 0.0)), 6),
                    "specificity_score": round(float(c.get("specificity_score", 0.0)), 6),
                    "adjusted_retrieval_score": round(float(c.get("adjusted_retrieval_score", 0.0)), 6),
                })

        results: list[dict[str, Any]] = []
        for item in fused.values():
            results.append({
                "snomed_code": item["snomed_code"],
                "term": item["term"],
                "semantic_tag": item["semantic_tag"],
                "in_qof": item["in_qof"],
                "in_opencodelists": item["in_opencodelists"],
                "usage_count_nhs": item["usage_count_nhs"],
                "num_parents": item["num_parents"],
                "num_children": item["num_children"],

                # fusion output
                "fusion_score": round(float(item["rrf_score"]), 6),
                "rrf_score": round(float(item["rrf_score"]), 6),

                # carry max per-signal values for later decision/rerank use
                "semantic_score": round(float(item["semantic_score_max"]), 6),
                "bm25_score": round(float(item["bm25_score_max"]), 6),
                "lexical_overlap": round(float(item["lexical_overlap_max"]), 6),
                "term_precision": round(float(item["term_precision_max"]), 6),
                "specificity_score": round(float(item["specificity_score_max"]), 6),

                "semantic_score_max": round(float(item["semantic_score_max"]), 6),
                "bm25_score_max": round(float(item["bm25_score_max"]), 6),
                "lexical_overlap_max": round(float(item["lexical_overlap_max"]), 6),
                "term_precision_max": round(float(item["term_precision_max"]), 6),
                "specificity_score_max": round(float(item["specificity_score_max"]), 6),
                "best_adjusted_retrieval_score": round(float(item["best_adjusted_retrieval_score"]), 6),

                "query_coverage_count": len(item["clinical_focuses_hit"]),
                "query_types_hit": sorted(x for x in item["query_types_hit"] if x is not None),
                "clinical_focuses_hit": sorted(x for x in item["clinical_focuses_hit"] if x is not None),

                "has_direct_retrieval": item["has_direct_retrieval"],
                "has_hierarchy_retrieval": item["has_hierarchy_retrieval"],
                "retrieval_method": (
                    "direct+hierarchy"
                    if item["has_direct_retrieval"] and item["has_hierarchy_retrieval"]
                    else "hierarchy_child"
                    if item["has_hierarchy_retrieval"]
                    else "direct"
                ),
                "hierarchy_parent_code": item["hierarchy_parent_code"],
                "hierarchy_parent_term": item["hierarchy_parent_term"],
                "retrieval_trace": item["retrieval_trace"],
            })

        results.sort(
            key=lambda x: (
                x["fusion_score"],
                x["best_adjusted_retrieval_score"],
                x["term_precision_max"],
                x["lexical_overlap_max"],
                x["semantic_score_max"],
            ),
            reverse=True,
        )
        return results
