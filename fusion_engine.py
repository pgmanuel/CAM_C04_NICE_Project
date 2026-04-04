from collections import defaultdict
from typing import Any


class CandidateFusionEngine:
    def fuse(self, retrieval_batches: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}

        for batch in retrieval_batches:
            for candidate in batch:
                snomed_code = candidate["snomed_code"]
                item = fused.setdefault(
                    snomed_code,
                    {
                        "snomed_code": snomed_code,
                        "term": candidate["term"],
                        "semantic_tag": candidate["semantic_tag"],
                        "in_qof": candidate["in_qof"],
                        "in_opencodelists": candidate["in_opencodelists"],
                        "usage_count_nhs": candidate["usage_count_nhs"],
                        "query_types_hit": set(),
                        "query_texts_hit": set(),
                        "score_by_query_type": defaultdict(list),
                        "retrieval_trace": [],
                        "semantic_score_max": 0.0,
                        "bm25_score_max": 0.0,
                        "lexical_overlap_max": 0.0,
                        "best_primary_condition_score": 0.0,
                    },
                )

                item["query_types_hit"].add(candidate["query_type"])
                item["query_texts_hit"].add(candidate["query_text"])
                item["score_by_query_type"][candidate["query_type"]].append(
                    candidate["weighted_retrieval_score"]
                )
                item["semantic_score_max"] = max(item["semantic_score_max"], candidate["semantic_score"])
                item["bm25_score_max"] = max(item["bm25_score_max"], candidate["bm25_score"])
                item["lexical_overlap_max"] = max(item["lexical_overlap_max"], candidate["lexical_overlap"])
                if candidate["query_type"] == "primary_condition":
                    item["best_primary_condition_score"] = max(
                        item["best_primary_condition_score"],
                        candidate["weighted_retrieval_score"],
                    )
                item["retrieval_trace"].append(
                    {
                        "query_text": candidate["query_text"],
                        "query_type": candidate["query_type"],
                        "clinical_focus": candidate["clinical_focus"],
                        "semantic_score": round(candidate["semantic_score"], 4),
                        "bm25_score": round(candidate["bm25_score"], 4),
                        "lexical_overlap": round(candidate["lexical_overlap"], 4),
                        "semantic_tag_weight": round(candidate["semantic_tag_weight"], 4),
                        "retrieval_score": round(candidate["retrieval_score"], 4),
                        "adjusted_retrieval_score": round(candidate["adjusted_retrieval_score"], 4),
                        "weighted_retrieval_score": round(candidate["weighted_retrieval_score"], 4),
                    }
                )

        fused_results: list[dict[str, Any]] = []
        for item in fused.values():
            score_by_type = item["score_by_query_type"]
            primary_score = max(score_by_type.get("primary_condition", [0.0]))
            combined_score = max(score_by_type.get("combined", [0.0]))
            secondary_score = max(score_by_type.get("secondary_condition", [0.0]))
            modifier_score = max(score_by_type.get("modifier", [0.0]))
            supporting_score = max(score_by_type.get("supporting_term", [0.0]))
            coverage_bonus = min(0.08, 0.015 * len(item["query_texts_hit"]))
            isolation_penalty = 0.08 if primary_score == 0.0 and combined_score == 0.0 else 0.0

            fusion_score = (
                (primary_score * 1.35)
                + (combined_score * 1.0)
                + (secondary_score * 0.55)
                + (modifier_score * 0.2)
                + (supporting_score * 0.08)
                + coverage_bonus
                - isolation_penalty
            )

            fused_results.append(
                {
                    "snomed_code": item["snomed_code"],
                    "term": item["term"],
                    "semantic_tag": item["semantic_tag"],
                    "in_qof": item["in_qof"],
                    "in_opencodelists": item["in_opencodelists"],
                    "usage_count_nhs": item["usage_count_nhs"],
                    "fusion_score": round(float(fusion_score), 6),
                    "best_primary_condition_score": round(
                        float(item["best_primary_condition_score"]), 6
                    ),
                    "semantic_score_max": round(float(item["semantic_score_max"]), 6),
                    "bm25_score_max": round(float(item["bm25_score_max"]), 6),
                    "lexical_overlap_max": round(float(item["lexical_overlap_max"]), 6),
                    "query_coverage_count": len(item["query_texts_hit"]),
                    "query_types_hit": sorted(item["query_types_hit"]),
                    "retrieval_trace": item["retrieval_trace"],
                }
            )

        return fused_results
