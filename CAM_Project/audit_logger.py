import json
import logging
import uuid
from datetime import datetime
from typing import Any


logger = logging.getLogger("IntegratedPipeline")


def generate_run_id() -> str:
    return f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


class RunLogger:
    def __init__(
        self,
        config: Any,
        run_id: str,
        original_query: str,
        config_snapshot: dict[str, Any] | None = None,
    ):
        self.config = config
        self.run_id = run_id
        self.original_query = original_query
        self.config_snapshot = config_snapshot or {}
        self.start_time = datetime.now()
        self.trace: dict[str, Any] = {
            "normalized_query": None,
            "structured_query": None,
            "search_jobs": [],
            "retrieval_batches_summary": [],
            "top_candidates_per_search_job": [],
            "fused_candidates_preview": [],
            "fused_candidates_before_decisioning": [],
            "final_decisions": [],
            "explanation_mode": None,
            "formatter_debug": None,
        }

    def log_normalized_query(self, normalized_query: str) -> None:
        self.trace["normalized_query"] = normalized_query

    def log_structured_query(self, structured_query: dict[str, Any]) -> None:
        self.trace["structured_query"] = structured_query

    def log_search_jobs(self, search_jobs: list[dict[str, Any]]) -> None:
        self.trace["search_jobs"] = search_jobs

    def log_retrieval_batches(
        self,
        search_jobs: list[dict[str, Any]],
        retrieval_batches: list[list[dict[str, Any]]],
    ) -> None:
        summary = []
        for job, batch in zip(search_jobs, retrieval_batches):
            summary.append(
                {
                    "query_text": job["query_text"],
                    "query_type": job["query_type"],
                    "result_count": len(batch),
                }
            )
        self.trace["retrieval_batches_summary"] = summary

        detailed = []
        for job, batch in zip(search_jobs, retrieval_batches):
            detailed.append(
                {
                    "query_text": job["query_text"],
                    "query_type": job["query_type"],
                    "top_candidates": [
                        {
                            "snomed_code": candidate["snomed_code"],
                            "term": candidate["term"],
                            "semantic_tag": candidate["semantic_tag"],
                            "weighted_retrieval_score": candidate["weighted_retrieval_score"],
                            "lexical_overlap": candidate["lexical_overlap"],
                            "semantic_tag_weight": candidate["semantic_tag_weight"],
                        }
                        for candidate in batch[:10]
                    ],
                }
            )
        self.trace["top_candidates_per_search_job"] = detailed

    def log_fused_candidates(self, fused_candidates: list[dict[str, Any]]) -> None:
        preview = []
        for candidate in sorted(
            fused_candidates,
            key=lambda item: item["fusion_score"],
            reverse=True,
        )[:10]:
            preview.append(
                {
                    "snomed_code": candidate["snomed_code"],
                    "term": candidate["term"],
                    "fusion_score": candidate["fusion_score"],
                    "best_primary_condition_score": candidate["best_primary_condition_score"],
                    "query_types_hit": candidate["query_types_hit"],
                    "query_coverage_count": candidate["query_coverage_count"],
                }
            )
        self.trace["fused_candidates_preview"] = preview

        self.trace["fused_candidates_before_decisioning"] = [
            {
                "snomed_code": candidate["snomed_code"],
                "term": candidate["term"],
                "semantic_tag": candidate["semantic_tag"],
                "fusion_score": candidate["fusion_score"],
                "best_primary_condition_score": candidate["best_primary_condition_score"],
                "lexical_overlap_max": candidate["lexical_overlap_max"],
                "query_types_hit": candidate["query_types_hit"],
                "query_coverage_count": candidate["query_coverage_count"],
            }
            for candidate in sorted(
                fused_candidates,
                key=lambda item: item["fusion_score"],
                reverse=True,
            )[:25]
        ]

    def log_final_decisions(self, final_decisions: Any) -> None:
        self.trace["final_decisions"] = final_decisions

    def log_explanation_mode(self, explanation_mode: str) -> None:
        self.trace["explanation_mode"] = explanation_mode

    def log_formatter_debug(self, formatter_debug: dict[str, Any]) -> None:
        self.trace["formatter_debug"] = formatter_debug

    def _trace_for_verbosity(self) -> dict[str, Any]:
        verbosity = getattr(self.config, "audit_verbosity", "standard")
        if verbosity in {"debug", "standard"}:
            return self.trace
        if verbosity == "submission":
            return {
                "normalized_query": self.trace["normalized_query"],
                "structured_query": self.trace["structured_query"],
                "search_jobs": self.trace["search_jobs"],
                "retrieval_batches_summary": self.trace["retrieval_batches_summary"],
                "fused_candidates_preview": self.trace["fused_candidates_preview"],
                "final_decisions": self.trace["final_decisions"],
                "explanation_mode": self.trace["explanation_mode"],
            }
        return self.trace

    def finish(self, final_structured_output: Any) -> Any:
        packet = {
            "run_id": self.run_id,
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "query": self.original_query,
            "config_snapshot": self.config_snapshot,
            "trace": self._trace_for_verbosity(),
            "final_structured_output": final_structured_output,
        }
        audit_dir = self.config.audit_dir
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / f"{self.run_id}.json"
        with open(audit_path, "w", encoding="utf-8") as handle:
            json.dump(packet, handle, indent=2)
        logger.info("Pipeline audit written to %s", audit_path)
        return final_structured_output
