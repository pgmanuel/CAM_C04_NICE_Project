"""audit_logger.py — pipeline run auditing with hierarchy/gate/CE trace slots.

Source: Playground.ipynb, Section 3 (cell XwjtS8PHD9-d).
Logic is unchanged from the notebook. New log slots added for v4 stages.
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"run_{ts}_{uuid.uuid4().hex[:6]}"


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
            "retrieval_evaluation_summary": {},
            "retrieval_evaluation_details": [],
            "hierarchy_enrichment_summary": [],  # new v4 slot
            "fused_candidates_preview": [],
            "gate_summary": {},                  # new v4 slot
            "gate_candidates_preview": [],        # new v4 slot
            "post_gate_evaluation_summary": {},
            "post_gate_evaluation_details": [],
            "post_ce_evaluation_summary": {},
            "post_ce_evaluation_details": [],
            "cross_encoder_summary": {},          # new v4 slot
            "cross_encoder_candidates_preview": [],  # new v4 slot
            "final_decision_evaluation_summary": {},
            "final_decision_evaluation_details": [],
            "final_decisions": [],
            "explanation_mode": None,
            "formatter_debug": None,
            "final_response_ragas_summary": {},
            "final_response_ragas_details": [],
        }

    def log_normalized_query(self, v: str) -> None:
        self.trace["normalized_query"] = v

    def log_structured_query(self, v: dict[str, Any]) -> None:
        self.trace["structured_query"] = v

    def log_search_jobs(self, v: list) -> None:
        self.trace["search_jobs"] = v

    def log_final_decisions(self, v: Any) -> None:
        self.trace["final_decisions"] = v

    def log_explanation_mode(self, v: str) -> None:
        self.trace["explanation_mode"] = v

    def log_formatter_debug(self, v: dict) -> None:
        self.trace["formatter_debug"] = v

    def log_retrieval_batches(self, search_jobs: list, retrieval_batches: list) -> None:
        self.trace["retrieval_batches_summary"] = [
            {
                "query_text":     job["query_text"],
                "query_type":     job["query_type"],
                "clinical_focus": job.get("clinical_focus"),
                "result_count":   len(batch),
            }
            for job, batch in zip(search_jobs, retrieval_batches)
        ]

    def log_retrieval_evaluation(self, retrieval_eval_result: dict[str, Any]) -> None:
        self.trace["retrieval_evaluation_summary"] = {
            "stage": retrieval_eval_result.get("stage"),
            "num_rows": retrieval_eval_result.get("num_rows"),
            "average_custom_score": retrieval_eval_result.get("average_custom_score"),
        }
        self.trace["retrieval_evaluation_details"] = retrieval_eval_result.get("details", [])

    def log_post_gate_evaluation(self, post_gate_eval_result: dict[str, Any]) -> None:
        self.trace["post_gate_evaluation_summary"] = {
            "stage": post_gate_eval_result.get("stage"),
            "num_rows": post_gate_eval_result.get("num_rows"),
            "average_custom_score": post_gate_eval_result.get("average_custom_score"),
        }
        self.trace["post_gate_evaluation_details"] = post_gate_eval_result.get("details", [])

    def log_post_ce_evaluation(self, post_ce_eval_result: dict[str, Any]) -> None:
        self.trace["post_ce_evaluation_summary"] = {
            "stage": post_ce_eval_result.get("stage"),
            "num_rows": post_ce_eval_result.get("num_rows"),
            "average_custom_score": post_ce_eval_result.get("average_custom_score"),
        }
        self.trace["post_ce_evaluation_details"] = post_ce_eval_result.get("details", [])

    def log_final_decision_evaluation(self, final_decision_eval_result: dict[str, Any]) -> None:
        self.trace["final_decision_evaluation_summary"] = {
            "stage": final_decision_eval_result.get("stage"),
            "num_rows": final_decision_eval_result.get("num_rows"),
            "average_custom_score": final_decision_eval_result.get("average_custom_score"),
        }
        self.trace["final_decision_evaluation_details"] = final_decision_eval_result.get("details", [])

    def log_final_response_ragas(self, eval_result: dict[str, Any]) -> None:
        self.trace["final_response_ragas_summary"] = {
            "stage": eval_result.get("stage"),
            "num_rows": eval_result.get("num_rows"),
            "faithfulness": eval_result.get("faithfulness"),
            "response_relevancy": eval_result.get("response_relevancy"),
        }
        self.trace["final_response_ragas_details"] = eval_result.get("details", [])

    def log_hierarchy_enrichment(self, enriched_batches: list[list[dict[str, Any]]]) -> None:
        """Log summary of hierarchy enrichment stage (v4 new slot)."""
        self.trace["hierarchy_enrichment_summary"] = [
            {
                "batch_index": i,
                "total_candidates_after_enrichment": len(batch),
                "hierarchy_added_count": sum(
                    1 for c in batch if c.get("retrieval_method") == "hierarchy_child"
                ),
            }
            for i, batch in enumerate(enriched_batches)
        ]

    def log_fused_candidates(self, fused_candidates: list[dict[str, Any]]) -> None:
        self.trace["fused_candidates_preview"] = [
            {
                "snomed_code":            c["snomed_code"],
                "term":                   c["term"],
                "fusion_score":           c.get("fusion_score", 0.0),
                "rrf_score":              c.get("rrf_score", 0.0),
                "query_types_hit":        c.get("query_types_hit", []),
                "clinical_focuses_hit":   c.get("clinical_focuses_hit", []),
                "query_coverage_count":   c.get("query_coverage_count", 0),
                "retrieval_method":       c.get("retrieval_method", "direct"),
            }
            for c in sorted(fused_candidates, key=lambda x: x.get("fusion_score", 0.0), reverse=True)[:10]
        ]

    def log_gate_candidates(
        self,
        gate_candidates: list[dict[str, Any]],
        relevance_threshold: float | None = None,
        top_n_per_condition: int | None = None,
    ) -> None:
        """Log summary of relevance gate reranker stage (v4 new slot)."""
        self.trace["gate_summary"] = {
            "relevance_threshold":    relevance_threshold,
            "top_n_per_condition":    top_n_per_condition,
            "survivor_count":         len(gate_candidates),
        }

        self.trace["gate_candidates_preview"] = [
            {
                "snomed_code":                  c["snomed_code"],
                "term":                         c["term"],
                "rerank_score":                 c.get("rerank_score", 0.0),
                "relevance_score":              c.get("relevance_score", 0.0),
                "matched_conditions_from_gate": c.get("matched_conditions_from_gate", []),
                "dominant_condition_from_gate": c.get("dominant_condition_from_gate"),
                "retrieval_method":             c.get("retrieval_method", "direct"),
            }
            for c in sorted(gate_candidates, key=lambda x: x.get("rerank_score", 0.0), reverse=True)[:20]
        ]

    def log_cross_encoder_candidates(
        self,
        ce_candidates: list[dict[str, Any]],
        model_name: str | None = None,
        top_n_per_condition: int | None = None,
        ce_condition_weight: float | None = None,
        ce_query_weight: float | None = None,
    ) -> None:
        """Log summary of cross-encoder reranker stage (v4 new slot)."""
        self.trace["cross_encoder_summary"] = {
            "model_name":            model_name,
            "top_n_per_condition":   top_n_per_condition,
            "ce_condition_weight":   ce_condition_weight,
            "ce_query_weight":       ce_query_weight,
            "survivor_count":        len(ce_candidates),
        }

        self.trace["cross_encoder_candidates_preview"] = [
            {
                "snomed_code":                    c["snomed_code"],
                "term":                           c["term"],
                "ce_score":                       c.get("ce_score", 0.0),
                "rerank_score":                   c.get("rerank_score", 0.0),
                "relevance_score":                c.get("relevance_score", 0.0),
                "matched_conditions_from_gate":   c.get("matched_conditions_from_gate", []),
                "dominant_condition_from_gate":   c.get("dominant_condition_from_gate"),
                "matched_conditions_from_ce":     c.get("matched_conditions_from_ce", []),
                "dominant_condition_from_ce":     c.get("dominant_condition_from_ce"),
                "ce_score_by_condition":          c.get("ce_score_by_condition", {}),
                "retrieval_method":               c.get("retrieval_method", "direct"),
            }
            for c in sorted(ce_candidates, key=lambda x: x.get("ce_score", 0.0), reverse=True)[:20]
        ]

    def finish(self, final_structured_output: Any) -> Any:
        packet = {
            "run_id":                  self.run_id,
            "timestamp":               self.start_time.isoformat(),
            "duration_seconds":        (datetime.now() - self.start_time).total_seconds(),
            "query":                   self.original_query,
            "config_snapshot":         self.config_snapshot,
            "trace":                   self.trace,
            "final_structured_output": final_structured_output,
        }

        try:
            audit_dir  = Path(self.config.base_dir) / "audit"
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / f"{self.run_id}.json"
            with open(audit_path, "w", encoding="utf-8") as fh:
                json.dump(packet, fh, indent=2)
            logger.info("Audit written to %s", audit_path)
        except Exception as exc:
            logger.warning("Audit write skipped (non-fatal): %s", exc)

        return final_structured_output
