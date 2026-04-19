"""Optional evaluation entrypoint for the modular backend.

This script intentionally sits outside main.py. Normal production runs should
continue to use main.run_pipeline(); this runner is for opt-in custom evaluator
and RAGAS checks. Evaluation judge calls use Nebius via NEBIUS_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from audit_logger import RunLogger, generate_run_id
from ce_reranker import CrossEncoderReranker
from config import build_config
from decision_engine import DecisionEngine
from fusion_engine import CandidateFusionEngine
from gate_reranker import RelevanceGateReranker
from hierarchy_enricher import HierarchyEnricher
from output_formatter import LLMExplanationFormatter
from query_planning import QueryDecomposer, QueryPlanner, normalize_query
from retrieval_engine import DataLoader, HybridRetriever


logger = logging.getLogger(__name__)


def _evaluation_defaults() -> tuple[str, str]:
    from evaluation import DEFAULT_EVAL_MODEL, DEFAULT_RAGAS_MODEL

    return DEFAULT_EVAL_MODEL, DEFAULT_RAGAS_MODEL


def run_pipeline_with_optional_evaluation(
    user_query: str,
    top_k: int | None = None,
    run_custom_eval: bool = False,
    run_ragas: bool = False,
    eval_model: str | None = None,
    ragas_model: str | None = None,
) -> dict[str, Any]:
    config = build_config()
    effective_top_k = top_k or config.default_top_k
    default_eval_model, default_ragas_model = _evaluation_defaults()
    eval_model = eval_model or default_eval_model
    ragas_model = ragas_model or default_ragas_model

    run_id = generate_run_id()
    config_snapshot = config.to_snapshot(user_query, effective_top_k)
    config_snapshot["evaluation"] = {
        "provider": "nebius",
        "custom_eval_enabled": run_custom_eval,
        "ragas_enabled": run_ragas,
        "eval_model": eval_model if run_custom_eval else None,
        "ragas_model": ragas_model if run_ragas else None,
    }

    tracker = RunLogger(
        config,
        run_id,
        user_query,
        config_snapshot=config_snapshot,
    )

    normalized_query = normalize_query(user_query)
    tracker.log_normalized_query(normalized_query)

    data_loader = DataLoader(config)
    decomposer = QueryDecomposer(config.llm_model)
    planner = QueryPlanner(config.query_type_weights)
    retriever = HybridRetriever(config, data_loader)
    hierarchy_enricher = HierarchyEnricher(config.snomed_path, config.edge_path)
    fusion_engine = CandidateFusionEngine()
    relevance_reranker = RelevanceGateReranker(
        relevance_threshold=0.20,
        top_n_per_condition=config.gate_top_n_per_condition,
    )
    cross_encoder_reranker = CrossEncoderReranker(
        model_name=config.cross_encoder_model_name,
        top_n_per_condition=config.ce_top_n_per_condition,
        batch_size=16,
        max_length=512,
    )
    decision_engine = DecisionEngine()
    formatter = LLMExplanationFormatter(config)

    evaluations: dict[str, Any] = {}

    structured_query = decomposer.decompose(normalized_query)
    tracker.log_structured_query(structured_query)

    search_jobs = planner.build_search_queries(structured_query)
    tracker.log_search_jobs(search_jobs)

    retrieval_batches = [
        retriever.retrieve(job, top_k=effective_top_k)
        for job in search_jobs
    ]
    tracker.log_retrieval_batches(search_jobs, retrieval_batches)

    if run_custom_eval:
        from evaluation import build_retrieval_eval_rows, run_retrieval_evaluation

        retrieval_eval_rows = build_retrieval_eval_rows(search_jobs, retrieval_batches)
        retrieval_eval_result = run_retrieval_evaluation(
            retrieval_eval_rows,
            eval_model=eval_model,
        )
        tracker.log_retrieval_evaluation(retrieval_eval_result)
        evaluations["retrieval"] = retrieval_eval_result

    enriched_batches = [
        hierarchy_enricher.enrich_batch(batch)
        for batch in retrieval_batches
    ]
    tracker.log_hierarchy_enrichment(enriched_batches)

    fused_candidates = fusion_engine.fuse(enriched_batches)
    tracker.log_fused_candidates(fused_candidates)

    gate_candidates = relevance_reranker.rerank(
        fused_candidates,
        structured_query=structured_query,
    )
    tracker.log_gate_candidates(
        gate_candidates,
        relevance_threshold=relevance_reranker.relevance_threshold,
        top_n_per_condition=relevance_reranker.top_n_per_condition,
    )

    if run_custom_eval:
        from evaluation import build_post_gate_eval_rows, run_post_gate_evaluation

        post_gate_eval_rows = build_post_gate_eval_rows(structured_query, gate_candidates)
        post_gate_eval_result = run_post_gate_evaluation(
            post_gate_eval_rows,
            eval_model=eval_model,
        )
        tracker.log_post_gate_evaluation(post_gate_eval_result)
        evaluations["post_gate"] = post_gate_eval_result

    ce_reranked_candidates = cross_encoder_reranker.rerank(
        gate_candidates,
        structured_query=structured_query,
    )
    tracker.log_cross_encoder_candidates(
        ce_reranked_candidates,
        model_name=cross_encoder_reranker.model_name,
        top_n_per_condition=cross_encoder_reranker.top_n_per_condition,
        ce_condition_weight=config.ce_condition_weight,
        ce_query_weight=config.ce_query_weight,
    )

    if run_custom_eval:
        from evaluation import build_post_ce_eval_rows, run_post_ce_evaluation

        post_ce_eval_rows = build_post_ce_eval_rows(structured_query, ce_reranked_candidates)
        post_ce_eval_result = run_post_ce_evaluation(
            post_ce_eval_rows,
            eval_model=eval_model,
        )
        tracker.log_post_ce_evaluation(post_ce_eval_result)
        evaluations["post_ce"] = post_ce_eval_result

    candidate_groups = decision_engine.assign_final_decisions(
        ce_reranked_candidates,
        structured_query=structured_query,
        top_k=effective_top_k,
    )
    tracker.log_final_decisions(candidate_groups)

    if run_custom_eval:
        from evaluation import build_final_decision_eval_rows, run_final_decision_evaluation

        final_decision_eval_rows = build_final_decision_eval_rows(
            normalized_query,
            structured_query,
            candidate_groups,
        )
        final_decision_eval_result = run_final_decision_evaluation(
            final_decision_eval_rows,
            eval_model=eval_model,
        )
        tracker.log_final_decision_evaluation(final_decision_eval_result)
        evaluations["final_decision"] = final_decision_eval_result

    formatted_output, explanation_mode, formatter_debug = formatter.format_candidates(
        normalized_query,
        candidate_groups,
    )
    tracker.log_explanation_mode(explanation_mode)
    tracker.log_formatter_debug(formatter_debug)

    if run_ragas:
        from evaluation import build_final_response_eval_rows, run_final_response_ragas_evaluation

        final_response_eval_rows = build_final_response_eval_rows(
            normalized_query,
            candidate_groups,
            formatted_output,
        )
        final_response_ragas_result = run_final_response_ragas_evaluation(
            final_response_eval_rows,
            ragas_model=ragas_model,
        )
        tracker.log_final_response_ragas(final_response_ragas_result)
        evaluations["final_response_ragas"] = final_response_ragas_result

    payload = {
        "run_id": run_id,
        "query": user_query,
        "evaluation_enabled": {
            "custom_eval": run_custom_eval,
            "ragas": run_ragas,
        },
        "evaluations": evaluations,
        "pipeline_output": formatted_output,
    }
    return tracker.finish(payload)


def _build_parser() -> argparse.ArgumentParser:
    default_eval_model, default_ragas_model = _evaluation_defaults()
    parser = argparse.ArgumentParser(
        description="Run the backend pipeline with optional evaluation checks.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Clinical query to run. Defaults to Config.demo_query.",
    )
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--custom-eval",
        action="store_true",
        help="Run Nebius-backed custom LLM evaluator checks for retrieval, gate, CE, and decisions.",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Run final response RAGAS checks using a Nebius judge LLM.",
    )
    parser.add_argument(
        "--all-evals",
        action="store_true",
        help="Run both custom evaluator checks and RAGAS.",
    )
    parser.add_argument("--eval-model", default=default_eval_model)
    parser.add_argument("--ragas-model", default=default_ragas_model)
    return parser


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args()
    config = build_config()

    run_custom_eval = args.custom_eval or args.all_evals
    run_ragas = args.ragas or args.all_evals
    query = args.query or config.demo_query

    result = run_pipeline_with_optional_evaluation(
        query,
        top_k=args.top_k,
        run_custom_eval=run_custom_eval,
        run_ragas=run_ragas,
        eval_model=args.eval_model,
        ragas_model=args.ragas_model,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
