"""main.py — 9-stage pipeline runner (v4, CE-first).

Source: Playground.ipynb, Section 4 (cell run-pipeline).
Orchestration is unchanged from the notebook. Mirrors run_pipeline() exactly.

Stages:
1. Decompose query
2. Build per-condition search jobs
3. Retrieve (hybrid BM25 + semantic)
4. Hierarchy enrichment (zero-trust children)
5. RRF fusion
6. Relevance gate reranking
7. Cross-encoder reranking
8. Decision engine (CE-first, anchor selection)
9. Format output
"""

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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def run_pipeline(user_query: str, top_k: int | None = None) -> Any:
    config          = build_config()
    effective_top_k = top_k or config.default_top_k
    run_id          = generate_run_id()

    tracker = RunLogger(
        config,
        run_id,
        user_query,
        config_snapshot=config.to_snapshot(user_query, effective_top_k),
    )

    normalized_query = normalize_query(user_query)
    tracker.log_normalized_query(normalized_query)

    # Instantiate components
    data_loader       = DataLoader(config)
    decomposer        = QueryDecomposer(config.llm_model)
    planner           = QueryPlanner(config.query_type_weights)
    retriever         = HybridRetriever(config, data_loader)
    hierarchy_enricher = HierarchyEnricher(config.snomed_path, config.edge_path)
    fusion_engine     = CandidateFusionEngine()

    # Stage 6: Relevance Gate
    relevance_reranker = RelevanceGateReranker(
        relevance_threshold=0.20,          # v2 plan: hard threshold at 0.20
        top_n_per_condition=config.gate_top_n_per_condition,
    )

    # Stage 7: Cross-Encoder Reranker
    cross_encoder_reranker = CrossEncoderReranker(
        model_name=config.cross_encoder_model_name,
        top_n_per_condition=config.ce_top_n_per_condition,
        batch_size=16,
        max_length=512,
    )

    decision_engine = DecisionEngine()
    formatter       = LLMExplanationFormatter(config)

    # 1. Query decomposition
    structured_query = decomposer.decompose(normalized_query)
    tracker.log_structured_query(structured_query)

    # 2. Build search jobs
    search_jobs = planner.build_search_queries(structured_query)
    tracker.log_search_jobs(search_jobs)

    # 3. Retrieval
    retrieval_batches = [
        retriever.retrieve(job, top_k=effective_top_k)
        for job in search_jobs
    ]
    tracker.log_retrieval_batches(search_jobs, retrieval_batches)

    # 4. Hierarchy enrichment
    enriched_batches = [
        hierarchy_enricher.enrich_batch(batch)
        for batch in retrieval_batches
    ]
    tracker.log_hierarchy_enrichment(enriched_batches)

    # 5. Fusion
    fused_candidates = fusion_engine.fuse(enriched_batches)
    tracker.log_fused_candidates(fused_candidates)

    # 6. Relevance Gate
    gate_candidates = relevance_reranker.rerank(
        fused_candidates,
        structured_query=structured_query,
    )
    tracker.log_gate_candidates(
        gate_candidates,
        relevance_threshold=relevance_reranker.relevance_threshold,
        top_n_per_condition=relevance_reranker.top_n_per_condition,
    )

    # 7. Cross-Encoder Rerank (STRICT per-condition only)
    ce_reranked_candidates = cross_encoder_reranker.rerank(
        gate_candidates,
        structured_query=structured_query,
    )
    tracker.log_cross_encoder_candidates(
        ce_reranked_candidates,
        model_name=cross_encoder_reranker.model_name,
        top_n_per_condition=cross_encoder_reranker.top_n_per_condition,
    )

    # 8. Decision Engine
    candidate_groups = decision_engine.assign_final_decisions(
        ce_reranked_candidates,
        structured_query=structured_query,
        top_k=effective_top_k,
    )
    tracker.log_final_decisions(candidate_groups)

    # 9. Formatter
    formatted_output, explanation_mode, formatter_debug = formatter.format_candidates(
        normalized_query,
        candidate_groups,
    )
    tracker.log_explanation_mode(explanation_mode)
    tracker.log_formatter_debug(formatter_debug)

    return tracker.finish(formatted_output)


if __name__ == "__main__":
    config = build_config()
    pipeline_output = run_pipeline(config.demo_query, top_k=config.top_k)
    print(json.dumps(pipeline_output, indent=2))
