import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import Config, build_config

from audit_logger import RunLogger, generate_run_id
from output_formatter import LLMExplanationFormatter
from pipeline_policy import (
    CAUSAL_TRIGGER_TERMS,
    CORE_BLOCK_TERMS,
    DEFAULT_CANDIDATE_POOL_LIMIT,
    DEFAULT_INCLUDE_CANDIDATES_CAP,
    DEFAULT_REVIEW_CANDIDATES_CAP,
    DEFAULT_SPECIFIC_VARIANTS_CAP,
    NARROW_SUBTYPE_TERMS,
    PREGNANCY_TRIGGER_TERMS,
    QUERY_EXCEPTION_TERMS,
)
from query_planning import QueryDecomposer, QueryPlanner, normalize_query
from retrieval_engine import (
    DataLoader,
    HybridRetriever,
    RETRIEVAL_HISTORY_EXCEPTION_TERMS,
    RETRIEVAL_PREGNANCY_EXCEPTION_TERMS,
    RETRIEVAL_PREGNANCY_MARKERS,
)
from ranking_engine import CandidateFusionEngine, DecisionEngine


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)




def run_pipeline(user_query: str, top_k: int | None = None) -> Any:
    config = build_config()
    effective_top_k = top_k or config.default_top_k
    run_id = generate_run_id()
    tracker = RunLogger(
        config,
        run_id,
        user_query,
        config_snapshot=config.to_snapshot(user_query, effective_top_k),
    )

    normalized_query = normalize_query(user_query)
    tracker.log_normalized_query(normalized_query)

    data_loader = DataLoader(config)
    decomposer = QueryDecomposer(config.llm_model)
    planner = QueryPlanner(config.query_type_weights)
    retriever = HybridRetriever(config, data_loader)
    fusion_engine = CandidateFusionEngine()
    decision_engine = DecisionEngine()
    formatter = LLMExplanationFormatter(config)

    structured_query = decomposer.decompose(normalized_query)
    tracker.log_structured_query(structured_query)

    search_jobs = planner.build_search_queries(structured_query)
    tracker.log_search_jobs(search_jobs)

    retrieval_batches = [retriever.retrieve(job, top_k=effective_top_k) for job in search_jobs]
    tracker.log_retrieval_batches(search_jobs, retrieval_batches)

    fused_candidates = fusion_engine.fuse(retrieval_batches)
    tracker.log_fused_candidates(fused_candidates)

    candidate_groups = decision_engine.assign_final_decisions(
        fused_candidates,
        structured_query=structured_query,
        top_k=effective_top_k,
    )
    tracker.log_final_decisions(candidate_groups)

    formatted_output, explanation_mode, formatter_debug = formatter.format_candidates(
        normalized_query,
        candidate_groups,
    )
    tracker.log_explanation_mode(explanation_mode)
    tracker.log_formatter_debug(formatter_debug)
    return tracker.finish(formatted_output)


if __name__ == "__main__":
    from config import build_config
    config = build_config()
    pipeline_output = run_pipeline(config.demo_query, top_k=config.top_k)
    print(json.dumps(pipeline_output, indent=2))
