import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "snomed_master_v3.csv").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent

@dataclass(frozen=True)
class Config:
    base_dir: Path
    snomed_path: Path
    chroma_persist_dir: Path
    chroma_collection_name: str = "snomed_master_v3_retrieval"
    embedding_model_name: str = "BAAI/bge-small-en"
    llm_model: str = "llama3.1"
    default_top_k: int = 20
    audit_verbosity: str = "standard"
    query_type_weights: dict[str, float] = field(
        default_factory=lambda: {
            "primary_condition": 1.0,
            "combined": 0.9,
            "secondary_condition": 0.75,
            "modifier": 0.45,
            "supporting_term": 0.35,
        }
    )

    def to_snapshot(self, user_query: str, effective_top_k: int) -> dict[str, Any]:
        return {
            "query": user_query,
            "input_assets": {
                "snomed_csv": str(self.snomed_path),
                "chroma_persist_dir": str(self.chroma_persist_dir),
                "chroma_collection_name": self.chroma_collection_name,
            },
            "models": {
                "embedding_model_name": self.embedding_model_name,
                "formatter_mode": "deterministic_disabled",
                "formatter_model_configured": self.llm_model,
            },
            "runtime": {
                "default_top_k": self.default_top_k,
                "effective_top_k": effective_top_k,
                "audit_verbosity": self.audit_verbosity,
            },
            "query_type_weights": dict(self.query_type_weights),
            "bucket_caps": {
                "include_candidates": DEFAULT_INCLUDE_CANDIDATES_CAP,
                "review_candidates": DEFAULT_REVIEW_CANDIDATES_CAP,
                "specific_variants": DEFAULT_SPECIFIC_VARIANTS_CAP,
                "candidate_pool_limit": DEFAULT_CANDIDATE_POOL_LIMIT,
            },
            "ranking_policy_terms": {
                "query_exception_terms": sorted(QUERY_EXCEPTION_TERMS),
                "causal_trigger_terms": list(CAUSAL_TRIGGER_TERMS),
                "pregnancy_trigger_terms": list(PREGNANCY_TRIGGER_TERMS),
                "core_block_terms": list(CORE_BLOCK_TERMS),
                "narrow_subtype_terms": list(NARROW_SUBTYPE_TERMS),
            },
            "retrieval_suppression_terms": {
                "history_exception_terms": sorted(RETRIEVAL_HISTORY_EXCEPTION_TERMS),
                "pregnancy_exception_terms": sorted(RETRIEVAL_PREGNANCY_EXCEPTION_TERMS),
                "pregnancy_markers": list(RETRIEVAL_PREGNANCY_MARKERS),
            },
        }


def build_config() -> Config:
    base_dir = find_project_root()
    snomed_path = Path(
        Path(
            os.environ.get(
                "INTEGRATED_AGENT_SNOMED_PATH",
                str(base_dir / "snomed_master_v3.csv"),
            )
        )
    )
    chroma_dir = Path(
        Path(
            os.environ.get(
                "INTEGRATED_AGENT_CHROMA_DIR",
                str(base_dir / "chroma_db"),
            )
        )
    )
    return Config(
        base_dir=base_dir,
        snomed_path=snomed_path,
        chroma_persist_dir=chroma_dir,
        llm_model=os.environ.get("INTEGRATED_AGENT_LLM_MODEL", "llama3.1"),
        audit_verbosity=os.environ.get("INTEGRATED_AGENT_AUDIT_VERBOSITY", "standard"),
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
    demo_query = "Obesity, diabetes mellitus, and hypertension"
    pipeline_output = run_pipeline(demo_query, top_k=20)
    print(json.dumps(pipeline_output, indent=2))
