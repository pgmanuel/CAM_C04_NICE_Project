import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
from retrieval_engine import (
    RETRIEVAL_HISTORY_EXCEPTION_TERMS,
    RETRIEVAL_PREGNANCY_EXCEPTION_TERMS,
    RETRIEVAL_PREGNANCY_MARKERS,
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
    embeddings_dir: Path
    demo_query: str = "Obesity, diabetes mellitus, and hypertension"
    top_k: int = 10
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
                "embeddings_dir": str(self.embeddings_dir),
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
    
    try:
        import user_settings
    except ImportError:
        user_settings = None

    def get_setting(name: str, env_name: str, default: Any) -> Any:
        if user_settings and hasattr(user_settings, name):
            val = getattr(user_settings, name)
            if val is not None:
                return val
        if env_name and env_name in os.environ:
            return os.environ[env_name]
        return default

    snomed_path = Path(get_setting("SNOMED_PATH", "INTEGRATED_AGENT_SNOMED_PATH", str(base_dir / "snomed_master_v3.csv")))

    explicit_chroma = get_setting("CHROMA_DIR", "INTEGRATED_AGENT_CHROMA_DIR", None)
    if explicit_chroma:
        chroma_dir = Path(explicit_chroma)
    else:
        chroma_dir = (base_dir / "../chroma_db").resolve()

    explicit_embeddings = get_setting("EMBEDDINGS_DIR", "INTEGRATED_AGENT_EMBEDDINGS_DIR", None)
    if explicit_embeddings:
        embeddings_dir = Path(explicit_embeddings)
    else:
        embeddings_dir = (base_dir / "../embeddings").resolve()
        
    demo_query = get_setting("DEMO_QUERY", "INTEGRATED_AGENT_DEMO_QUERY", "Obesity, diabetes mellitus, and hypertension")
    embedding_model_name = get_setting("EMBEDDING_MODEL_NAME", "INTEGRATED_AGENT_EMBEDDING_MODEL_NAME", "BAAI/bge-small-en")
    llm_model = get_setting("LLM_MODEL", "INTEGRATED_AGENT_LLM_MODEL", "llama3.1")
    
    raw_top_k = get_setting("RETRIEVAL_LIMIT", "INTEGRATED_AGENT_RETRIEVAL_LIMIT", 10)
    try:
        top_k = int(raw_top_k)
    except (ValueError, TypeError):
        top_k = 10

    return Config(
        base_dir=base_dir,
        snomed_path=snomed_path,
        chroma_persist_dir=chroma_dir,
        embeddings_dir=embeddings_dir,
        demo_query=demo_query,
        top_k=top_k,
        embedding_model_name=embedding_model_name,
        llm_model=llm_model,
        audit_verbosity=os.environ.get("INTEGRATED_AGENT_AUDIT_VERBOSITY", "standard"),
    )
