"""config.py — centralised pipeline configuration.

Source: Playground.ipynb, Section 3 (cell bA6xBfFTEhpd).
Config dataclass updated to v4: edge_path, CE/gate params, v4 defaults.
build_config() reads from user_settings.py, env vars, or hardcoded defaults.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "snomed_master_v4.csv").exists():
            return current
        if (current / "snomed_master_v3.csv").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    base_dir:               Path
    snomed_path:            Path
    edge_path:              Path
    chroma_persist_dir:     Path
    embeddings_dir:         Path

    demo_query:             str = "Obesity, diabetes mellitus, and hypertension"
    top_k:                  int = 10

    chroma_collection_name: str = "snomed_master_v4_retrieval"
    embedding_model_name:   str = "BAAI/bge-small-en"
    llm_model:              str = "llama3.1"

    # Cross-encoder reranker config
    cross_encoder_model_name: str   = "BAAI/bge-reranker-v2-m3"
    gate_top_n_per_condition: int   = 50
    ce_top_n_per_condition:   int   = 20
    ce_condition_weight:      float = 0.75
    ce_query_weight:          float = 0.25

    default_top_k:          int = 20
    audit_verbosity:        str = "standard"

    # query_weight is metadata only — NOT used to sort retrieval candidates
    query_type_weights: dict[str, float] = field(default_factory=lambda: {
        "primary_condition":  1.0,
        "secondary_condition": 1.0,
        "modifier":           0.5,
        "supporting_term":    0.35,
        "combined":           0.0,
    })

    def to_snapshot(self, user_query: str, effective_top_k: int) -> dict[str, Any]:
        return {
            "query": user_query,
            "input_assets": {
                "snomed_csv":             str(self.snomed_path),
                "hierarchy_edge_csv":     str(self.edge_path),
                "chroma_persist_dir":     str(self.chroma_persist_dir),
                "chroma_collection_name": self.chroma_collection_name,
            },
            "models": {
                "embedding_model_name":       self.embedding_model_name,
                "llm_model":                  self.llm_model,
                "cross_encoder_model_name":   self.cross_encoder_model_name,
            },
            "runtime": {
                "effective_top_k":          effective_top_k,
                "default_top_k":            self.default_top_k,
                "audit_verbosity":          self.audit_verbosity,
                "gate_top_n_per_condition": self.gate_top_n_per_condition,
                "ce_top_n_per_condition":   self.ce_top_n_per_condition,
                "ce_condition_weight":      self.ce_condition_weight,
                "ce_query_weight":          self.ce_query_weight,
            },
            "architecture": {
                "condition_isolated_planning":      True,
                "hierarchy_zero_trust_enrichment":  True,
                "rrf_fusion":                       True,
                "relevance_gate_reranker":          True,
                "cross_encoder_reranker":           True,
                "bounded_authority_decisioning":    True,
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

    snomed_path = Path(get_setting(
        "SNOMED_PATH", "NICE_SNOMED_PATH",
        str(base_dir / "snomed_master_v4.csv")
    ))

    edge_path = Path(get_setting(
        "EDGE_PATH", "NICE_EDGE_PATH",
        str(base_dir / "snomed_parent_child_edges_clean.csv")
    ))

    explicit_chroma = get_setting("CHROMA_DIR", "NICE_CHROMA_DIR", None)
    if explicit_chroma:
        chroma_dir = Path(explicit_chroma)
    else:
        chroma_dir = (base_dir / "../chroma_db_v4").resolve()

    explicit_embeddings = get_setting("EMBEDDINGS_DIR", "NICE_EMBEDDINGS_DIR", None)
    if explicit_embeddings:
        embeddings_dir = Path(explicit_embeddings)
    else:
        embeddings_dir = (base_dir / "../embeddings").resolve()

    demo_query           = get_setting("DEMO_QUERY", "NICE_DEMO_QUERY", "Obesity, diabetes mellitus, and hypertension")
    embedding_model_name = get_setting("EMBEDDING_MODEL_NAME", "NICE_EMBEDDING_MODEL_NAME", "BAAI/bge-small-en")
    llm_model            = get_setting("LLM_MODEL", "NICE_LLM_MODEL", "llama3.1")
    cross_encoder_model  = get_setting("CROSS_ENCODER_MODEL", "NICE_CROSS_ENCODER_MODEL", "BAAI/bge-reranker-v2-m3")
    chroma_collection    = get_setting("CHROMA_COLLECTION", "NICE_CHROMA_COLLECTION", "snomed_master_v4_retrieval")

    raw_top_k = get_setting("RETRIEVAL_LIMIT", "NICE_RETRIEVAL_LIMIT", 10)
    try:
        top_k = int(raw_top_k)
    except (ValueError, TypeError):
        top_k = 10

    return Config(
        base_dir=base_dir,
        snomed_path=snomed_path,
        edge_path=edge_path,
        chroma_persist_dir=chroma_dir,
        embeddings_dir=embeddings_dir,
        demo_query=demo_query,
        top_k=top_k,
        chroma_collection_name=chroma_collection,
        embedding_model_name=embedding_model_name,
        llm_model=llm_model,
        cross_encoder_model_name=cross_encoder_model,
        audit_verbosity=os.environ.get("NICE_AUDIT_VERBOSITY", "standard"),
    )
