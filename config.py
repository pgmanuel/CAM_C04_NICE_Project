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
    return Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    base_dir:               Path
    snomed_path:            Path
    edge_path:              Path
    chroma_persist_dir:     Path
    embeddings_dir:         Path
    audit_dir:              Path

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
    rebuild_chroma:         bool = False
    chroma_rebuild_batch_size: int = 1000

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
                "embeddings_dir":         str(self.embeddings_dir),
                "audit_dir":              str(self.audit_dir),
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
                "rebuild_chroma":           self.rebuild_chroma,
                "chroma_rebuild_batch_size": self.chroma_rebuild_batch_size,
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


def _resolve_path(value: Any, base_dir: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def validate_source_files(config: Config) -> None:
    missing = []
    if not config.snomed_path.is_file():
        missing.append(
            f"SNOMED CSV not found at {config.snomed_path}. "
            "Set NICE_SNOMED_PATH or SNOMED_PATH in user_settings.py."
        )
    if not config.edge_path.is_file():
        missing.append(
            f"Hierarchy edge CSV not found at {config.edge_path}. "
            "Set NICE_EDGE_PATH or EDGE_PATH in user_settings.py."
        )
    if missing:
        raise FileNotFoundError("Required source files are missing:\n- " + "\n- ".join(missing))


def ensure_runtime_dirs(config: Config) -> None:
    for path in [config.chroma_persist_dir, config.embeddings_dir, config.audit_dir]:
        path.mkdir(parents=True, exist_ok=True)


def build_config() -> Config:
    base_dir = find_project_root()

    try:
        import user_settings
    except ImportError:
        user_settings = None

    def get_setting(name: str, env_name: str, default: Any) -> Any:
        if env_name and env_name in os.environ:
            return os.environ[env_name]
        if user_settings and hasattr(user_settings, name):
            val = getattr(user_settings, name)
            if val is not None:
                return val
        return default

    snomed_path = _resolve_path(get_setting(
        "SNOMED_PATH", "NICE_SNOMED_PATH",
        "../snomed_master_v4.csv"
    ), base_dir)

    edge_path = _resolve_path(get_setting(
        "EDGE_PATH", "NICE_EDGE_PATH",
        "../snomed_parent_child_edges_clean.csv"
    ), base_dir)

    explicit_chroma = get_setting("CHROMA_DIR", "NICE_CHROMA_DIR", None)
    if explicit_chroma:
        chroma_dir = _resolve_path(explicit_chroma, base_dir)
    else:
        chroma_dir = (base_dir / "../chroma_db_v4").resolve()

    explicit_embeddings = get_setting("EMBEDDINGS_DIR", "NICE_EMBEDDINGS_DIR", None)
    if explicit_embeddings:
        embeddings_dir = _resolve_path(explicit_embeddings, base_dir)
    else:
        embeddings_dir = (base_dir / "../embeddings").resolve()

    explicit_audit = get_setting("AUDIT_DIR", "NICE_AUDIT_DIR", None)
    if explicit_audit:
        audit_dir = _resolve_path(explicit_audit, base_dir)
    else:
        audit_dir = (base_dir / "audit").resolve()

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

    raw_rebuild_chroma = get_setting("REBUILD_CHROMA", "NICE_REBUILD_CHROMA", "false")
    raw_rebuild_batch_size = get_setting("CHROMA_REBUILD_BATCH_SIZE", "NICE_CHROMA_REBUILD_BATCH_SIZE", 1000)
    try:
        chroma_rebuild_batch_size = max(1, int(raw_rebuild_batch_size))
    except (ValueError, TypeError):
        chroma_rebuild_batch_size = 1000

    config = Config(
        base_dir=base_dir,
        snomed_path=snomed_path,
        edge_path=edge_path,
        chroma_persist_dir=chroma_dir,
        embeddings_dir=embeddings_dir,
        audit_dir=audit_dir,
        demo_query=demo_query,
        top_k=top_k,
        chroma_collection_name=chroma_collection,
        embedding_model_name=embedding_model_name,
        llm_model=llm_model,
        cross_encoder_model_name=cross_encoder_model,
        audit_verbosity=os.environ.get("NICE_AUDIT_VERBOSITY", "standard"),
        rebuild_chroma=_truthy(raw_rebuild_chroma),
        chroma_rebuild_batch_size=chroma_rebuild_batch_size,
    )
    validate_source_files(config)
    ensure_runtime_dirs(config)
    return config
