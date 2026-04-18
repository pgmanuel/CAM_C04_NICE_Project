# ==========================================
# NICE Pipeline v4 — User Settings
# Copy this file to user_settings.py and fill in your local paths.
# ==========================================

# ==========================================
# Paths (Absolute paths recommended)
# ==========================================
# v4 requires snomed_master_v4.csv AND the hierarchy edge file.
SNOMED_PATH   = "../snomed_master_v4.csv"
EDGE_PATH     = "../snomed_parent_child_edges_clean.csv"
CHROMA_DIR    = "../chroma_db_v4"
EMBEDDINGS_DIR = "../embeddings"

# ==========================================
# Models
# ==========================================
EMBEDDING_MODEL_NAME    = "BAAI/bge-small-en"
CROSS_ENCODER_MODEL     = "BAAI/bge-reranker-v2-m3"
CHROMA_COLLECTION       = "snomed_master_v4_retrieval"
LLM_MODEL               = "llama3.1"

# ==========================================
# Runtime
# ==========================================
DEMO_QUERY      = "Obesity, diabetes mellitus, and hypertension"

RETRIEVAL_LIMIT = 20
