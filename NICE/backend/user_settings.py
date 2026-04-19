"""
Template for local project configuration.

Copy this file to `user_settings.py` (which is ignored by git)
to apply your personal local settings without modifying core files.

SECURITY WARNING: 
Do NOT store API keys, tokens, or any secret credentials in this file.
Secrets should always remain in your environment variables.
"""

# ==========================================
# Paths (Absolute or Relative to project root)
# ==========================================
SNOMED_PATH    = "/Users/vic/Desktop/NICE/snomed_master_v4.csv"
EDGE_PATH      = "/Users/vic/Desktop/NICE/snomed_parent_child_edges_clean.csv"
CHROMA_DIR     = "/Users/vic/Desktop/NICE/chroma_db_v4"
EMBEDDINGS_DIR = "/Users/vic/Desktop/NICE/embeddings"

# ==========================================
# Models
# ==========================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
CROSS_ENCODER_MODEL  = "BAAI/bge-reranker-v2-m3"
CHROMA_COLLECTION    = "snomed_master_v4_retrieval"
LLM_MODEL            = "llama3.1"

# ==========================================
# Runtime
# ==========================================
DEMO_QUERY      = "Obesity, diabetes mellitus, and hypertension"
RETRIEVAL_LIMIT = 20

# ==========================================
# Optional API Modes (For future use)
# ==========================================
USE_API_EMBEDDINGS = False
USE_API_LLM = False

# Example of configuring provider and model names (no credentials here)
EMBEDDING_API_PROVIDER = None
EMBEDDING_API_MODEL = None
LLM_API_PROVIDER = None
LLM_API_MODEL = None
