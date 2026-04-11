# ==========================================
# Paths (Absolute or Relative to project root)
# ==========================================
SNOMED_PATH = "../snomed_master_v3.csv"
CHROMA_DIR = "../chroma_db"
EMBEDDINGS_DIR = "../embeddings"

# ==========================================
# Models
# ==========================================
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
LLM_MODEL = "llama3.1"

# ==========================================
# Runtime
# ==========================================
DEMO_QUERY = "Type 2 Diabetes Mellitus"
RETRIEVAL_LIMIT = 10

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
