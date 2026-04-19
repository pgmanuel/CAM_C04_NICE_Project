"""
Pod1_Pod2_Integrated_VSCode.py

Fully VS Code / Python compatible.
Dynamically resolves paths so it works seamlessly with app_v4.py.
"""

import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

# 1. ADD DOTENV TO MATCH APP.PY
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent.parent / ".env" # Adjust path to point to root .env
load_dotenv(dotenv_path=_env_path, encoding="utf-8-sig", override=True)

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# -----------------------------
# Dynamic Path Resolution
# -----------------------------
CHROMA_COLLECTION_NAME = "snomed_master_v3_retrieval"

# We dynamically hunt for the 'data' directory relative to this script.
# This ensures it works on your machine, your colleagues' machines, and in production.
_script_dir = Path(__file__).resolve().parent

# Potential locations for the database, in order of preference
_possible_paths = [
    _script_dir.parent / "data" / "chroma_db",                 # If script is in a /src folder
    _script_dir / "data" / "chroma_db",                        # If script is in root next to /data
    Path.cwd() / "data" / "chroma_db",                         # General fallback to current working dir
    Path(r"C:\Users\joeem\CAM_C04_NICE_Project\CAM_Project\data\chroma_db") # Absolute fallback
]

CHROMA_PERSIST_DIR = None
for p in _possible_paths:
    # If the parent folder (e.g., 'data' or 'raw data') exists, this is a valid place to load/create the db
    if p.parent.exists():
        CHROMA_PERSIST_DIR = str(p)
        break

if not CHROMA_PERSIST_DIR:
    # Absolute failsafe if the folder structure is completely unrecognised
    CHROMA_PERSIST_DIR = r"C:\Users\joeem\CAM_C04_NICE_Project\CAM_Project\data\chroma_db"

print(f"[ChromaDB] Using persistence directory: {CHROMA_PERSIST_DIR}")


# -----------------------------
# Initialize Chroma client
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
print("Loaded collection:", CHROMA_COLLECTION_NAME)
print("Chroma count:", collection.count())

# -----------------------------
# Initialize Ollama LLM via Python API
# -----------------------------
# REMOVE OR COMMENT OUT THESE HARDCODED LINES:
# llm = OllamaLLM(model="phi4-mini")
# decomp_chain = decomp_prompt | llm

decomp_prompt = ChatPromptTemplate.from_template("""
You are a clinical NLP assistant.

Break the query into:
- primary_condition
- comorbidities

Query:
{query}

Return ONLY valid JSON in this format:
{{
  "primary_condition": "",
  "comorbidities": [],
}}
""")

# decomp_chain = decomp_prompt | llm

# -----------------------------
# Query decomposition
# -----------------------------
def query_decompose(query: str, model_name: str = "phi4:mini") -> dict:
    # Strip /v1 from the URL so LangChain's native Ollama endpoint works
    raw_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    base_url = raw_url.rstrip("/")       
    if base_url.endswith("/v1"):
        base_url = base_url[:-3]         
        
    llm = OllamaLLM(model=model_name, base_url=base_url)
    decomp_chain = decomp_prompt | llm
    try:
        response = decomp_chain.invoke({"query": query})
        if isinstance(response, dict):
            return response
            
        clean_response = str(response).replace("```json", "").replace("```", "").strip()
        
        # --- ADD THIS SAFETY CHECK ---
        if not clean_response:
            return {"primary_condition": query, "comorbidities": []}
        # -----------------------------
        
        return json.loads(clean_response)
    except Exception as e:
        print(f"[engine] Decomposition failed: {e}")
        return {"primary_condition": query, "comorbidities": []}

# -----------------------------
# Sentence embeddings
# -----------------------------
embedding_model = SentenceTransformer("BAAI/bge-small-en")

def embed_query(text: str):
    return embedding_model.encode(text).tolist()

# -----------------------------
# Cross-encoder reranker
# -----------------------------
reranker = CrossEncoder("BAAI/bge-reranker-base")

def rerank_results(query: str, retrieved_docs: list, top_k: int = 10):
    if not retrieved_docs:
        return []
        
    pairs = [(query, doc["document"]) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(retrieved_docs, scores):
        doc["rerank_score"] = float(score)
    return sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

def build_sub_queries(structured: dict) -> list[str]:
    """Generates sub-queries from the decomposed JSON for the pipeline."""
    queries = []
    primary = structured.get("primary_condition", "")
    if primary:
        queries.append(primary)
    
    comorbidities = structured.get("comorbidities", [])
    for cb in comorbidities:
        if primary:
            queries.append(f"{primary} and {cb}")
        else:
            queries.append(cb)
            
    # If LLM returned nothing, return the original query as a fallback
    return queries if queries else [structured.get("primary_condition", "unknown")]

# -----------------------------
# Multi-query Chroma search
# -----------------------------
def chroma_multi_query_search(user_query: str, n_results=15):
    structured = query_decompose(user_query)
    sub_queries = build_sub_queries(structured)

    all_matches = []

    # Safe fallback if collection is completely empty
    if collection.count() == 0:
        print("[Warning] ChromaDB is empty! Please run your ingestion script first.")
        return {
            "original_query": user_query,
            "structured_query": structured,
            "sub_queries": sub_queries,
            "retrieved_context": []
        }

    for q in sub_queries:
        embedding = embed_query(q)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"] # "ids"]
        )

        # Catch cases where no documents are returned
        if not results["documents"] or not results["documents"][0]:
            continue

        for doc, meta, dist, snomed_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0]
        ):
            all_matches.append({
                "sub_query": q,
                "document": doc,
                "snomed_code": snomed_id,
                "term": meta.get("term"),
                "metadata": meta,
                "distance": dist
            })

    # Deduplicate by SNOMED code
    unique = {}
    for m in all_matches:
        key = m["snomed_code"]
        if key not in unique or m["distance"] < unique[key]["distance"]:
            unique[key] = m

    merged_results = sorted(unique.values(), key=lambda x: x["distance"])
    reranked = rerank_results(user_query, merged_results, top_k=10)

    return {
        "original_query": user_query,
        "structured_query": structured,
        "sub_queries": sub_queries,
        "retrieved_context": reranked
    }

# -----------------------------
# Hybrid reranker
# -----------------------------
def hybrid_rerank(
    query: str,
    retrieved_docs: list,
    top_k: int = 10,
    alpha: float = 0.7,
    beta: float = 0.2,
    gamma: float = 0.1
):
    if not retrieved_docs:
        return []

    semantic_scores = [doc["rerank_score"] for doc in retrieved_docs]
    max_usage = max(float(doc["metadata"].get("usage_count_nhs", 0)) for doc in retrieved_docs) or 1

    for doc, sem_score in zip(retrieved_docs, semantic_scores):
        usage = float(doc["metadata"].get("usage_count_nhs", 0))
        normalized_usage = usage / max_usage
        qof_bonus = gamma if doc["metadata"].get("in_qof", "False") == "True" else 0.0

        doc["hybrid_score"] = alpha * sem_score + beta * normalized_usage + qof_bonus
        doc["result"] = {
            "snomed_code": doc.get("snomed_code"),
            "term": doc.get("term"),
            "hybrid_score": doc["hybrid_score"]
        }

    return sorted(retrieved_docs, key=lambda x: x["hybrid_score"], reverse=True)[:top_k]

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    query = "Severe obesity with poorly controlled diabetes and hypertension"
    results = chroma_multi_query_search(query)
    reranked_results = hybrid_rerank(
        query=results["original_query"],
        retrieved_docs=results["retrieved_context"],
        top_k=10
    )

    for doc in reranked_results:
        print(
            "SNOMED:", doc["snomed_code"],
            "| score:", round(doc["hybrid_score"], 3),
            "| term:", doc["term"]
        )