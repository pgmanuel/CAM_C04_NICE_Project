"""
Pod1_Pod2_Integrated_VSCode.py

Fully VS Code / Python compatible.
No Colab or shell commands required.
"""

import json
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# -----------------------------
# Paths and Chroma config
# -----------------------------
CHROMA_COLLECTION_NAME = "snomed_master_v3_retrieval"
CHROMA_PERSIST_DIR = "C:/Users/YourName/NICE/chroma_db"  # Update to your local path
DEFAULT_TOP_K = 10

# -----------------------------
# Initialize Chroma client
# -----------------------------
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
print("Loaded collection:", CHROMA_COLLECTION_NAME)
print("Chroma count:", collection.count())

# -----------------------------
# Initialize Ollama LLM via Python API
# -----------------------------
llm = OllamaLLM(model="phi4-mini")  # Make sure the model is available in Ollama Python

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

decomp_chain = decomp_prompt | llm

# -----------------------------
# Query decomposition
# -----------------------------
def query_decompose(query: str) -> dict:
    raw_output = decomp_chain.invoke({"query": query})
    try:
        cleaned = raw_output.strip().replace("```json", "").replace("```", "")
        return json.loads(cleaned)
    except Exception as e:
        print("JSON parsing failed. Raw output:")
        print(raw_output)
        raise e

def build_sub_queries(structured_query: dict) -> list[str]:
    queries = []
    pc = structured_query.get("primary_condition", "")
    comorbs = structured_query.get("comorbidities", [])

    if pc:
        queries.append(pc)
    queries.extend(comorbs)
    if pc and comorbs:
        queries.append(f"{pc} with {', '.join(comorbs)}")

    return queries

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
    pairs = [(query, doc["document"]) for doc in retrieved_docs]
    scores = reranker.predict(pairs)
    for doc, score in zip(retrieved_docs, scores):
        doc["rerank_score"] = float(score)
    return sorted(retrieved_docs, key=lambda x: x["rerank_score"], reverse=True)[:top_k]

# -----------------------------
# Multi-query Chroma search
# -----------------------------
def chroma_multi_query_search(user_query: str, n_results=15):
    structured = query_decompose(user_query)
    sub_queries = build_sub_queries(structured)

    all_matches = []

    for q in sub_queries:
        embedding = embed_query(q)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances", "ids"]
        )

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