import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

# Pod 1 & 2 Retrieval Dependencies
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Pod 3 LangChain Dependencies
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. Config
def find_project_root():
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "snomed_master_v3.csv").exists():
            return current
        current = current.parent
    return Path(__file__).resolve().parent.parent

BASE_DIR = find_project_root()
SNOMED_PATH = BASE_DIR / "snomed_master_v3.csv"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
CHROMA_COLLECTION_NAME = "snomed_master_v3_retrieval"
DEFAULT_TOP_K = 10
# Note: Reranker is optional and omitted for default v1 execution as per plan
USE_RERANKER = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationEngine")

# Single global dependencies (lazy loaded implicitly via first run or eagerly here)
LLM_MODEL = "llama3.1"
_GLOBAL_LLM = None

def get_llm():
    global _GLOBAL_LLM
    if _GLOBAL_LLM is None:
        try:
            _GLOBAL_LLM = ChatOllama(model=LLM_MODEL, temperature=0, format="json") # Using typical LLM instantiation
        except Exception as e:
            logger.warning("Could not initialize local LLM: %s", e)
            _GLOBAL_LLM = False # Marker for failure setup
    return _GLOBAL_LLM if _GLOBAL_LLM is not False else None

# 2. DataLoader
def init_data_resources():
    df, bm25, collection = None, None, None
    try:
        df = pd.read_csv(SNOMED_PATH)
        df = df.dropna(subset=["snomed_code", "term"]).copy()
        df["snomed_code"] = df["snomed_code"].astype(str).str.strip()
        df["term"] = df["term"].astype(str).str.strip()
        df["in_qof"] = df["in_qof"].fillna(False).astype(bool)
        df["in_opencodelists"] = df["in_opencodelists"].fillna(False).astype(bool)
        df["log_usage_nhs"] = pd.to_numeric(df.get("log_usage_nhs", 0), errors="coerce").fillna(0)
        if "usage_count_nhs" not in df.columns:
            df["usage_count_nhs"] = np.exp(df["log_usage_nhs"]) - 1
        df["usage_count_nhs"] = pd.to_numeric(df["usage_count_nhs"], errors="coerce").fillna(0)
        df = df.drop_duplicates(subset=["snomed_code"]).copy()

        def safe_text(x):
            return "" if pd.isna(x) else str(x).strip()

        df["text_for_embedding"] = (
            df["term"].apply(safe_text) + " | " +
            df.get("semantic_tag", pd.Series(dtype=str)).apply(safe_text) + " | " +
            df.get("opencodelist_clinical_areas", pd.Series(dtype=str)).apply(safe_text) + " | " +
            df.get("qof_cluster_description", pd.Series(dtype=str)).apply(safe_text)
        ).str.strip()
        
        tokenized_corpus = [str(t).lower().split() for t in df["text_for_embedding"].fillna("")]
        bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Successfully loaded SNOMED Dataframe and initialized BM25.")
    except Exception as e:
        logger.warning(f"Could not load SNOMED CSV: {e}")

    try:
        client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
        collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
        logger.info(f"Loaded Chroma DB with {collection.count()} vectors.")
    except Exception as e:
        logger.warning(f"Could not load ChromaDB: {e}")

    return df, bm25, collection

_GLOBAL_RESOURCES = None

def get_resources():
    global _GLOBAL_RESOURCES
    if _GLOBAL_RESOURCES is None:
        _GLOBAL_RESOURCES = init_data_resources()
    return _GLOBAL_RESOURCES

# 3. QueryDecomposer
def decompose_query(raw_query: str) -> dict:
    llm = get_llm()
    fallback = {
        "primary_condition": raw_query,
        "comorbidities": [],
        "modifiers": []
    }
    if not llm:
        return fallback

    decomp_prompt = ChatPromptTemplate.from_template("""
You are a clinical NLP assistant.
Break the user's query into precisely 3 parts: primary_condition, comorbidities, and modifiers.

Query: {query}

Return ONLY valid JSON matching this schema:
{{
  "primary_condition": "string without markdown",
  "comorbidities": ["string"],
  "modifiers": ["string"]
}}
""")
    chain = decomp_prompt | llm | StrOutputParser()
    try:
        raw_out = chain.invoke({"query": raw_query})
        clean_out = raw_out.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_out)
    except Exception as e:
        logger.warning(f"Decomposition failed, using fallback. Error: {e}")
        return fallback

# 4. QueryPlanner
def build_search_queries(decomp: dict) -> list[dict]:
    primary = decomp.get("primary_condition", "")
    comorbidities = decomp.get("comorbidities", [])
    modifiers = decomp.get("modifiers", [])
    queries = []

    if primary:
        queries.append({"text": primary, "type": "primary_condition", "weight": 1.0})
    for mod in modifiers:
        if primary:
            queries.append({"text": f"{mod} {primary}", "type": "modifiers", "weight": 0.5})
    for c in comorbidities:
        queries.append({"text": c, "type": "comorbidities", "weight": 0.8})
        for mod in modifiers:
            queries.append({"text": f"{mod} {c}", "type": "modifiers", "weight": 0.4})
    if primary and comorbidities:
        combined = f"{primary} with {' and '.join(comorbidities)}"
        queries.append({"text": combined, "type": "combined", "weight": 0.9})
    
    if not queries and primary:
        queries.append({"text": primary, "type": "primary_condition", "weight": 1.0})
    return queries

# 5. HybridRetriever
def retrieve_candidates(job: dict, top_k: int = 10) -> list[dict]:
    df, bm25, collection = get_resources()
    if df is None or collection is None or bm25 is None:
        return []
    
    query = job["text"]
    q_type = job["type"]
    q_weight = job["weight"]

    # Semantic 
    results = GLOBAL_CHROMA.query(query_texts=[query], n_results=top_k)
    s_scores = {}
    if "ids" in results and results["ids"]:
        for idx, s_id in enumerate(results["ids"][0]):
            dist = results["distances"][0][idx]
            # convert distance to similarity-like roughly
            sim = 1.0 / (1.0 + dist)
            s_scores[s_id] = sim

    # BM25
    toks = query.lower().split()
    b_scores_raw = GLOBAL_BM25.get_scores(toks)
    top_idx = np.argsort(b_scores_raw)[::-1][:top_k]
    b_scores = {}
    for i in top_idx:
        s_id = str(df.iloc[i]["snomed_code"])
        # normalize BM25 via simple max scaling
        b_scores[s_id] = b_scores_raw[i] / (max(b_scores_raw) + 0.001)

    all_ids = set(s_scores.keys()).union(set(b_scores.keys()))
    candidates = []
    
    for s_id in all_ids:
        row = df[df["snomed_code"] == s_id].iloc[0]
        s_sim = s_scores.get(s_id, 0.0)
        b_sim = b_scores.get(s_id, 0.0)
        
        hybrid_score = (s_sim * 0.7) + (b_sim * 0.3)
        weighted_score = hybrid_score * q_weight
        
        candidates.append({
            "snomed_code": s_id,
            "term": row["term"],
            "semantic_tag": str(row.get("semantic_tag", "")),
            "in_qof": bool(row["in_qof"]),
            "in_opencodelists": bool(row["in_opencodelists"]),
            "usage_count_nhs": float(row["usage_count_nhs"]),
            "job_type": q_type,
            "job_weight": q_weight,
            "hybrid_score": hybrid_score,
            "weighted_score": weighted_score
        })
        
    return candidates

# 6. CandidateFusionEngine
def fuse_candidates(retrieval_batches: list[list[dict]]) -> list[dict]:
    fused = defaultdict(lambda: {
        "fusion_score": 0.0,
        "queries_hit": set(),
        "candidate_info": {}
    })
    
    for batch in retrieval_batches:
        for c in batch:
            s_id = c["snomed_code"]
            info = fused[s_id]
            info["fusion_score"] += c["weighted_score"]
            info["queries_hit"].add(c["job_type"])
            if not info["candidate_info"]:
                info["candidate_info"] = {
                    "snomed_code": s_id,
                    "term": c["term"],
                    "semantic_tag": c["semantic_tag"],
                    "in_qof": c["in_qof"],
                    "in_opencodelists": c["in_opencodelists"],
                    "usage_count_nhs": c["usage_count_nhs"]
                }
    
    results = []
    for s_id, val in fused.items():
        doc = dict(val["candidate_info"])
        doc["fusion_score"] = val["fusion_score"]
        doc["query_coverage_count"] = len(val["queries_hit"])
        doc["queries_hit"] = list(val["queries_hit"])
        results.append(doc)
    return results

# Optional reranking hook (Phase 5) omitted per v1 constraints
# def maybe_rerank(user_query, fused_candidates): ...

# 7. DecisionEngine
def assign_confidence(row: dict) -> str:
    """Strict Deterministic Python Logic based purely on generic evidence"""
    in_qof = row.get("in_qof", False)
    in_open = row.get("in_opencodelists", False)
    usage = row.get("usage_count_nhs", 0.0)
    
    if in_qof:
        return "HIGH"
    if in_open and usage >= 29097:
        return "HIGH"
    if in_open or usage >= 6730:
        return "MEDIUM"
    return "REVIEW"

def assign_final_decisions(fused_candidates: list[dict], top_k: int = 10) -> list[dict]:
    """Sorts primarily by fusion_score but assigns separate logical confidence tier."""
    ranked = sorted(fused_candidates, key=lambda x: x["fusion_score"], reverse=True)
    decisions = []
    for row in ranked[:top_k]:
        conf = assign_confidence(row)
        dec_obj = {
            "snomed_code": row["snomed_code"],
            "term": row["term"],
            "semantic_tag": row.get("semantic_tag", ""),
            "in_qof": row["in_qof"],
            "in_opencodelists": row["in_opencodelists"],
            "usage_count_nhs": row["usage_count_nhs"],
            "confidence_tier": conf,
            "fusion_score": round(row["fusion_score"], 4),
            "retrieval_trace": row["queries_hit"]
        }
        decisions.append(dec_obj)
    return decisions

# 8. LLMExplanationFormatter
def explain_final_candidates(user_query: str, decided_candidates: list[dict]) -> str:
    llm = get_llm()
    if not llm:
        return json.dumps(decided_candidates, indent=2)
        
    FORMATTER_PROMPT = """
You are a clinical assistant mapping retrieved SNOMED codes.
DO NOT INVENT CODES. Focus solely on explaining the provided candidate batch.

Original user query: {query}
Deterministically resolved SNOMED candidates:
{candidates}

RETURN ONLY STRICT VALID JSON. Schema constraint per candidate array:
[
  {{
    "snomed_code": "string (must match candidate exactly)",
    "term": "string (must match candidate exactly)",
    "confidence_tier": "string (preserve EXACTLY as provided)",
    "rationale": "Briefly explain why this code applies to the query, given evidence.",
    "evidence": {{
      "in_qof": boolean,
      "in_opencodelists": boolean,
      "usage_count_nhs": number
    }}
  }}
]
"""
    # Exclude internal tracking variables from being sent to LLM payload
    # Drop "fusion_score" and "retrieval_trace" to prevent confusion
    cleaned_cands = []
    for c in decided_candidates:
        cleaned_cands.append({
            "snomed_code": c["snomed_code"],
            "term": c["term"],
            "confidence_tier": c["confidence_tier"],
            "in_qof": c["in_qof"],
            "in_opencodelists": c["in_opencodelists"],
            "usage_count_nhs": c["usage_count_nhs"],
        })
        
    prompt = ChatPromptTemplate.from_template(FORMATTER_PROMPT)
    chain = prompt | llm | StrOutputParser()
    try:
        out = chain.invoke({"query": user_query, "candidates": json.dumps(cleaned_cands)})
        return out.strip().replace("```json", "").replace("```", "")
    except Exception as e:
        logger.warning("Formatting failed! Falling back to raw structured JSON.")
        return json.dumps(decided_candidates, indent=2)

# 9. RunLogger
class RunLogger:
    def __init__(self, run_id, query):
        self.run_id = run_id
        self.query = query
        self.start_time = datetime.now()
        self.logs = {
            "structured_query": None,
            "search_queries": None,
            "fused_count": 0,
            "decisions": None
        }

    def log_structured_query(self, sq):
        self.logs["structured_query"] = sq

    def log_search_queries(self, sq_list):
        self.logs["search_queries"] = sq_list
        logger.info(f"Generated {len(sq_list)} sub-queries for execution.")

    def log_retrieval_batches(self, rb):
        total = sum(len(b) for b in rb)
        logger.info(f"Retrieved total {total} raw candidate hits across sub-queries.")

    def log_fused_candidates(self, fused):
        self.logs["fused_count"] = len(fused)
        logger.info(f"Fused into {len(fused)} unique candidate records.")

    def log_decisions(self, decisions):
        self.logs["decisions"] = decisions

    def finish(self, final_output_str):
        try:
            final_obj = json.loads(final_output_str)
        except json.JSONDecodeError:
            final_obj = {"error": "LLM failed strict JSON formatting", "raw_output": final_output_str}

        audit_packet = {
            "run_id": self.run_id,
            "timestamp": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "query": self.query,
            "trace": self.logs,
            "final_structured_output": final_obj
        }
        
        log_file = BASE_DIR / "agentic_audit_trail.json"
        with open(log_file, "w") as f:
            json.dump(audit_packet, f, indent=2)
        logger.info(f"Pipeline complete. Telemetry exported to {log_file}")
        return final_obj

# 10. EntryPoint
import uuid
def run_pipeline(user_query: str, top_k: int = 10) -> dict:
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"
    tracker = RunLogger(run_id, user_query)
    logger.info(f"Pipeline triggered for: '{user_query}'")

    structured_query = decompose_query(user_query)
    tracker.log_structured_query(structured_query)

    search_queries = build_search_queries(structured_query)
    tracker.log_search_queries(search_queries)

    retrieval_batches = [retrieve_candidates(job, top_k=top_k) for job in search_queries]
    tracker.log_retrieval_batches(retrieval_batches)

    fused_candidates = fuse_candidates(retrieval_batches)
    tracker.log_fused_candidates(fused_candidates)

    decided_candidates = assign_final_decisions(fused_candidates, top_k=top_k)
    tracker.log_decisions(decided_candidates)

    final_output = explain_final_candidates(user_query, decided_candidates)
    return tracker.finish(final_output)

if __name__ == "__main__":
    test_query = "Severe obesity with poorly controlled diabetes and hypertension"
    out = run_pipeline(test_query, top_k=5)
    print("\nFINAL PIPELINE JSON OUTPUT:")
    print(json.dumps(out, indent=2))
