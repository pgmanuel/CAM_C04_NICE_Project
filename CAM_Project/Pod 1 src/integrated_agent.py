import os
import json
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import warnings

# Suppress noisy warnings
warnings.filterwarnings("ignore")

# Pod 1: Retrieval Dependencies
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Pod 3: LangChain Agent Dependencies
from langchain_ollama import ChatOllama
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

# ============================================================================
# COMPONENT 6: PATH PORTABILITY
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent # Points to CAM_C04_NICE_Project
SNOMED_PATH = BASE_DIR / "snomed_master_v3.csv"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en"
CHROMA_COLLECTION_NAME = "snomed_master_v3_retrieval"
DEFAULT_TOP_K = 10

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("IntegrationEngine")

# ============================================================================
# POD 1: HYBRID RETRIEVAL ENGINE Initialization
# ============================================================================
logger.info("Initializing Hybrid Retrieval Engine...")
logger.info(f"Targeting SNOMED CSV: {SNOMED_PATH}")
logger.info(f"Targeting Chroma DB: {CHROMA_PERSIST_DIR}")

try:
    df = pd.read_csv(SNOMED_PATH)
    df = df.dropna(subset=["snomed_code", "term"]).copy()
    df["snomed_code"] = df["snomed_code"].astype(str).str.strip()
    df["term"] = df["term"].astype(str).str.strip()
    df["in_qof"] = df["in_qof"].fillna(False).astype(bool)
    df["in_opencodelists"] = df["in_opencodelists"].fillna(False).astype(bool)
    df["log_usage_nhs"] = pd.to_numeric(df["log_usage_nhs"], errors="coerce").fillna(0)
    # Ensure raw usage_count_nhs exists, if not construct it from log if original column missing, but it is present in master
    if "usage_count_nhs" not in df.columns:
        df["usage_count_nhs"] = np.exp(df["log_usage_nhs"]) - 1
    
    df["usage_count_nhs"] = pd.to_numeric(df["usage_count_nhs"], errors="coerce").fillna(0)
    
    df = df.drop_duplicates(subset=["snomed_code"]).copy()

    def safe_text(x):
        return "" if pd.isna(x) else str(x).strip()

    df["text_for_embedding"] = (
        df["term"].apply(safe_text) + " | " +
        df["semantic_tag"].apply(safe_text) + " | " +
        df["opencodelist_clinical_areas"].apply(safe_text) + " | " +
        df["qof_cluster_description"].apply(safe_text)
    ).str.strip()
    
    tokenized_corpus = [str(text).lower().split() for text in df["text_for_embedding"].fillna("").tolist()]
    bm25 = BM25Okapi(tokenized_corpus)
    logger.info("Successfully loaded SNOMED Dataframe and initialized BM25.")
except Exception as e:
    logger.warning(f"Could not load SNOMED CSV. System will fail gracefully: {e}")
    df = None
    bm25 = None

try:
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    collection = client.get_collection(name=CHROMA_COLLECTION_NAME)
    logger.info(f"Successfully loaded Chroma DB Collection with {collection.count()} vectors.")
except Exception as e:
    logger.warning(f"Could not load ChromaDB. Ensure path exists: {e}")
    collection = None

# ============================================================================
# COMPONENT 7: RETRIEVAL HEURISTIC ISOLATION
# ============================================================================
def apply_mvp_generic_penalty(merged_df: pd.DataFrame) -> pd.DataFrame:
    """
    [TEMPORARY MVP HEURISTIC]
    Penalizes common generic parent codes so they rank slightly lower in hybrid search.
    This does NOT impact the formal deterministic confidence logic downstream.
    """
    merged_df["generic_penalty"] = merged_df["term"].str.lower().isin([
        "diabetes mellitus (disorder)", "asthma (disorder)",
        "hypertensive disorder (disorder)", "obesity (disorder)"
    ]).astype(int)
    return merged_df

# ============================================================================
# COMPONENT 1: RETRIEVAL OUTPUT CLEANUP
# ============================================================================
def hybrid_retrieve(query: str, n_semantic: int = DEFAULT_TOP_K, n_bm25: int = DEFAULT_TOP_K) -> pd.DataFrame:
    """Combines semantic + BM25, reranks based on NHS/NICE evidence."""
    if df is None or collection is None or bm25 is None:
        return pd.DataFrame({"error": ["Retrieval engines not initialized. (Dataset Missing)"]})

    results = collection.query(query_texts=[query], n_results=n_semantic)
    output = []
    
    if "ids" in results and results["ids"]:
        for i in range(len(results["ids"][0])):
            output.append({
                "snomed_code": results["ids"][0][i],
                "semantic_distance": results["distances"][0][i]
            })
    sem_df = pd.DataFrame(output)
    
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(bm25_scores)[::-1][:n_bm25]
    bm25_df = df.iloc[top_idx].copy()
    bm25_df["bm25_score"] = bm25_scores[top_idx]

    sem_df = sem_df.drop_duplicates(subset=["snomed_code"]).reset_index(drop=True)
    sem_df["semantic_rank_score"] = 1 / (pd.Series(range(1, len(sem_df) + 1)))

    bm25_df = bm25_df.drop_duplicates(subset=["snomed_code"]).reset_index(drop=True)
    bm25_df["bm25_rank_score"] = 1 / (pd.Series(range(1, len(bm25_df) + 1)))

    merged = pd.merge(
        sem_df[["snomed_code", "semantic_distance", "semantic_rank_score"]],
        bm25_df[["snomed_code", "bm25_score", "bm25_rank_score"]],
        on="snomed_code", how="outer"
    )

    merged = merged.merge(df, on="snomed_code", how="left")
    
    merged["hybrid_score"] = merged["semantic_rank_score"].fillna(0) + merged["bm25_rank_score"].fillna(0)
    merged["evidence_score"] = (
        merged["in_qof"].astype(int) * 3 +
        merged["in_opencodelists"].astype(int) * 2 +
        merged["log_usage_nhs"].fillna(0) * 0.5
    )
    
    # MVP Generic tweaks
    merged = apply_mvp_generic_penalty(merged)
    merged["final_score"] = merged["hybrid_score"] + merged["evidence_score"] - merged["generic_penalty"] * 1.0
    merged = merged.sort_values("final_score", ascending=False).reset_index(drop=True)

    # Component 1 check: Reliably return all evidence fields needed downstream
    return merged[["snomed_code", "term", "semantic_tag", "in_qof", "in_opencodelists", "usage_count_nhs"]]


# ============================================================================
# COMPONENT 2 & 3: DETERMINISTIC CONFIDENCE & TOOL RESTRICTION
# ============================================================================
def assign_confidence(row):
    """Component 2: Deterministic Python Layer"""
    in_qof = bool(row.get("in_qof", False))
    in_opencodelists = bool(row.get("in_opencodelists", False))
    usage = float(row.get("usage_count_nhs", 0.0))
    
    if in_qof:
        return "HIGH"
    if in_opencodelists and usage >= 29097:
        return "HIGH"
    if in_opencodelists or usage >= 6730:
        return "MEDIUM"
    return "REVIEW"

@tool
def search_clinical_codes(query: str, n_candidates: int = 5) -> str:
    """
    Search SNOMED concepts related to a clinical condition.
    Returns structured JSON with candidates containing ONLY objective evidence fields.
    """
    logger.info(f"AGENT ACTION => search_clinical_codes(query='{query}', n={n_candidates})")
    results_df = hybrid_retrieve(query, n_semantic=n_candidates, n_bm25=n_candidates)
    
    if "error" in results_df.columns:
        return json.dumps({"error": results_df["error"].iloc[0]})
    
    top_results = results_df.head(n_candidates).copy()
    top_results["confidence_tier"] = top_results.apply(assign_confidence, axis=1)
    
    # Component 3 Check: Restrictions - Drop score attributes.
    cols_to_keep = ["snomed_code", "term", "semantic_tag", "in_qof", "in_opencodelists", "usage_count_nhs", "confidence_tier"]
    clean_dict = top_results[cols_to_keep].fillna("N/A").to_dict(orient="records")
    return json.dumps(clean_dict, indent=2)

# ============================================================================
# COMPONENT 8: AUDIT TRAIL IMPROVEMENT
# ============================================================================
class RunLogger:
    def __init__(self, run_id, query):
        self.run_id = run_id
        self.query = query
        self.start_time = datetime.now()
        self.tool_calls = []
        self.model = "Ollama Local (llama3.1)"
        
    def log_tool(self, action, observation):
        """Records granular detail about tool usages"""
        self.tool_calls.append({
            "action": getattr(action, "tool", "UnknownTool"),
            "observation_preview": str(observation)[:500] + "...",
            "full_observation": observation,
            "timestamp": datetime.now().isoformat()
        })
        logger.info(f"[{self.run_id}][TOOL] Executed {getattr(action, 'tool', 'UnknownTool')} - retrieved candidates.")

    def finish(self, final_output):
        # We try to ensure we can parse the final_decision if it was successfully formatted.
        try:
            parsed_decision = json.loads(final_output)
        except json.JSONDecodeError:
            parsed_decision = final_output # Fallback to raw string if bad JSON
        
        return {
            "run_id": self.run_id,
            "model": self.model,
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "query_received": self.query,
            "tool_telemetry": self.tool_calls,
            "final_structured_output": parsed_decision
        }

# ============================================================================
# COMPONENT 4 & 5: SYSTEM PROMPT & STRUCTURED OUTPUT
# ============================================================================
SYSTEM_PROMPT = """
You are a clinical informatics assistant for NICE.

Your role is to explain and format pre-computed clinical code decisions.
You are NOT allowed to invent new confidence rules or override deterministic logic.

MANDATORY RULES:
1. Always call `search_clinical_codes` first for each clinical concept in the user query.
2. You must not recommend any SNOMED code that was not returned by the tool.
3. You will receive candidate codes that already include:
   - in_qof
   - in_opencodelists
   - usage_count_nhs
   - confidence_tier
4. You must preserve the provided confidence_tier exactly as given.
5. Your job is to explain the decision using only the returned evidence fields.
6. Do not upgrade, downgrade, or reinterpret any confidence tier.
7. If evidence is weak or ambiguous, state that clearly in the rationale.
8. Return results in the required structured schema only.

REQUIRED SCHEMA (Output ONLY valid JSON matching this structure array):
[
  {{
    "snomed_code": "string",
    "term": "string",
    "confidence_tier": "HIGH | MEDIUM | REVIEW",
    "rationale": "string",
    "evidence": {{
      "in_qof": true,
      "in_opencodelists": false,
      "usage_count_nhs": 12345
    }}
  }}
]
"""

def run_agentic_workflow(user_query: str):
    tracker = RunLogger("run_001", user_query)
    logger.info(f"[{tracker.run_id}][SYSTEM] Initialization started.")

    try:
        # LLM binding natively ensures tool passing.
        llm = ChatOllama(model="llama3.1", temperature=0)
    except Exception as e:
        logger.error(f"Cannot initialize ChatOllama: {e}")
        return

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT), 
        ("human", "{input}"), 
        ("placeholder", "{agent_scratchpad}")
    ])
    
    tools = [search_clinical_codes]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        max_iterations=8, 
        return_intermediate_steps=True
    )
    
    logger.info(f"[{tracker.run_id}][AGENT] Execution Start - Query: {user_query}")
    try:
        result = agent_executor.invoke({"input": user_query})
        
        for step in result.get("intermediate_steps", []):
            action, observation = step
            tracker.log_tool(action, observation)
            
        final_out = tracker.finish(result["output"])
    except Exception as e:
        logger.error(f"Agent failed pipeline: {e}")
        final_out = tracker.finish(f'[{{"error": "Agent crashed during execution: {str(e)}"}} ]')

    print("\n" + "="*80)
    print("FINAL STRUCTURED OUTPUT:")
    print("="*80)
    if isinstance(final_out["final_structured_output"], (list, dict)):
        print(json.dumps(final_out["final_structured_output"], indent=2))
    else:
        print(final_out["final_structured_output"])
        
    log_file = "agentic_audit_trail.json"
    with open(log_file, "w") as f:
        json.dump(final_out, f, indent=2)
    logger.info(f"Agent run completed. Telemetry exported -> {log_file}")

if __name__ == "__main__":
    test_query = "Build a targeted code list for patients having Type 2 Diabetes."
    run_agentic_workflow(test_query)
