"""
pipeline.py
-----------
Retrieval and Ranking Engine — the only file app.py needs for queries.

WHAT THIS FILE DOES
-------------------
This file is the bridge between app.py (the UI) and the underlying
retrieval engine (pod1_pod2_integrated_V2.py). It:

  1. Connects to the existing ChromaDB database at startup
  2. Exposes retrieve_and_rank() to search and score codes
  3. Exposes add_llm_explanations() to explain each code
  4. Exposes is_ready() and db_code_count() for UI status display

THIS FILE DOES NOT BUILD THE DATABASE
--------------------------------------
ingest_data.py builds the database. That runs ONCE (or when source
data changes). This file ONLY READS from the existing database.

STARTUP FAILURE DIAGNOSIS
--------------------------
If is_ready() returns False, it means ONE of three things:
  a) DB missing or empty         → run ingest_data.py
  b) Ollama not running          → run `ollama serve`
  c) Python packages missing     → pip install (shown in error msg)

These are detected separately so the user gets a specific message.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


# ── Step 1: Try full engine import ────────────────────────────────
# pod1_pod2_integrated_V2 initialises OllamaLLM at module level.
# If Ollama is offline, this may raise — we catch it gracefully.

_engine       = None
_engine_error = ""

try:
    import pod1_pod2_integrated_V2 as _engine
    print(f"[pipeline] Engine loaded. ChromaDB: {_engine.collection.count():,} codes.")
except ImportError as e:
    _engine_error = f"MISSING_PACKAGE:{e}"
    print(f"[pipeline] Import error (missing packages): {e}")
except Exception as e:
    _engine_error = f"OLLAMA_OFFLINE:{type(e).__name__}:{e}"
    print(f"[pipeline] Engine startup error (Ollama may be offline): {e}")


# ── Step 2: Direct ChromaDB check (independent of Ollama) ─────────
# This verifies whether the database exists and has codes,
# even when the full engine import failed.

_db_direct_count = 0
_db_direct_error = ""


def _check_db_directly() -> int:
    global _db_direct_error
    try:
        import chromadb  # type: ignore
        script_dir = Path(__file__).resolve().parent
        paths = [
            script_dir.parent / "data" / "chroma_db",
            script_dir / "data" / "chroma_db",
            Path.cwd() / "data" / "chroma_db",
            Path(r"C:\Users\joeem\CAM_C04_NICE_Project\CAM_Project\data\chroma_db"),
        ]
        for p in paths:
            if p.exists():
                try:
                    col = chromadb.PersistentClient(path=str(p)).get_collection("snomed_master_v3_retrieval")
                    n   = col.count()
                    print(f"[pipeline] Direct DB check: {n:,} codes at {p}")
                    return n
                except Exception:
                    continue
        _db_direct_error = "ChromaDB collection not found at any known path."
        return 0
    except ImportError:
        _db_direct_error = "chromadb package not installed."
        return 0
    except Exception as e:
        _db_direct_error = str(e)
        return 0


_db_direct_count = _check_db_directly()


# ── Public status API ────────────────────────────────────────────

def is_ready() -> tuple[bool, str]:
    """
    Returns (True, "") if the pipeline can serve queries.
    Returns (False, reason) with a plain-English explanation if not.
    """
    count = db_code_count()
    if count == 0:
        if _db_direct_error:
            return False, (
                f"Database error: {_db_direct_error}\n\n"
                "Ensure ingest_data.py has been run and the chroma_db folder exists."
            )
        return False, (
            "ChromaDB is empty.\n\n"
            "Run `python ingest_data.py` once to build the database.\n"
            "This only needs to be done once (or when source data changes)."
        )

    if _engine is None:
        if "MISSING_PACKAGE" in _engine_error:
            pkg = _engine_error.split(":", 1)[1] if ":" in _engine_error else _engine_error
            return False, (
                f"Missing packages: {pkg}\n\n"
                "Run: pip install sentence-transformers chromadb langchain-ollama langchain-core"
            )
        return False, (
            f"The database has {count:,} codes but the engine needs Ollama.\n\n"
            "Start Ollama:\n"
            "  1. Run `ollama serve` in a terminal\n"
            "  2. Run `ollama pull phi4:mini` if you haven't already\n"
            "  3. Restart app.py\n\n"
            "You do NOT need to re-run ingest_data.py."
        )

    return True, ""


def db_code_count() -> int:
    """Number of codes in ChromaDB. Uses engine count if available."""
    if _engine is not None:
        try:
            return _engine.collection.count()
        except Exception:
            pass
    return _db_direct_count


# ── Query decomposition with Ollama fallback ─────────────────────

def _safe_decompose(query: str, model_choice: str) -> dict: # Add model_choice here
    """
    Decompose query via Ollama LLM. Falls back to simple text split
    on 'with'/'and' if Ollama is unavailable at query time.
    """
    if _engine is None:
        return {"primary_condition": query, "comorbidities": []}
    try:
        # Pass the model_choice through to the engine
        return _engine.query_decompose(query, model_name=model_choice)
    except Exception as e:
        print(f"[pipeline] query_decompose fallback: {e}")
        parts = re.split(r'\bwith\b|\band\b', query, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        return {"primary_condition": parts[0], "comorbidities": parts[1:]} if len(parts) > 1 \
               else {"primary_condition": query, "comorbidities": []}


# ── Main retrieval function ───────────────────────────────────────

def retrieve_and_rank(query: str, model_choice: str, top_k: int = 10) -> dict:
    """
    Search ChromaDB and return ranked clinical codes.

    Steps:
      1. Decompose query → primary condition + comorbidities
      2. Build sub-queries (one per condition + one combined)
      3. Embed each sub-query with BGE model → vector
      4. Search ChromaDB for nearest 15 codes per sub-query
      5. Deduplicate by SNOMED code
      6. CrossEncoder reranking (more accurate relevance scoring)
      7. Hybrid scoring: 70% semantic + 20% NHS usage + 10% QOF
      8. Return top_k codes as structured report items

    Returns dict with:
        items             list[dict]  — ranked code records
        sub_queries       list[str]   — sub-queries that were searched
        primary_condition str         — extracted primary condition
        comorbidities     list[str]   — extracted comorbidities
        engine_ready      bool        — whether the engine was available
        error             str         — empty if successful

    Each item contains: code, term, rank, confidence_score, in_qof,
    usage_count, semantic_score, sub_query_found, explanation (empty)
    """
    ready, reason = is_ready()
    if not ready:
        return {"items": [], "sub_queries": [], "primary_condition": query,
                "comorbidities": [], "engine_ready": False, "error": reason}

    structured  = _safe_decompose(query, model_choice)
    sub_queries = _engine.build_sub_queries(structured)

    all_matches: list[dict] = []
    for sq in sub_queries:
        try:
            emb  = _engine.embed_query(sq)
            # ── CHROMADB INCLUDE FIX ──────────────────────────────────────
            # ChromaDB >= 0.4.x does NOT accept "ids" in the include list.
            # IDs are always returned by default regardless of include[].
            # Valid options: documents, embeddings, metadatas, distances, uris, data
            # Passing "ids" raises:
            #   "Expected include item to be one of ... got ids in query"
            # res["ids"][0] below still works — IDs always come back automatically.
            # ──────────────────────────────────────────────────────────────
            res  = _engine.collection.query(
                query_embeddings=[emb], n_results=15,
                include=["documents", "metadatas", "distances"]
            )
            if not res["documents"] or not res["documents"][0]:
                continue
            for doc, meta, dist, sid in zip(
                res["documents"][0], res["metadatas"][0],
                res["distances"][0],  res["ids"][0]
            ):
                all_matches.append({
                    "sub_query": sq, "document": doc, "snomed_code": sid,
                    "term": meta.get("term"), "metadata": meta, "distance": dist,
                })
        except Exception as e:
            print(f"[pipeline] Sub-query '{sq}' error: {e}")

    if not all_matches:
        return {"items": [], "sub_queries": sub_queries,
                "primary_condition": structured.get("primary_condition", query),
                "comorbidities": structured.get("comorbidities", []),
                "engine_ready": True, "error": ""}

    # Deduplicate: keep closest distance per code
    unique: dict[str, dict] = {}
    for m in all_matches:
        k = m["snomed_code"]
        if k not in unique or m["distance"] < unique[k]["distance"]:
            unique[k] = m
    merged = sorted(unique.values(), key=lambda x: x["distance"])

    # CrossEncoder reranking
    try:
        reranked = _engine.rerank_results(query, merged, top_k=min(20, len(merged)))
    except Exception as e:
        print(f"[pipeline] CrossEncoder fallback: {e}")
        for d in merged:
            d["rerank_score"] = 1.0 / (1.0 + d.get("distance", 1.0))
        reranked = sorted(merged, key=lambda x: x.get("rerank_score", 0), reverse=True)[:20]

    # Hybrid scoring
    try:
        final = _engine.hybrid_rerank(query=query, retrieved_docs=reranked,
                                      top_k=top_k, alpha=0.7, beta=0.2, gamma=0.1)
    except Exception as e:
        print(f"[pipeline] Hybrid fallback: {e}")
        for d in reranked[:top_k]:
            d["hybrid_score"] = d.get("rerank_score", 0.5)
        final = reranked[:top_k]

    # Build standard report items
    items = []
    for pos, doc in enumerate(final, start=1):
        meta = doc.get("metadata", {})
        term = doc.get("term") or meta.get("term", "Unknown term")
        in_qof = str(meta.get("in_qof", "False")).lower() == "true"
        try:
            usage = int(float(meta.get("usage_count_nhs", "0")))
        except (ValueError, TypeError):
            usage = 0
        items.append({
            "code": doc["snomed_code"], "term": term,
            "flag": "CANDIDATE_INCLUDE",
            "rank": pos, "priority": pos,
            "score":            round(float(doc.get("hybrid_score", 0.0)), 4),
            "confidence_score": round(float(doc.get("hybrid_score", 0.0)), 4),
            "ranked_by":        "Hybrid: Semantic 70% + NHS Usage 20% + QOF 10%",
            "in_qof":           in_qof,
            "usage_count":      usage,
            "semantic_score":   round(float(doc.get("rerank_score", 0.0)), 4),
            "sub_query_found":  doc.get("sub_query", ""),
            "explanation": "", "evidence": [],
        })

    return {
        "items": items, "sub_queries": sub_queries,
        "primary_condition": structured.get("primary_condition", query),
        "comorbidities": structured.get("comorbidities", []),
        "engine_ready": True, "error": "",
    }


# ── LLM explanation layer ────────────────────────────────────────

def add_llm_explanations(report: dict, query: str) -> dict:
    """
    Ask the selected LLM to explain each retrieved code.

    The LLM is instructed it cannot add codes or change rankings.
    If it fails, codes get a safe generic explanation instead.
    """
    if not report.get("items"):
        return report

    try:
        from llm import call_llm

        codes_text = "\n".join(
            f"- {i['term']} ({i['code']})"
            + (" [QOF MANDATED]" if i.get("in_qof") else "")
            + (f" [NHS usage: {i['usage_count']:,}]" if i.get("usage_count") else "")
            for i in report["items"]
        )

        system = (
            "You are a clinical coding assistant for NICE.\n\n"
            "RULES (follow exactly):\n"
            "- DO NOT add new codes\n"
            "- DO NOT change the order\n"
            "- 1-2 sentences per code maximum\n"
            "- Mention QOF mandate if [QOF MANDATED] is shown\n"
            "- Return ONLY valid JSON, nothing else\n\n"
            'Format: [{"code": "...", "explanation": "..."}, ...]'
        )
        user = f"Query: {query}\n\nCodes:\n{codes_text}\n\nReturn ONLY JSON."

        raw = call_llm(system, user)
        c   = raw.strip()
        if c.startswith("```"):
            c = re.sub(r"^```(?:json)?", "", c).strip()
            c = re.sub(r"```$", "", c).strip()

        emap = {str(p.get("code","")): p.get("explanation","") for p in json.loads(c)}
        for item in report["items"]:
            e = emap.get(str(item["code"]), "")
            if e:
                item["explanation"] = e

    except Exception as e:
        print(f"[pipeline] LLM explanations failed: {type(e).__name__}: {e}")

    for item in report["items"]:
        if not item.get("explanation"):
            qn = " QOF-mandated." if item.get("in_qof") else ""
            item["explanation"] = f"Relevant to '{query}'.{qn}"

    return report


# ── Ingestion trigger ────────────────────────────────────────────

def run_ingestion() -> tuple[bool, str]:
    """
    Run ingest_data.py as a subprocess (rarely needed).
    Only call this if the DB is missing or the source CSV changed.
    Normal use: run `python ingest_data.py` once from the terminal.
    """
    script = Path(__file__).parent / "ingest_data.py"
    if not script.exists():
        return False, f"ingest_data.py not found at {script}"
    try:
        r = subprocess.run([sys.executable, str(script)],
                           capture_output=True, text=True, timeout=600)
        return (True, r.stdout[-500:]) if r.returncode == 0 \
               else (False, r.stderr[-500:])
    except subprocess.TimeoutExpired:
        return False, "Ingestion timed out (10 min)."
    except Exception as e:
        return False, str(e)
