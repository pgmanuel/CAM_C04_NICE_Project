# NICE Clinical Code Assistant — Complete Technical Reference

> **How to read this document.** Think of the system as a three-floor building. The **Ground Floor** is the database and ML models — heavy machinery that lives on disk. The **Middle Floor** is the pipeline — it organises all the work. The **Top Floor** is the UI — the only thing the user sees. This guide starts at the top and works down, explaining every file, every function, every input and output, and every connection between them.

---

## Quick-Start

```bash
pip install gradio sentence-transformers chromadb langchain-ollama langchain-core python-dotenv openai

python ingest_data.py          # once — builds the vector database

ollama pull phi4:mini          # query decomposition model
ollama pull llama3.2:1b        # explanation model (or any model you prefer)

# Edit .env:
# OLLAMA_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=llama3.2:1b

python app.py                  # open http://localhost:7860
```

---

## File Map

| File | Who runs it | When | Purpose |
|---|---|---|---|
| `app.py` | You | Every session | Gradio UI + event wiring |
| `pipeline.py` | Imported by app.py | On startup | Clean API over the ML engine |
| `pod1_pod2_integrated_V2.py` | Imported by pipeline.py | On startup | ML models + ChromaDB search + hybrid ranking |
| `llm.py` | Imported by pipeline.py + ragas_eval | On each query | Sends messages to the LLM |
| `ragas_eval.py` | Imported by app.py | After each query | Quality metrics |
| `reasoning_eval.py` | Imported by app.py | After each query | Step-by-step reasoning trace |
| `feedback_hitl.py` | Imported by app.py | On thumbs click | Saves analyst feedback |
| `app_audit.py` | Imported by app.py | On each query | Audit trail + validation flags |
| `ingest_data.py` | You | Once | Builds ChromaDB from SNOMED CSV |
| `service.py` | Only by demo.py | On CLI test | Thin wrapper calling llm.py via prompts.md |
| `demo.py` | You (optionally) | CLI testing | Standalone query test without the UI |
| `prompts.md` | Read by service.py | On CLI test | System prompt for service.py's LLM calls |

---

## End-to-End Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  USER  types "obesity with type 2 diabetes"  →  clicks Search       │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  app.py  ·  process_query(message, history, model_choice)           │
│                                                                     │
│  1. _audit.start_run(message, model_choice, "ChromaDB Hybrid")      │
│     └─ creates RunRecord in memory, returns run_id                  │
│                                                                     │
│  2. pipeline.is_ready()                                             │
│     ├─ checks _engine loaded?                                       │
│     └─ checks ChromaDB has codes?                                   │
│                                                                     │
│  3. pipeline.retrieve_and_rank(query, model_choice, top_k=10)       │
│     │                                                               │
│     │  ┌──────────────────────────────────────────────────────┐    │
│     │  │  pod1_pod2_integrated_V2.py                          │    │
│     │  │                                                      │    │
│     │  │  a) query_decompose(query, model_choice)             │    │
│     │  │     └─ Ollama phi4:mini via LangChain                │    │
│     │  │        → {primary:"obesity",                         │    │
│     │  │           comorbidities:["type 2 diabetes"]}         │    │
│     │  │                                                      │    │
│     │  │  b) build_sub_queries(structured_query)              │    │
│     │  │     → ["obesity", "type 2 diabetes",                 │    │
│     │  │        "obesity with type 2 diabetes"]               │    │
│     │  │                                                      │    │
│     │  │  c) For each sub-query:                              │    │
│     │  │     embed_query(sub_query)                           │    │
│     │  │     └─ BAAI/bge-small-en SentenceTransformer         │    │
│     │  │        → [384 floats]                                │    │
│     │  │     chromadb.collection.query(embedding, n=15)       │    │
│     │  │        → 15 nearest SNOMED codes by cosine distance  │    │
│     │  │                                                      │    │
│     │  │  d) Deduplicate by SNOMED code                       │    │
│     │  │                                                      │    │
│     │  │  e) rerank_results(query, candidates, top_k)         │    │
│     │  │     └─ BAAI/bge-reranker-base CrossEncoder           │    │
│     │  │        → rerank_score added to each doc              │    │
│     │  │                                                      │    │
│     │  │  f) hybrid_rerank(query, docs, top_k, 0.7, 0.2, 0.1)│    │
│     │  │     → hybrid = 0.7×semantic + 0.2×usage + 0.1×qof   │    │
│     │  │     → top 10 sorted by hybrid score                  │    │
│     │  └──────────────────────────────────────────────────────┘    │
│     └─ returns report dict                                          │
│                                                                     │
│  4. pipeline.add_llm_explanations(report, query)                    │
│     └─ llm.call_llm(system_prompt, json(report_items))              │
│        └─ Ollama llama3.2:1b (or model from dropdown)               │
│           → JSON list of {code, explanation}                        │
│                                                                     │
│  5. _format_codes_response(report, model_choice)                    │
│     → Markdown string for chat bubble                               │
│       (all text locked to 1rem via CSS — no size jumps)             │
│                                                                     │
│  6. _audit.finish_run(run_id, report["items"])                      │
│     → CodeProvenance recorded per code                              │
│     → 3 validation checks run                                       │
│     _audit.save(run_id)                                             │
│     → outputs/run_logs/{run_id}.json written                        │
│                                                                     │
│  7. ragas_eval.evaluate(report, query)                              │
│     ├─ _score_faithfulness()   → LLM judge or heuristic             │
│     ├─ _score_answer_relevancy() → uses existing rerank_scores      │
│     └─ _score_context_recall()  → sub-query coverage check         │
│                                                                     │
│  8. reasoning_eval.generate_reasoning_trace(report, query)          │
│     → 5-section Markdown document                                   │
│                                                                     │
│  Returns tuple (5 values) → Gradio maps to 5 UI components:        │
│    new_history  → gr.Chatbot                                        │
│    eval_md      → 📊 Evaluation Metrics accordion                   │
│    reasoning_md → 🔍 Reasoning Trace accordion                      │
│    audit_md     → 🗂️ Audit Trail accordion                          │
│    run_id       → gr.State (invisible, used by feedback buttons)    │
└─────────────────────────────────────────────────────────────────────┘
                                │
          ┌─────────────────────┴──────────────────────┐
          │  USER clicks 👍 or 👎                       │
          ▼                                             │
┌─────────────────────────────┐                        │
│  app.py · handle_feedback() │                        │
│  └─ feedback_hitl.record_feedback()                  │
│     ├─ outputs/feedback/{run_id}_th.json             │
│     └─ outputs/feedback/feedback_summary.csv         │
│  → updates 👍👎 Feedback Summary accordion           │
└─────────────────────────────┘
```

---

## File 1: `ingest_data.py` — Database Builder

**Run once before first use. Run again only when the SNOMED CSV changes.**

This file reads every row in the master SNOMED CSV, converts each code description into a 384-float vector (a mathematical representation of its clinical meaning), and saves everything to a ChromaDB database on disk.

### Why vectors?

When you search "obesity", you want to find "BMI ≥30" too, even though the words differ. Keyword search cannot do this. Vector similarity search can: "obesity" and "BMI ≥30" produce vectors that are close together in 384-dimensional space because both phrases appear in similar clinical contexts in the training data.

### Key variables

| Variable | Default | Meaning |
|---|---|---|
| `BASE_DIR` | `data/` | Root of the data folder |
| `INPUT_CSV` | `data/processed_data/snomed/snomed_master_v3.csv` | Source SNOMED data |
| `DB_DIR` | `data/chroma_db/` | Where ChromaDB is stored |
| `COLLECTION_NAME` | `snomed_master_v3_retrieval` | Name of the collection inside ChromaDB |
| `batch_size` | `5000` | Rows per ChromaDB write call |

### Functions

#### `main() → None`

The only function. Does everything in order:

```python
# Step 1 — Read CSV
df = pd.read_csv(INPUT_CSV)
# Input:  CSV file with columns: snomed_code, term, semantic_tag, in_qof, usage_count_nhs
# Output: pandas DataFrame, one row per code

# Step 2 — Build text for embedding
df["text_for_ai"] = df["term"] + " (" + df["semantic_tag"] + ")"
# Input:  "Type 2 diabetes mellitus", "finding"
# Output: "Type 2 diabetes mellitus (finding)"
# Why:    The semantic tag (finding/disorder/observable) adds clinical
#         context that improves embedding quality

# Step 3 — Embed all texts
model = SentenceTransformer("BAAI/bge-small-en")
embeddings = model.encode(df["text_for_ai"].tolist(), batch_size=256)
# Input:  list of ~500,000 text strings
# Output: numpy array of shape (N, 384)
# Time:   ~20 minutes on CPU, ~3 minutes on GPU

# Step 4 — Save to ChromaDB in batches
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)
for i in range(0, len(df), batch_size):
    collection.add(
        ids=        df["snomed_code"][i:i+batch_size].tolist(),
        embeddings= embeddings[i:i+batch_size].tolist(),
        documents=  df["text_for_ai"][i:i+batch_size].tolist(),
        metadatas=  [{"term":t, "in_qof":q, "usage_count_nhs":u} ...],
    )
```

**Input:** `data/processed_data/snomed/snomed_master_v3.csv`

**Output:** `data/chroma_db/` directory containing ChromaDB files

---

## File 2: `pod1_pod2_integrated_V2.py` — ML Search and Ranking Engine

**Never run directly. Imported by `pipeline.py` at startup.**

This file is the technical core. It owns all ML model logic: the embedding model (for converting text to vectors), the cross-encoder reranker (for accurate relevance scoring), and the query decomposition LLM (for splitting multi-condition queries).

### What happens at import time

When Python imports this file, ALL of the following runs immediately (before any function is called):

```python
# These all execute at module level — at the moment of import
client     = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
collection = client.get_or_create_collection(COLLECTION_NAME)
embedding_model = SentenceTransformer("BAAI/bge-small-en")   # 130MB loaded
reranker        = CrossEncoder("BAAI/bge-reranker-base")       # 270MB loaded
# OllamaLLM init was moved INSIDE query_decompose() to prevent crash if Ollama offline
```

### Path resolution logic

The engine tries four folder locations for `chroma_db/` in order:

```python
_CANDIDATE_PATHS = [
    Path(__file__).parent / "data" / "chroma_db",
    Path(__file__).parent.parent / "data" / "chroma_db",
    Path.home() / "data" / "chroma_db",
    Path("/absolute/fallback/path/data/chroma_db"),
]
CHROMA_PERSIST_DIR = next(
    (str(p) for p in _CANDIDATE_PATHS if p.parent.exists()), str(_CANDIDATE_PATHS[0])
)
```

This is why the app works on different machines and in different project structures.

### Functions

#### `query_decompose(query: str, model_choice: str) → dict`

Breaks a multi-condition query into structured clinical concepts.

```python
# Input
query        = "obesity with type 2 diabetes and hypertension"
model_choice = "llama3.2:1b"

# What it does
llm   = OllamaLLM(model=model_choice)   # created here, not at import time
chain = decomp_prompt | llm
raw   = chain.invoke({"query": query})
result = json.loads(raw)

# Output
{
    "primary_condition": "obesity",
    "comorbidities": ["type 2 diabetes", "hypertension"]
}
```

**Why decompose?** Searching "obesity with type 2 diabetes and hypertension" as a single string produces a vector that is an average of all three concepts. It is less specific than searching each concept separately. Decomposition improves recall.

**Fallback:** If this fails (Ollama offline), `pipeline._safe_decompose()` splits on "with" and "and" instead.

---

#### `build_sub_queries(structured_query: dict) → list[str]`

Converts the decomposition result into a list of search strings.

```python
# Input
structured_query = {
    "primary_condition": "obesity",
    "comorbidities": ["type 2 diabetes", "hypertension"]
}

# Output
[
    "obesity",
    "type 2 diabetes",
    "hypertension",
    "obesity with type 2 diabetes, hypertension"  # combined string last
]
```

---

#### `embed_query(text: str) → list[float]`

Converts a text string to a 384-dimensional vector using the BGE embedding model.

```python
# Input
text = "type 2 diabetes"

# What it does
vector = embedding_model.encode(text)

# Output
[0.023, -0.147, 0.892, ...]  # 384 floats
```

---

#### `chroma_multi_query_search(user_query: str, n_results: int) → dict`

Searches ChromaDB for each sub-query and merges the results.

```python
# Input
user_query = "obesity with type 2 diabetes"
n_results  = 15   # per sub-query

# Internal steps:
# 1. Calls query_decompose → structured_query
# 2. Calls build_sub_queries → sub_queries list
# 3. For each sub_query:
#      embed_query(sub_query) → vector
#      collection.query(vector, n_results=15) → 15 nearest codes
# 4. Deduplicates: if same code found by multiple sub-queries,
#    keeps the entry with the LOWEST distance (highest similarity)
# 5. Calls rerank_results()

# Output dict
{
    "original_query":   "obesity with type 2 diabetes",
    "structured_query": {"primary_condition": "obesity", ...},
    "sub_queries":      ["obesity", "type 2 diabetes", ...],
    "retrieved_context": [
        {
            "id":           "44054006",
            "document":     "Type 2 diabetes mellitus (finding)",
            "distance":     0.21,
            "rerank_score": 4.7,
            "metadata":     {"term": "...", "in_qof": "True", "usage_count_nhs": 350000},
            "sub_query_found": "type 2 diabetes"
        },
        ...
    ]
}
```

---

#### `rerank_results(query: str, retrieved_docs: list, top_k: int) → list`

Applies the CrossEncoder model to score each (query, code) pair.

```python
# Input
query         = "obesity with type 2 diabetes"
retrieved_docs = [{"id": "44054006", "document": "Type 2 diabetes mellitus (finding)", ...}]
top_k         = 20

# What it does — CrossEncoder reads BOTH texts together:
pairs  = [(query, doc["document"]) for doc in retrieved_docs]
scores = reranker.predict(pairs)   # BAAI/bge-reranker-base
for doc, score in zip(retrieved_docs, scores):
    doc["rerank_score"] = float(score)

# Output — same list, with rerank_score added, sorted descending, truncated to top_k
# CrossEncoder scores typically range from -5 to +10; higher = more relevant
```

**Why a CrossEncoder after embedding search?** The embedding model encodes query and document independently and compares them as vectors. The CrossEncoder reads both texts simultaneously, which is 10–100× slower but significantly more accurate. Using it only on the shortlist (top 20) gives accuracy without the speed cost.

---

#### `hybrid_rerank(query, retrieved_docs, top_k, alpha=0.7, beta=0.2, gamma=0.1) → list`

Computes the final hybrid score combining three signals.

```python
# Formula
hybrid_score = (alpha × semantic_score_normalised)
             + (beta  × usage_normalised)
             + (gamma × qof_bonus)

# Where:
# semantic_score_normalised  = rerank_score / max(rerank_score in results)
# usage_normalised           = usage_count_nhs / max(usage_count_nhs in results)
# qof_bonus                  = gamma (0.1) if in_qof == "True", else 0

# Input example — one doc dict
{
    "id": "44054006",
    "rerank_score": 4.7,          # from CrossEncoder
    "metadata": {
        "usage_count_nhs": 350000,
        "in_qof": "True",
    }
}

# Output example — same doc, hybrid_score added
{
    "id":            "44054006",
    "term":          "Type 2 diabetes mellitus",
    "rerank_score":  4.7,
    "hybrid_score":  0.912,        # 0.7×0.93 + 0.2×1.0 + 0.1×1.0
    "in_qof":        True,
    "usage_count":   350000,
    "rank":          1,
}
```

**Why 70/20/10?** Clinical relevance to the query is the primary selection criterion (70%). NHS usage frequency ensures mainstream codes appear (20%). The QOF bonus nudges nationally mandated codes to the top when relevance is similar (10%). These weights can be tuned in `pipeline.py` or via backtest comparison.

---

## File 3: `pipeline.py` — Clean API Layer

**Never run directly. Imported by `app.py`.**

This is the "middle floor" — a thin layer that sits between the UI and the ML engine. It exists for three reasons:

1. **Isolation:** `app.py` calls `pipeline.retrieve_and_rank()` — a simple, stable function. If the ML engine is rewritten, only `pipeline.py` changes, not the UI.
2. **Graceful failure:** Every ML step is wrapped in `try/except`. If the CrossEncoder fails, the app falls back to raw embedding scores. If Ollama is offline, it falls back to keyword splitting. The UI never crashes.
3. **Accurate health checks:** The previous `is_ready()` only checked whether the engine import succeeded. This caused false "DB not built" errors when Ollama was offline. The fixed version does a **direct ChromaDB check** independently of the engine.

### Module-level startup

```python
# Runs once when app.py imports pipeline

try:
    import pod1_pod2_integrated_V2 as _engine
    _engine_error = None
except ImportError as e:
    _engine_error = f"MISSING_PACKAGE:{e}"
except Exception as e:
    _engine_error = f"OLLAMA_OFFLINE:{e}"   # Ollama not running

# Direct DB check — works even if _engine is None
_db_direct_count = _check_db_directly()
```

### Functions

#### `is_ready() → tuple[bool, str]`

Checks whether the system can handle queries.

```python
# Returns (True, "") when everything is working

# Returns (False, reason_string) when something is wrong.
# reason_string is shown to the user in the chat window.

# Possible outputs:
(True,  "")
(False, "DB empty — run ingest_data.py")
(False, "Start Ollama — DB has 487,234 codes")
(False, "Missing packages — run pip install ...")
```

---

#### `db_code_count() → int`

Returns the number of SNOMED codes in ChromaDB.

```python
# Input:  none
# Output: integer (e.g. 487234)
# Used by: app.py to show the green/amber/red status pill in the header
```

---

#### `_check_db_directly() → int`

Internal function. Opens ChromaDB directly (without the engine) and counts codes.

```python
# Input:  none
# Output: int code count, or 0 if DB not found
# Why:    Allows is_ready() to work even when the engine import failed
```

---

#### `_safe_decompose(query: str, model_choice: str) → dict`

Tries the ML query decomposition, falls back to keyword splitting.

```python
# Input
query        = "hypertension and CKD"
model_choice = "llama3.2:1b"

# Path 1 — engine available and Ollama running
result = _engine.query_decompose(query, model_choice)
# → {"primary_condition": "hypertension", "comorbidities": ["CKD"]}

# Path 2 — engine unavailable or Ollama offline
parts  = re.split(r"\bwith\b|\band\b", query.lower())
result = {"primary_condition": parts[0].strip(), "comorbidities": parts[1:]}
# → {"primary_condition": "hypertension", "comorbidities": ["ckd"]}
```

---

#### `retrieve_and_rank(query: str, model_choice: str, top_k: int = 10) → dict`

The main function called by `app.py` on every query.

```python
# Input
query        = "obesity with type 2 diabetes"
model_choice = "llama3.2:1b"
top_k        = 10

# Internal steps (all wrapped in try/except):
structured = _safe_decompose(query, model_choice)
sub_queries = _engine.build_sub_queries(structured)

all_candidates = []
for sq in sub_queries:
    vector    = _engine.embed_query(sq)
    results   = collection.query(vector, n_results=15)
    all_candidates.extend(results)

deduplicated = _deduplicate(all_candidates)
reranked     = _engine.rerank_results(query, deduplicated, top_k=20)
hybrid       = _engine.hybrid_rerank(query, reranked, top_k=top_k)

# Output dict
{
    "query":             "obesity with type 2 diabetes",
    "primary_condition": "obesity",
    "comorbidities":     ["type 2 diabetes"],
    "sub_queries":       ["obesity", "type 2 diabetes", ...],
    "items": [
        {
            "rank":             1,
            "code":             "44054006",
            "term":             "Type 2 diabetes mellitus",
            "confidence_score": 0.912,
            "semantic_score":   4.7,
            "in_qof":           True,
            "usage_count":      350000,
            "sub_query_found":  "type 2 diabetes",
            "explanation":      ""   # filled in by add_llm_explanations
        },
        ...
    ],
    "error": ""  # populated if something failed
}
```

---

#### `add_llm_explanations(report: dict, query: str) → dict`

Takes the ranked report and asks the LLM to write an explanation for each code.

```python
# Input
report = {...}         # from retrieve_and_rank()
query  = "obesity with type 2 diabetes"

# What it does
payload = [{"code": item["code"], "term": item["term"], ...} for item in report["items"]]
system  = "You are a clinical coding assistant. For each code, write 1-2 sentences
           explaining why it is relevant to the query. Do NOT add new codes."
raw     = call_llm(system, json.dumps(payload))
parsed  = json.loads(raw)  # [{"code": "44054006", "explanation": "..."}, ...]

# Merges explanations back into report["items"]

# Output — same report dict, each item now has explanation string filled in
```

**Note:** The LLM here uses whatever system prompt is hardcoded in `pipeline.py`, NOT `prompts.md`. See the Prompts section below for why.

---

#### `run_ingestion() → tuple[bool, str]`

Runs `ingest_data.py` as a subprocess.

```python
# Input:  none
# Output: (True, "Ingestion complete") or (False, error_message)
# Usage:  normally run ingest_data.py from the terminal directly
```

---

## File 4: `llm.py` — LLM Communication Layer

**One job: send a message to an LLM and return the response.**

This file knows nothing about clinical codes, ChromaDB, or the UI. This separation means you can swap LLM providers by changing `.env` only — no code changes.

### Provider priority

```python
def _client() -> OpenAI:
    # Try each provider in order; first match wins

    # 1. Ollama (local, no internet)
    if base_url := os.getenv("OLLAMA_BASE_URL"):
        return OpenAI(api_key="ollama", base_url=base_url, timeout=120)

    # 2. OpenRouter (cloud gateway)
    if key := os.getenv("OPENROUTER_API_KEY"):
        return OpenAI(api_key=key, base_url="https://openrouter.ai", timeout=60)

    # 3. OpenAI direct
    if key := os.getenv("OPENAI_API_KEY"):
        return OpenAI(api_key=key, timeout=60)

    raise RuntimeError("No LLM provider configured")
```

**Why the same `OpenAI` SDK for all three providers?** Ollama, OpenRouter, and OpenAI all expose an OpenAI-compatible REST API (same HTTP endpoints, same JSON message format). The `OpenAI` Python SDK just needs a `base_url` to point at any of them. No provider-specific code needed.

### Functions

#### `_client() → OpenAI`

```python
# Input:  none (reads environment variables)
# Output: a configured OpenAI SDK client object
# Side effects: reads OLLAMA_BASE_URL, OPENROUTER_API_KEY, OPENAI_API_KEY from os.environ
```

---

#### `call_llm(system: str, user: str) → str`

```python
# Input
system = "You are a clinical coding assistant. Return only valid JSON."
user   = '[{"code": "44054006", "term": "Type 2 diabetes mellitus", ...}]'

# What it does
model       = os.getenv("LLM_MODEL")          # read at CALL time, not import time
temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
kwargs = {
    "model": model,
    "messages": [{"role": "system", "content": system},
                 {"role": "user",   "content": user}],
    "temperature": temperature,
}
# For cloud models: adds response_format={"type": "json_object"}
# For Ollama (small models): omits response_format, relies on prompt instruction

resp = client.chat.completions.create(**kwargs)

# Output
'[{"code": "44054006", "explanation": "Primary T2DM code. QOF-mandated."}, ...]'
# Raw string — parsing is the caller's responsibility
```

**Why read `LLM_MODEL` at call time?** `app.py` writes the dropdown selection to `os.environ["LLM_MODEL"]` before calling the pipeline. If `call_llm` read the env var once at import time, it would be stuck on whatever model was configured when the app started. Reading it at call time means every dropdown change takes effect immediately.

---

## File 5: `app.py` — The UI Orchestrator

**The only file you run. Contains the Gradio UI and wires all modules together.**

### Module-level setup (runs once at startup)

```python
load_dotenv(...)                                      # reads .env
import pipeline, ragas_eval, reasoning_eval           # imports all modules
import feedback_hitl, app_audit
_audit = app_audit.AuditLogger(output_dir="outputs/run_logs")  # singleton
```

**Why a singleton AuditLogger?** If you created a new `AuditLogger()` inside `process_query()`, each call would produce a new in-memory store and `finish_run()` would never find the record created by `start_run()`. One shared instance means records persist correctly across the start/finish calls.

### Section A — Model Selection

#### `_get_ollama_models() → list[str]`

```python
# Input:  none
# What:   calls GET {OLLAMA_BASE_URL}/api/tags with 3-second timeout
# Output: ["llama3.2:1b", "phi4:mini", ...] — installed models
#         [] if Ollama not running or URL not set
```

#### `_get_available_models() → list[str]`

```python
# Input:  none
# Output: combined list of all configured providers' models
#         e.g. ["llama3.2:1b", "phi4:mini"]  (if only Ollama configured)
#         ["llama3.2:1b", "gpt-4o-mini", "openai/gpt-4o-mini"]  (all configured)
#         ["no-provider-configured"]  (nothing configured)
```

#### `_get_default_model() → str`

```python
# Input:  none (reads LLM_MODEL from .env)
# Output: LLM_MODEL from .env if it's in the available list,
#         otherwise first available model
```

#### `_apply_model_choice(model_choice: str) → None`

```python
# Input:  "llama3.2:1b"
# Output: none
# Side effect: os.environ["LLM_MODEL"] = "llama3.2:1b"
# This is what makes the dropdown control the LLM used by llm.call_llm()
```

### Section B — Response Formatting

#### `_format_codes_response(report: dict, model_choice: str) → str`

Converts the pipeline report dict into the Markdown shown in the chat bubble.

```python
# Input
report = {
    "primary_condition": "obesity",
    "comorbidities":     ["type 2 diabetes"],
    "sub_queries":       ["obesity", "type 2 diabetes"],
    "items": [
        {
            "rank": 1, "code": "44054006",
            "term": "Type 2 diabetes mellitus",
            "confidence_score": 0.912,
            "in_qof": True, "usage_count": 350000,
            "explanation": "Primary QOF-mandated code for T2DM."
        }
    ]
}
model_choice = "llama3.2:1b"

# Output Markdown string (simplified)
"""
Model: `llama3.2:1b` · Pipeline: Hybrid (Semantic 70% + NHS Usage 20% + QOF 10%)
Decomposed into: Primary: obesity · Comorbidities: type 2 diabetes
Sub-queries: `obesity`, `type 2 diabetes`

📋 Code Recommendations — 1 found
---
**#1  Type 2 diabetes mellitus** · 🏥 QOF
`44054006` · Score: 91% · NHS usage: 350,000
*Primary QOF-mandated code for T2DM.*
---
"""
```

**Formatting decision: why no `###` headings in the chat bubble?**

Gradio's Markdown renderer converts `###` to `<h3>` tags. Browsers apply their own UA stylesheet to `<h3>`, which typically renders it at 1.17–1.25× the base font size. Even with `font-size:1rem !important` on the container, the `<h3>` would inherit a larger size from the UA stylesheet unless explicitly overridden. Similarly, `**bold**` produces `<strong>`, which some browsers render slightly larger. The fix is a CSS rule that locks every child element to `1rem`:

```css
.chatbot .message.bot > div *,
.chatbot .message.bot > div strong,
.chatbot .message.bot > div h1,
.chatbot .message.bot > div h2,
.chatbot .message.bot > div h3 {
    font-size: 1rem !important;
    font-family: Arial, sans-serif !important;
}
```

With this rule in place, `**bold**` renders at 1rem but with `font-weight:700`, providing visual hierarchy through weight alone, not size. Every line in the bubble is visually consistent.

---

#### `_format_audit_panel(run_id: str) → str`

```python
# Input:  run_id string (e.g. "run_20260410_143022_obesity")
# Reads:  _audit._active[run_id]  — the RunRecord in memory
# Output: Markdown string for the Audit Trail accordion

# Example output
"""
**Run ID:** `run_20260410_143022_obesity`
**Model:** `llama3.2:1b` · **Ranker:** `ChromaDB Hybrid`
**Codes logged:** 10 · **Completed:** `2026-04-10T14:30:25`

✅ **No validation issues detected.**

**Code Provenance (top 5):**

| Rank | Code | Term | Score | Source |
|------|------|------|-------|--------|
| #1 | `44054006` | Type 2 diabetes mellitus | 91% | csv_match |
...
"""
```

### Section C — Main Chat Handler

#### `process_query(message, history, model_choice) → tuple`

The central function. Called by Gradio on every query submission.

```python
# Inputs
message      = "obesity with type 2 diabetes"   # user's text
history      = [{"role": "user", ...}, ...]      # previous messages
model_choice = "llama3.2:1b"                     # dropdown value

# Returns — 5 values (Gradio maps these to 5 UI components)
(
    new_history,     # list — updated chat history → gr.Chatbot
    eval_panel_md,   # str  — RAGAS metrics → 📊 accordion
    reasoning_md,    # str  — reasoning trace → 🔍 accordion
    audit_md,        # str  — audit trail → 🗂️ accordion
    run_id,          # str  — stored in gr.State for feedback buttons
)
```

**Every early-return path** (empty message, greetings, engine not ready, retrieval error) also returns 5 values. If you add another output to Gradio, every return statement must be updated to include it.

---

#### `handle_feedback(vote, run_id, history, model_choice, note) → str`

Called by the 👍 or 👎 buttons.

```python
# Inputs
vote         = "thumbs_up"               # or "thumbs_down"
run_id       = "run_20260410_143022_..."  # from gr.State
history      = [...]                     # chat history (to extract codes)
model_choice = "llama3.2:1b"
note         = "Missing hypertension code"  # optional free text

# Extracts SNOMED codes from last bot message using regex:
found_codes = re.findall(r"`(\d{6,18})`", last_assistant_message)
# This works because _format_codes_response wraps every code in backticks

# Calls feedback_hitl.record_feedback(...)

# Output — confirmation string shown next to the buttons
"👍 Feedback recorded. Thank you."
```

**`gr.State`:** An invisible Gradio component. It stores one Python value between two different event calls (process_query and handle_feedback). Without it, the feedback buttons would have no way to know which run they are rating.

---

## File 6: `ragas_eval.py` — Quality Metrics

**Measures three dimensions of result quality after every query.**

RAGAS (Retrieval Augmented Generation Assessment) defines metrics for evaluating RAG pipelines. This file implements them locally — no external services, no API key required — because NHS clinical data should not leave the machine.

### The three metrics

| Metric | Question it answers | How computed |
|---|---|---|
| **Faithfulness** | Are explanations grounded in evidence? | LLM judge compares explanation to code metadata; heuristic fallback checks that term words appear in explanation |
| **Answer Relevancy** | Are the codes relevant to the query? | Uses CrossEncoder `rerank_score` values already computed during retrieval, rescaled to 0–1 |
| **Context Recall** | Did we find codes for all conditions? | Counts how many sub-queries produced at least one result; adjusted by top confidence score |

### Functions

#### `_score_faithfulness(report, query) → float`

```python
# Input
report = {items: [{code, term, explanation, in_qof, usage_count}, ...]}
query  = "obesity with type 2 diabetes"

# Path 1 — LLM judge (preferred)
raw = call_llm(
    system="Judge whether each explanation is faithful to the evidence...",
    user=f"Entries:\n{evidence_text}"
)
parsed = json.loads(raw)
return float(parsed["mean"])   # 0.0–1.0

# Path 2 — heuristic fallback (if LLM fails)
for item in items:
    term_words = item["term"].lower().split()
    term_hit   = any(w in item["explanation"].lower() for w in term_words if len(w) > 3)
    qof_ok     = not (item["in_qof"] and "qof" not in item["explanation"].lower())
    score      = 1.0 if (term_hit and qof_ok) else 0.5

# Output — float 0.0–1.0
```

#### `_score_answer_relevancy(report, query) → float`

```python
# Input — same as above

# Uses rerank_scores already in report items
# CrossEncoder scores range ~-5 to +10, rescaled to 0–1 via: max(0, min(1, (s+5)/10))

# Output — float 0.0–1.0
```

#### `_score_context_recall(report, query) → float`

```python
# Input — same as above

# For each sub-query, checks whether any item has sub_query_found matching it
# covered_ratio = (sub-queries with ≥1 result) / total sub-queries
# Adjusted: covered_ratio × (0.5 + 0.5 × top_confidence_score)

# Output — float 0.0–1.0
```

#### `evaluate(report, query) → dict`

```python
# Input
report = {...}    # from pipeline with explanations added
query  = "obesity with type 2 diabetes"

# Output
{
    "faithfulness":     0.87,
    "answer_relevancy": 0.91,
    "context_recall":   0.75,
    "overall":          0.84,
    "interpretation":   "**Overall quality: High.** The results appear well-grounded.",
    "codes_evaluated":  10,
    "eval_method":      "llm_judge+heuristic",
    "eval_time_s":      2.3,
    "timestamp":        "2026-04-10T14:30:27+00:00"
}
```

#### `format_eval_panel(metrics) → str`

```python
# Input:  metrics dict from evaluate()
# Output: Markdown table for the 📊 accordion
# Colour coding: 🟢 ≥80%, 🟡 ≥60%, 🔴 <60%
```

---

## File 7: `reasoning_eval.py` — Step-by-Step Reasoning Trace

**Explains HOW the pipeline reached each recommendation.**

This module produces a structured, step-by-step account of the full pipeline run. It reads the report dict (which carries all the intermediate data) and narrates what happened at each stage.

### Functions

#### `generate_reasoning_trace(report, query) → str`

```python
# Input
report = {
    "primary_condition": "obesity",
    "comorbidities":     ["type 2 diabetes"],
    "sub_queries":       ["obesity", "type 2 diabetes"],
    "items": [
        {"rank": 1, "code": "44054006", "term": "...",
         "semantic_score": 4.7, "confidence_score": 0.912,
         "in_qof": True, "usage_count": 350000,
         "sub_query_found": "type 2 diabetes"}
    ]
}
query = "obesity with type 2 diabetes"

# Output — multi-section Markdown string (displayed in 🔍 accordion):
"""
## 🔍 Reasoning Trace
*Generated 2026-04-10 14:30 UTC for query: "obesity with type 2 diabetes"*

### Step 1 — Query Decomposition
...

### Step 2 — Semantic Search (ChromaDB)
...

### Step 3 — Cross-Encoder Reranking
Highest CrossEncoder score: **4.700** — Type 2 diabetes mellitus (44054006)

### Step 4 — Hybrid Scoring
| Rank | Code | Term | Semantic | Usage | QOF | Hybrid |
...

### Step 5 — LLM Explanation
...

### Summary
- 10 codes returned
- 3 are QOF-mandated
- 3 sub-queries searched
- Top hybrid score: 91% (Type 2 diabetes mellitus)
"""
```

No LLM calls are made here. The trace is generated purely from data already in the report dict.

---

#### `generate_score_breakdown(item) → str`

```python
# Input — one item dict from report["items"]
item = {
    "code": "44054006", "term": "Type 2 diabetes mellitus",
    "semantic_score": 4.7, "confidence_score": 0.912,
    "in_qof": True, "usage_count": 350000,
    "sub_query_found": "type 2 diabetes"
}

# Output — short Markdown breakdown for one code
"""
**`44054006` — Type 2 diabetes mellitus**

- Found by sub-query: `type 2 diabetes`
- CrossEncoder semantic score: `4.700`
- NHS annual usage: 350,000
- QOF mandated: Yes (10% bonus applied)
- **Final hybrid score: 0.912** (91%)
"""
```

---

## File 8: `feedback_hitl.py` — Human-in-the-Loop Feedback

**Records analyst thumbs up/down votes for quality monitoring and future training.**

### Data structures

#### `FeedbackRecord` (dataclass)

| Field | Type | Example | Description |
|---|---|---|---|
| `run_id` | str | `"run_20260410_..."` | Links to audit log |
| `query` | str | `"obesity with T2DM"` | Original query |
| `vote` | str | `"thumbs_up"` | `"thumbs_up"` or `"thumbs_down"` |
| `recommended_codes` | list | `["44054006"]` | Codes that were shown |
| `model_name` | str | `"llama3.2:1b"` | Model used |
| `note` | str | `"Missing code X"` | Analyst comment |
| `n_codes` | int | `10` | Count of recommended codes |
| `n_qof_codes` | int | `3` | QOF codes in results |
| `timestamp` | str | ISO datetime | When feedback was recorded |

### Functions

#### `record_feedback(run_id, query, vote, recommended_codes, model_name, note, n_qof_codes) → dict`

```python
# Input
run_id            = "run_20260410_143022_obesity"
query             = "obesity with type 2 diabetes"
vote              = "thumbs_up"   # or "thumbs_down"
recommended_codes = ["44054006", "414916001"]
model_name        = "llama3.2:1b"
note              = "Both codes correct"
n_qof_codes       = 2

# Output — the saved FeedbackRecord as a dict
{
    "run_id": "run_20260410_143022_obesity",
    "query":  "obesity with type 2 diabetes",
    "vote":   "thumbs_up",
    ...
}

# Side effects:
# 1. Writes outputs/feedback/run_20260410_143022_obesity_th.json
# 2. Appends one row to outputs/feedback/feedback_summary.csv
```

#### `load_feedback_summary() → dict`

```python
# Input:  none
# Reads:  outputs/feedback/feedback_summary.csv
# Output
{
    "total":         47,
    "thumbs_up":     38,
    "thumbs_down":   9,
    "positive_rate": 0.809,
    "recent":        [...last 10 rows...],
    "by_model": {
        "llama3.2:1b": {"total": 30, "thumbs_up": 24, "positive_rate": 0.8},
        "gpt-4o-mini": {"total": 17, "thumbs_up": 14, "positive_rate": 0.824}
    }
}
```

#### `format_feedback_summary_panel() → str`

```python
# Input:  none
# Output: Markdown table for the 👍👎 accordion
# Called by: app.py on startup AND after every feedback button click
```

---

## File 9: `app_audit.py` — Audit Trail and Backtesting

**Records every run for governance review, comparison, and backtesting.**

### Data structures

#### `CodeProvenance` (dataclass)

| Field | Type | Example | Description |
|---|---|---|---|
| `snomed_code` | str | `"44054006"` | The code |
| `term` | str | `"Type 2 diabetes mellitus"` | Human name |
| `confidence_score` | float | `0.912` | Hybrid score |
| `rank` | int | `1` | Position in output |
| `ranked_by` | str | `"ChromaDB Hybrid"` | Which ranker |
| `match_score` | float | `0.8` | Initial embedding distance |
| `explanation` | str | `"Primary T2DM code"` | LLM explanation |
| `source_type` | str | `"csv_match"` | `"csv_match"` or `"llm_only"` |
| `is_hallucinated` | bool | `False` | True if code not from CSV |

#### `RunRecord` (dataclass)

| Field | Type | Example | Description |
|---|---|---|---|
| `run_id` | str | `"run_20260410_..."` | Unique identifier |
| `query` | str | `"obesity with T2DM"` | User's query |
| `model_name` | str | `"llama3.2:1b"` | LLM used |
| `ranking_model` | str | `"ChromaDB Hybrid"` | Ranker used |
| `started_at` | str | ISO datetime | Run start time |
| `completed_at` | str | ISO datetime | Run end time |
| `codes` | list | `[CodeProvenance, ...]` | All recommended codes |
| `validation_flags` | list | `[{severity, type, msg}]` | Issues found |
| `query_hash` | str | `"a3f2b1c9"` | MD5 of lowercased query |

### Class: `AuditLogger`

#### `start_run(query, model_name, ranking_model) → str`

```python
# Input
query         = "obesity with type 2 diabetes"
model_name    = "llama3.2:1b"
ranking_model = "ChromaDB Hybrid"

# Creates a RunRecord in self._active dict
# Output — run_id string
"run_20260410_143022_obesitywithtyp"
```

#### `finish_run(run_id, report_items) → None`

```python
# Input
run_id       = "run_20260410_143022_..."
report_items = [{"code": "44054006", "term": "...", "confidence_score": 0.912, ...}]

# What it does:
# 1. Creates a CodeProvenance for each item
# 2. Runs _validate() — three automatic checks:
#    CHECK 1 — MISSING_EXPLANATION: explanation == ""
#    CHECK 2 — POSSIBLE_HALLUCINATION: code == "UNKNOWN" or source == "llm_only"
#    CHECK 3 — CONFIDENCE_MISMATCH: confidence ≥ 0.9 but match_score < 0.5

# Output: none (modifies RunRecord in self._active)
```

#### `save(run_id) → Path | None`

```python
# Input:  run_id string
# Output: Path to saved file, or None if failed
# Writes: outputs/run_logs/{run_id}.json
```

### Functions

#### `score_against_gold_standard(recommended_codes, gold_standard_codes, usage_counts) → dict`

```python
# Input
recommended_codes    = ["44054006", "414916001", "99999999"]
gold_standard_codes  = {"44054006", "414916001", "73211009"}
usage_counts         = {"44054006": 350000, "414916001": 80000, "73211009": 45000}

# Output
{
    "recall":                  0.6667,   # 2/3 gold codes found
    "precision":               0.6667,   # 2/3 recommendations were correct
    "f1":                      0.6667,
    "patient_weighted_recall": 0.8977,   # 73211009 (45k patients) was missed; 2 found cover 430k
    "true_positives":          ["44054006", "414916001"],
    "false_negatives":         ["73211009"],   # missed
    "false_positives":         ["99999999"],   # wrong recommendation
    "n_recommended":           3,
    "n_gold_standard":         3
}
```

#### `compare_runs(run_a_path, run_b_path) → dict`

```python
# Input:  two file paths to saved run JSON files
# Output: dict describing what changed
{
    "run_a_id":       "run_20260410_143022_...",
    "run_b_id":       "run_20260410_150344_...",
    "query":          "obesity with type 2 diabetes",
    "config_changes": {"model_name": {"run_a": "llama3.2:1b", "run_b": "gpt-4o-mini"}},
    "codes_added":    [{"code": "...", "term": "...", "now_rank": 3}],
    "codes_removed":  [{"code": "...", "term": "...", "was_rank": 7}],
    "rank_changed":   [{"code": "...", "rank_a": 2, "rank_b": 1}],
    "unchanged_count": 7
}
```

#### `run_backtest(chat_fn, gold_standard_dir, model_name, ranking_model, output_dir) → dict`

```python
# Input
chat_fn           = process_query  # from app.py
gold_standard_dir = "data/gold_standard/"
model_name        = "llama3.2:1b"
ranking_model     = "ChromaDB Hybrid"

# What it does:
# For each DAAR_2025_004_*.txt file in gold_standard_dir:
#   1. Extracts condition name from filename
#   2. Calls chat_fn(condition_name) to get recommendations
#   3. Extracts SNOMED codes from the response Markdown
#   4. Calls score_against_gold_standard()
#   5. Prints recall/precision/F1 per condition

# Output
{
    "hypertension": {"recall": 0.82, "precision": 0.70, "f1": 0.755, ...},
    "obesity":      {"recall": 0.91, "precision": 0.75, "f1": 0.823, ...},
    "_summary":     {
        "conditions_tested": 12,
        "mean_recall":       0.84,
        "mean_precision":    0.73,
        "mean_f1":           0.781,
        "pass_count":        10    # conditions where recall >= 0.6
    }
}
# Also saves outputs/backtest/backtest_llama3.2_1b_ChromaDBHybrid.json
```

---

## File 10: `service.py` — CLI Query Service

**Only used by `demo.py`. Not connected to `app.py`.**

```python
def generate_report(payload: dict) -> dict:
    # Input:  payload dict (see demo.py's SAMPLE_PAYLOAD)
    # Reads:  prompts.md — the system prompt for governance-aware LLM calls
    # Calls:  llm.call_llm(system_prompt, json.dumps(payload))
    # Output: parsed JSON dict from the LLM response
```

---

## File 11: `prompts.md` — System Prompt for Service Layer

**Read by `service.py` only. Not used by `app.py`'s pipeline.**

This file contains governance-critical instructions for the LLM when classifying code candidates. Key rules:

```
Flag rules — flag must be exactly one of:
  CANDIDATE_INCLUDE | REVIEW | STRATIFIER | UNCLASSIFIED

- Do NOT hallucinate codes, terms, or clinical facts.
- Return only valid JSON. No prose outside the JSON.
- Only recommend codes that appear in the payload as evidence.
```

**Why isn't this used by the live app?** `app.py` calls `pipeline.add_llm_explanations()` which has its own internal system prompt hardcoded in `pipeline.py`. To bring `prompts.md`'s governance rules into the live app, add this to `pipeline.py`:

```python
# Recommended addition to pipeline.py
_SYSTEM_PROMPT = (Path(__file__).parent / "prompts.md").read_text(encoding="utf-8")

def add_llm_explanations(report: dict, query: str) -> dict:
    ...
    raw = call_llm(system=_SYSTEM_PROMPT, user=json.dumps(payload))
```

---

## File 12: `demo.py` — CLI Test Script

**Standalone test — not connected to `app.py`.**

```python
# Runs a hardcoded test payload through service.py and prints the JSON result
# Useful for: testing LLM connectivity, checking service.py output format
# Run with:   python demo.py
```

---

## The Formatting Fix — Full Technical Explanation

### What was wrong (from the screenshot)

The chat bubble showed code titles like "#1 Renal disorder due to type 2 diabetes mellitus" in very large, heavy text — noticeably larger than the code IDs and explanations beneath them.

### Why it happened

Gradio converts `**text**` to `<strong>text</strong>` in HTML. Browsers ship a built-in (UA) stylesheet that styles HTML elements. For `<strong>`, most browsers apply:

```css
/* Browser UA stylesheet (you cannot override this directly in most cases) */
strong { font-weight: bold; }
```

That's fine. But Gradio also processes `**text**` inside `<p>` tags, and some Gradio versions add extra heading or paragraph margins that effectively push content upward, making bold text appear larger. Additionally, if any `###` heading was present (as in an earlier version), the browser would render it at 1.17× base size.

### Why `font-size: 1rem` on the container is not enough

Setting `font-size: 1rem` on `.chatbot .message.bot > div` sets the base font size for that container. But child elements like `<strong>` and `<h3>` can override that via the UA stylesheet's **relative** sizes or their own explicit size rules. The container rule does not cascade down unless we explicitly say so.

### The fix — CSS specificity cascade

```css
/* This rule in app.py's _CSS string */
.chatbot .message.bot > div *,
.chatbot .message.bot > div p,
.chatbot .message.bot > div strong,
.chatbot .message.bot > div b,
.chatbot .message.bot > div h1,
.chatbot .message.bot > div h2,
.chatbot .message.bot > div h3,
.chatbot .message.bot > div h4,
.chatbot .message.bot > div li {
    font-size: 1rem !important;
    font-family: Arial, sans-serif !important;
}
```

The `!important` flag overrides the UA stylesheet. The explicit listing of each element type ensures nothing is missed. The result: every element inside the bot bubble renders at exactly 1rem, in Arial. Bold text (`font-weight:700`) is the only visual differentiator between titles and body text — weight, not size.

---

## Changes Made — Session-by-Session Summary

### Session 1 Changes

| What changed | File | Reason |
|---|---|---|
| Imported `app_audit` | `app.py` | Was never imported — audit features were inaccessible |
| Added `_audit` singleton | `app.py` | Needed for start_run/finish_run to share state |
| Replaced manual `run_id` with `_audit.start_run()` | `app.py` | run_id was floating; now tied to an actual audit record |
| Added `_audit.finish_run()` + `_audit.save()` | `app.py` | Completes and persists the audit record |
| Added `_format_audit_panel()` function | `app.py` | Formats audit data for UI display |
| Added Audit Trail accordion to UI | `app.py` | Validation flags now visible to analysts |
| Updated `_outputs` list from 4 to 5 | `app.py` | New accordion needs a matching output slot |
| Updated `process_query` to return 5 values | `app.py` | Must match `_outputs` count exactly |
| Changed `### heading` to `**bold**` in `_format_codes_response` | `app.py` | `###` was rendering as large H3 heading |

### Session 2 Changes

| What changed | File | Reason |
|---|---|---|
| Added CSS rule locking ALL child elements to `1rem` | `app.py` (`_CSS`) | Bold text in chat bubble was rendering larger than normal text due to browser UA stylesheet |
| Reformatted `_format_codes_response` cards | `app.py` | Removed blockquote `>` prefixes and `###` headers; plain text + `**bold**` now consistent |
| Added explicit `font-family:Arial` to bot bubble CSS | `app.py` | Ensure monospace code elements don't affect surrounding text size |
| Added `hr` styling, `em` colour, `code` border styling | `app.py` | Visual polish to match NHS design language |

---

## Required Files Summary

```
project_root/
├── app.py                        ← run this
├── pipeline.py                   ← imported by app.py
├── pod1_pod2_integrated_V2.py    ← imported by pipeline.py
├── llm.py                        ← imported by pipeline.py + ragas_eval.py
├── ragas_eval.py                 ← imported by app.py
├── reasoning_eval.py             ← imported by app.py
├── feedback_hitl.py              ← imported by app.py
├── app_audit.py                  ← imported by app.py
├── ingest_data.py                ← run once
├── .env                          ← OLLAMA_BASE_URL, LLM_MODEL etc.
├── nhs-logo-880x4951.jpeg        ← optional (fallback text badge if missing)
├── prompts.md                    ← used by service.py / demo.py only
├── service.py                    ← used by demo.py only
├── demo.py                       ← optional CLI test
└── data/
    ├── chroma_db/                ← ChromaDB (built by ingest_data.py)
    └── processed_data/
        └── snomed/
            └── snomed_master_v3.csv
```

---

## Backtesting Guide

```python
from app import process_query
from app_audit import run_backtest

results = run_backtest(
    chat_fn=process_query,
    gold_standard_dir="data/gold_standard/",
    model_name="llama3.2:1b",
    ranking_model="ChromaDB Hybrid",
    output_dir="outputs/backtest/",
)

# Results printed to terminal + saved to:
# outputs/backtest/backtest_llama3.2_1b_ChromaDBHybrid.json
```

**Gold-standard files** must be named `DAAR_2025_004_{condition}_codes.txt` and be tab-delimited with a `code` column.

**What to compare:**
- Two models for explanation quality: which gives higher faithfulness scores?
- Two sets of hybrid weights (70/20/10 vs 80/10/10): which retrieves more gold codes?
- Two query decomposition approaches: phi4:mini vs llama3.2 for splitting conditions

---

## Models Reference

| Model | File that uses it | Purpose | Download |
|---|---|---|---|
| `BAAI/bge-small-en` | `ingest_data.py`, `pod1_pod2` | Embed text to 384-float vectors | Auto via sentence-transformers |
| `BAAI/bge-reranker-base` | `pod1_pod2.rerank_results()` | Score (query, document) pairs together | Auto via sentence-transformers |
| `phi4:mini` | `pod1_pod2.query_decompose()` | Decompose query into conditions | `ollama pull phi4:mini` |
| Any dropdown model | `llm.call_llm()` | Write explanations + evaluate faithfulness | `ollama pull llama3.2:1b` etc. |
