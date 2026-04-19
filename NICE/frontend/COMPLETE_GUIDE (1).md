# NICE Clinical Code Assistant — Complete Technical Reference

> This document explains every file, every function, every variable, every input and output, and how everything connects. It assumes no prior knowledge of the codebase.

---

## First: How to Run the App

```bash
# Step 1 — Install dependencies (once)
pip install gradio sentence-transformers chromadb langchain-ollama langchain-core python-dotenv openai

# Step 2 — Build the database (once, or when SNOMED data changes)
python ingest_data.py

# Step 3 — Start Ollama and pull the required models (once per machine)
ollama serve
ollama pull phi4:mini      # used for query decomposition
ollama pull llama3.2:1b    # used for explanations (or any model you prefer)

# Step 4 — Configure .env (once)
# Edit the .env file and set:
# OLLAMA_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=llama3.2:1b

# Step 5 — Run the app (every time you want to use it)
python app.py
# Open http://localhost:7860
```

After Step 2, you never need to run `ingest_data.py` again unless the source SNOMED CSV changes. The database is stored on disk and `app.py` reads from it every time without rebuilding.

---

## The Problem That Was Fixed

The previous version of `pipeline.py` showed "Engine not ready — run ingest_data.py" even when the database was perfectly fine. This was caused by `pod1_pod2_integrated_V2.py` initialising `OllamaLLM(model="phi4-mini")` at module level. When Python imports that file, it runs all module-level code immediately, including the Ollama connection. If Ollama was not yet running at that moment, the import would fail. `pipeline.py` would set `_engine = None` and the error message incorrectly told the user to rebuild the database.

The fix: `pipeline.py` now does a **separate, direct ChromaDB health check** independently of the engine import. This means it can tell the difference between:
- Database is missing → run ingest_data.py
- Database exists but Ollama is offline → start Ollama
- Database exists and Ollama is running → ready to use

---

## All Files in the Final App

| File | Run by you? | Runs automatically? | Purpose |
|---|---|---|---|
| `ingest_data.py` | Yes, once | Never | Builds ChromaDB from SNOMED CSV |
| `pod1_pod2_integrated_V2.py` | Never directly | On app start | ML models + ChromaDB search + hybrid ranking |
| `pipeline.py` | Never directly | On app start | Clean API over the engine |
| `llm.py` | Never directly | On each query | Sends messages to the LLM |
| `app.py` | Yes, every time | — | Gradio UI + event wiring |
| `ragas_eval.py` | Never directly | On each query | Quality metrics |
| `reasoning_eval.py` | Never directly | On each query | Step-by-step reasoning trace |
| `feedback_hitl.py` | Never directly | On thumbs click | Saves analyst feedback |
| `app_audit.py` | Optionally | Optionally | Audit trail + backtesting |
| `.env` | You edit it | Read on start | API keys + model config |

---

## File 1: `ingest_data.py` — Database Builder

**Why it exists:** The ChromaDB vector database must be built before the app can search anything. This file reads the master SNOMED CSV, converts every code description into a vector (a list of 384 numbers that represent its clinical meaning), and saves everything to disk.

**When to run:** Once before first use. Again only if `snomed_master_v3.csv` changes.

**What it does, step by step:**

1. `pd.read_csv(INPUT_CSV)` — reads the master SNOMED CSV into a pandas DataFrame. Each row is one clinical code with its term (description), semantic tag, QOF status, and NHS usage count.

2. `df.dropna(subset=['snomed_code', 'term'])` — removes rows with missing code or term. Empty rows would crash the embedding model.

3. `df['text_for_ai'] = df['term'] + " (" + df['semantic_tag'] + ")"` — creates the text that gets embedded. Combining term and semantic tag (e.g. "Type 2 diabetes mellitus (finding)") gives the embedding model more clinical context than the term alone.

4. `SentenceTransformer("BAAI/bge-small-en").encode(...)` — converts every text description into a 384-float vector. This is the embedding model. The resulting vectors capture clinical meaning: "obesity" and "BMI 30+" will have similar vectors because they mean the same thing clinically, even though the words differ.

5. `chromadb.PersistentClient(path=DB_DIR)` — opens a ChromaDB client that writes to disk. The database is stored in `data/chroma_db/`.

6. `collection.add(ids=..., embeddings=..., documents=..., metadatas=...)` — saves codes in batches of 5,000. Each record stored:
   - `id`: the SNOMED code string
   - `embedding`: the 384-float vector
   - `document`: the text that was embedded (term + semantic tag)
   - `metadata`: term, in_qof (True/False), usage_count_nhs (number)

**Key variables:**
- `BASE_DIR` — root of the data folder. Change this to match your machine.
- `INPUT_CSV` — path to the processed SNOMED master CSV.
- `DB_DIR` — where ChromaDB is stored. Must match the path in `pod1_pod2_integrated_V2.py`.
- `batch_size = 5000` — how many codes to add per ChromaDB call. Larger values are faster but use more RAM.

**Input:** `data/processed_data/snomed/snomed_master_v3.csv`
**Output:** `data/chroma_db/` — a persistent ChromaDB collection named `snomed_master_v3_retrieval`

---

## File 2: `pod1_pod2_integrated_V2.py` — ML Search and Ranking Engine

**Why it exists:** This is the technical core produced by Pods 1 and 2. It owns all the ML model logic: the embedding model, the reranker, the query decomposition LLM, and the hybrid scoring formula. It is kept as a separate file so that the team can update the ranking logic without touching the UI.

**What happens when Python imports this file:**

All of the following runs immediately at import time (before any function is called):

```python
chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)    # connects to DB on disk
collection = client.get_or_create_collection(...)      # gets the collection
embedding_model = SentenceTransformer("BAAI/bge-small-en")  # loads 120MB model
reranker = CrossEncoder("BAAI/bge-reranker-base")      # loads reranker model
llm = OllamaLLM(model="phi4-mini")                     # connects to Ollama
decomp_chain = decomp_prompt | llm                     # builds the LangChain chain
```

The Ollama connection is the reason this import can fail if Ollama is not running.

**Path resolution:** The file searches four possible locations for `chroma_db/` in order, using the first one whose parent directory exists. The last option is an absolute fallback path. If none are found, it uses the absolute path. This is why the app works on different machines even if folder structures differ.

**Functions:**

`query_decompose(query: str) → dict`
- Input: "obesity with type 2 diabetes and hypertension"
- What it does: sends the query to `phi4-mini` via Ollama with a prompt asking it to extract clinical concepts
- Output: `{"primary_condition": "obesity", "comorbidities": ["type 2 diabetes", "hypertension"]}`
- Why this matters: searching each condition separately retrieves more relevant codes than searching the full string, because the vector for "obesity with type 2 diabetes" is an average of both concepts rather than being specific to either.

`build_sub_queries(structured_query: dict) → list[str]`
- Input: the dict from query_decompose
- Output: list of search strings — one per condition plus one combined string
- Example: `["obesity", "type 2 diabetes", "hypertension", "obesity with type 2 diabetes, hypertension"]`

`embed_query(text: str) → list[float]`
- Input: any text string
- Output: a list of 384 floats representing the text's clinical meaning
- This is called for each sub-query before searching ChromaDB

`chroma_multi_query_search(user_query: str, n_results: int) → dict`
- Input: original query, how many results per sub-query
- Calls `query_decompose`, `build_sub_queries`, `embed_query` and `collection.query` for each sub-query
- Deduplicates results (if same code found by multiple sub-queries, keeps the closest match)
- Calls `rerank_results` to add CrossEncoder scores
- Output: `{"original_query": str, "structured_query": dict, "sub_queries": list, "retrieved_context": list}`

`rerank_results(query: str, retrieved_docs: list, top_k: int) → list`
- Input: original query and the merged Chroma results
- What it does: sends each (query, document) pair to the CrossEncoder model
- The CrossEncoder reads both texts together (not as independent vectors), giving a more accurate relevance score
- Output: same list, sorted by `rerank_score`, truncated to top_k

`hybrid_rerank(query, retrieved_docs, top_k, alpha, beta, gamma) → list`
- Input: CrossEncoder-scored docs and three weight parameters
- Computes: `hybrid_score = (alpha × rerank_score) + (beta × usage_normalised) + (gamma × qof_bonus)`
- Default weights: `alpha=0.7`, `beta=0.2`, `gamma=0.1`
- `usage_normalised` = usage_count_nhs divided by the highest usage count in this result set
- `qof_bonus` = gamma (0.1) if `in_qof == "True"`, else 0
- Output: list sorted by hybrid_score descending, truncated to top_k

---

## File 3: `pipeline.py` — Clean API Layer (Fixed in This Version)

**Why it exists:** Before this file existed, `app.py` imported `pod1_pod2_integrated_V2` directly. That meant the UI had a hard dependency on all the ML models, and any import failure would crash the app with a confusing error. `pipeline.py` is a buffer — it wraps the engine, handles failures gracefully, and gives the rest of the app clean functions to call.

**Why it was the problem file:** The previous version's `is_ready()` function only checked whether `_engine` was not None. If the engine import failed (for any reason, including Ollama being offline), `is_ready()` returned False with "run ingest_data.py". This was wrong — the database was fine.

**What the fix does:** `pipeline.py` now does a **direct ChromaDB connection** completely independently of the engine import. Even if the engine fails to load, the direct check can confirm the database exists and has codes. The `is_ready()` function now distinguishes between three failure modes and returns a different message for each.

**Module-level startup (runs when `app.py` imports `pipeline`):**

```python
# Attempt 1: full engine import (includes ML models + Ollama)
try:
    import pod1_pod2_integrated_V2 as _engine
except ImportError as e:
    _engine_error = f"MISSING_PACKAGE:{e}"
except Exception as e:
    _engine_error = f"OLLAMA_OFFLINE:{e}"   # Ollama offline → caught here

# Attempt 2: direct ChromaDB check (no Ollama needed)
_db_direct_count = _check_db_directly()
# This opens ChromaDB directly and counts codes
# Works even when _engine is None
```

**Functions:**

`is_ready() → (bool, str)`
- Returns `(True, "")` when both DB and engine are working
- Returns `(False, "DB empty — run ingest_data.py")` when DB has no codes
- Returns `(False, "Start Ollama — DB has N codes")` when DB is fine but Ollama is offline
- Returns `(False, "Missing packages — pip install...")` when packages are absent
- Called by `app.py` before every query and shown to the user if not ready

`db_code_count() → int`
- Returns the number of SNOMED codes in ChromaDB
- Used by `app.py` to show the green/amber/red status pill in the header

`_safe_decompose(query: str) → dict`
- Tries `_engine.query_decompose()` first
- If Ollama is unavailable at query time (not import time), falls back to splitting the query on "with" and "and"
- This means the app can still search even if query decomposition is degraded

`retrieve_and_rank(query: str, top_k: int) → dict`
- The main function called by `app.py` on every query
- Calls `_safe_decompose`, then `_engine.build_sub_queries`, then searches ChromaDB for each sub-query, then calls `_engine.rerank_results`, then calls `_engine.hybrid_rerank`
- All three ML steps (embed, rerank, hybrid) have try/except fallbacks so the function never crashes the UI

`add_llm_explanations(report: dict, query: str) → dict`
- Takes the ranked report and calls `llm.call_llm()` to generate explanations
- The LLM is told it cannot add codes or change rankings
- Falls back to generic "Relevant to query" text if LLM is unavailable

`run_ingestion() → (bool, str)`
- Runs `ingest_data.py` as a subprocess
- Used by the (optional) "Rebuild Database" button
- You would normally just run `python ingest_data.py` from the terminal instead

---

## File 4: `llm.py` — LLM Communication Layer

**Why it exists:** This file's only job is sending a message to an LLM and returning the response. It knows nothing about clinical codes, ChromaDB, or the UI. Separating it means you can swap the LLM provider (Ollama → OpenAI → OpenRouter) by changing only `.env` — no code change.

**`_client() → OpenAI`**
- Reads environment variables in order: `OLLAMA_BASE_URL`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`
- Returns a configured `OpenAI` SDK client for whichever provider is available
- Why the same SDK for all three: Ollama, OpenRouter, and OpenAI all speak the OpenAI API format (same HTTP endpoints, same message structure). The `OpenAI` Python SDK works for all three — you just point `base_url` at the right server.

**`call_llm(system: str, user: str) → str`**
- Input: `system` — standing instructions for the LLM, `user` — the specific message
- Output: the LLM's response as a raw string
- Reads `LLM_MODEL` from `os.environ` at call time (not import time). This is why changing the model dropdown in the UI works immediately — `app.py` writes the new model name to `os.environ["LLM_MODEL"]` before calling `call_llm`.
- `USE_JSON_RESPONSE_FORMAT=true` adds `response_format={"type":"json_object"}` which forces JSON output from cloud models. Set to `false` for Ollama (small local models often don't support this parameter).

---

## File 5: `app.py` — The UI Orchestrator

**Why it exists:** This is the only file you run (`python app.py`). It owns the Gradio browser interface and wires all the other modules together. It has no clinical logic of its own — it calls `pipeline`, `ragas_eval`, `reasoning_eval`, and `feedback_hitl` and assembles their outputs into the UI.

**Why `gr.Blocks` instead of `gr.ChatInterface`:**
`gr.ChatInterface` is Gradio's pre-built conversational widget. It is simpler to use but only supports one output value (the text response). The final app needs to update multiple components simultaneously when a query runs: the chat history, the eval metrics panel, the reasoning trace panel, and the stored run_id for feedback. `gr.Blocks` gives full control over layout and event wiring at the cost of slightly more code.

**Section A — Model Selection**

`_get_ollama_models() → list[str]`
- Calls `GET {OLLAMA_BASE_URL}/api/tags` with a 3-second timeout
- Returns the names of all models currently installed in Ollama
- Used to build the dropdown so only actually-installed models appear

`_get_available_models() → list[str]`
- Calls `_get_ollama_models()` for Ollama, uses hardcoded lists for cloud providers
- Returns a combined list in priority order (Ollama first)
- If nothing is configured, returns `["no-provider-configured"]`

`_get_default_model() → str`
- Returns `LLM_MODEL` from `.env` if it appears in the available list
- Otherwise returns the first available model

`_apply_model_choice(model_choice: str) → None`
- Writes the selected model name into `os.environ["LLM_MODEL"]`
- `llm.py` reads this at call time, so the change takes effect immediately on the next LLM call

**Section B — Response Formatting**

`_format_codes_response(report: dict, model_choice: str) → str`
- Input: the report dict from `pipeline.retrieve_and_rank()`, the selected model name
- Output: Markdown string displayed in the chat window
- Includes: provenance header (model, pipeline), query decomposition, sub-queries searched, one card per code with rank, hybrid score, QOF badge, NHS usage, and explanation

**Section C — Main Chat Handler**

`process_query(message, history, model_choice) → (history, eval_md, reasoning_md, run_id)`
- Called by Gradio when the user submits a query (Submit button or Enter key)
- Returns four values simultaneously:
  1. Updated chat history → displayed in the `gr.Chatbot` component
  2. Evaluation metrics Markdown → populates the "📊 Evaluation Metrics" Accordion
  3. Reasoning trace Markdown → populates the "🔍 Reasoning Trace" Accordion
  4. Run ID string → stored in `gr.State` for the feedback buttons to use

`handle_feedback(vote, run_id, history, model_choice, note) → str`
- Called by the 👍 or 👎 buttons
- Extracts SNOMED codes from the last chat message using regex: `r"`(\d{6,18})`"`
- Calls `feedback_hitl.record_feedback()`
- Returns a confirmation string shown next to the buttons

**`gr.State`:** An invisible Gradio component that stores a Python value between event calls. The `run_id` is generated by `process_query()` and must be passed to `handle_feedback()` when the analyst clicks a button. `gr.State` is the bridge between these two events without embedding the run_id in any visible UI element.

---

## File 6: `ragas_eval.py` — Quality Metrics

**Why it exists:** The analyst needs to know whether to trust the results they are seeing. This file computes three quality scores and displays them in the Evaluation Metrics panel. It runs after the main response is already shown (non-blocking) and uses no external services.

**Three metrics explained in plain English:**

**Faithfulness** — "Can I trust the explanations?" Checks whether the LLM's explanations are grounded in the evidence available for each code (its term name, QOF status, usage count). A faithfulness score below 60% means the LLM may be making clinical claims it cannot support from the data. Computed by asking the local LLM to judge each explanation against its evidence, with a heuristic fallback.

**Answer Relevancy** — "Are the codes relevant to what I asked?" Measures whether the returned codes actually match the clinical query. Uses the CrossEncoder `rerank_score` values already computed during retrieval (scaled to 0–1) as a proxy for relevance. A low score suggests the semantic search found codes in the wrong clinical domain.

**Context Recall** — "Did we find codes for all conditions mentioned?" Estimates whether all sub-queries produced results. Checks how many of the sub-queries that were searched produced at least one code in the final output. Adjusted by the top confidence score. A low score suggests the query decomposition missed a clinical concept.

**`evaluate(report, query) → dict`**
- Input: the report from `pipeline.retrieve_and_rank()` (with explanations added)
- Runs all three metrics
- Adds a plain-English `interpretation` string
- Returns: `{faithfulness, answer_relevancy, context_recall, overall, interpretation, codes_evaluated, timestamp}`

**`format_eval_panel(metrics) → str`**
- Formats the metrics dict as a Markdown table with colour indicators: 🟢 ≥80%, 🟡 ≥60%, 🔴 <60%
- This string is displayed in the Evaluation Metrics Accordion

---

## File 7: `reasoning_eval.py` — Step-by-Step Reasoning Trace

**Why it exists:** Clinical AI systems must be explainable. This file produces a structured, step-by-step account of exactly how the pipeline reached each recommendation. It serves both clinical analysts (understanding individual codes) and governance reviewers (auditing the decision process).

**`generate_reasoning_trace(report, query) → str`**
- Produces a five-section Markdown document:
  1. Query decomposition (what conditions were extracted and why)
  2. Sub-query search (which sub-queries found which codes)
  3. CrossEncoder reranking (what the scores mean)
  4. Hybrid scoring formula (a table showing exact scores for each signal)
  5. LLM explanation (sample explanations with a note that the LLM cannot add codes)

**`generate_score_breakdown(item) → str`**
- Produces a short breakdown for one specific code showing each scoring component
- Used for drilling into why a particular code ranked where it did

---

## File 8: `feedback_hitl.py` — Human-in-the-Loop Feedback

**Why it exists:** The only way to know whether the system is actually useful for real clinical analysts is to collect feedback. This file provides the 👍/👎 mechanism and stores the results for analysis.

**`record_feedback(run_id, query, vote, recommended_codes, model_name, note) → dict`**
- Saves two things per feedback event:
  1. A JSON file `outputs/feedback/{run_id}_th.json` (or `_th.json`) with full detail
  2. A row appended to `outputs/feedback/feedback_summary.csv` for spreadsheet analysis
- The `run_id` links this feedback record to the corresponding run in `app_audit.py`'s output

**`load_feedback_summary() → dict`**
- Reads the CSV and aggregates: total votes, positive rate, breakdown by model
- Called every time a feedback button is clicked to update the summary panel

**`format_feedback_summary_panel() → str`**
- Returns a Markdown summary table for the "👍👎 Analyst Feedback Summary" Accordion

---

## File 9: `app_audit.py` — Audit Trail and Backtesting

**Why it exists:** For governance and academic evaluation. `feedback_hitl.py` captures analyst opinions; `app_audit.py` captures objective system behaviour.

**`AuditLogger`**
- Call `start_run()` before a query to begin recording
- Call `finish_run()` after to capture results and run validation checks
- Call `save()` to write a JSON file to `outputs/run_logs/`
- Three automatic checks: missing explanations, hallucinated codes, confidence mismatches

**`score_against_gold_standard(recommended_codes, gold_standard_codes, usage_counts) → dict`**
- Computes recall, precision, F1, and patient-weighted recall
- Patient-weighted recall weights each missed code by how many patients it represents nationally
- This is the primary metric for comparing different model/ranker configurations

**`run_backtest(chat_fn, gold_standard_dir, model_name, ranking_model) → dict`**
- Runs the chat function against all NICE DAAR_2025_004 gold-standard files
- Measures how well the system reproduces expert-validated code lists
- Saves a full report to `outputs/backtest/`

---

## How Everything Connects — The Complete Data Flow

```
User types: "obesity with type 2 diabetes"
─────────────────────────────────────────
app.py: process_query() is called

  1. pipeline.is_ready()
     ├── _engine check → is engine loaded?
     └── _db_direct_count check → does ChromaDB have data?
     → returns (True, "")

  2. pipeline.retrieve_and_rank(query)
     │
     ├── pipeline._safe_decompose(query)
     │     └── pod1_pod2.query_decompose()
     │           └── Ollama phi4:mini (via langchain)
     │               → {primary:"obesity", comorbidities:["type 2 diabetes"]}
     │
     ├── pod1_pod2.build_sub_queries()
     │     → ["obesity", "type 2 diabetes", "obesity with type 2 diabetes"]
     │
     ├── For each sub-query:
     │     pod1_pod2.embed_query()
     │         └── SentenceTransformer BGE → 384-float vector
     │     chromadb.collection.query()
     │         → 15 nearest codes by cosine distance
     │
     ├── Deduplicate by SNOMED code
     │
     ├── pod1_pod2.rerank_results()
     │     └── CrossEncoder BGE-reranker
     │         → adds rerank_score to each doc
     │
     └── pod1_pod2.hybrid_rerank()
           → 0.7×semantic + 0.2×usage + 0.1×qof
           → top 10 codes sorted by hybrid_score
           → translated to standard report items

  3. pipeline.add_llm_explanations(report, query)
     └── llm.call_llm()
           └── os.environ["LLM_MODEL"] (llama3.2:1b)
               via Ollama at localhost:11434
               → JSON list of {code, explanation}

  4. app.py: _format_codes_response(report, model_choice)
     → Markdown: provenance header + code cards

  5. ragas_eval.evaluate(report, query)
     ├── _score_faithfulness()  → llm.call_llm() or heuristic
     ├── _score_answer_relevancy()  → uses existing rerank_scores
     └── _score_context_recall()  → sub-query coverage check
     → {faithfulness, answer_relevancy, context_recall, overall}

  6. reasoning_eval.generate_reasoning_trace(report, query)
     → Multi-section Markdown: decomposition → search → reranking → hybrid → LLM

  Returns to Gradio:
    [chat_history]      → gr.Chatbot
    [eval_metrics_md]   → 📊 Evaluation Accordion
    [reasoning_md]      → 🔍 Reasoning Trace Accordion
    [run_id]            → gr.State (for feedback buttons)

User clicks 👍
──────────────
app.py: handle_feedback("thumbs_up", run_id, ...)
  └── feedback_hitl.record_feedback()
        ├── outputs/feedback/{run_id}_th.json
        └── outputs/feedback/feedback_summary.csv
  └── feedback_hitl.format_feedback_summary_panel()
        → updates 👍👎 Feedback Summary Accordion
```

---

## Models Required — What, Where, and Why

### Model 1: `BAAI/bge-small-en` (SentenceTransformer)
- Used by: `pod1_pod2_integrated_V2.py` at import time, and by `ingest_data.py` during ingestion
- Purpose: converts clinical text into 384-dimensional vectors for similarity search
- Downloaded automatically by `sentence-transformers` on first use
- Size: ~130MB
- Why this model: it is a general-purpose sentence embedding model that performs well on medical terminology. BGE (Beijing Academy of AI General Embedding) models consistently outperform older models on medical text retrieval tasks.

### Model 2: `BAAI/bge-reranker-base` (CrossEncoder)
- Used by: `pod1_pod2_integrated_V2.rerank_results()`
- Purpose: more accurately scores each (query, code description) pair by reading them together
- Downloaded automatically by `sentence-transformers` on first use
- Size: ~270MB
- Why separate from Model 1: the embedding model (Model 1) encodes query and document independently and compares them as vectors. The CrossEncoder reads both simultaneously, which is slower but gives better relevance scores. Using CrossEncoder only on the shortlist (top 20 from embedding search) gives the accuracy benefits without the speed cost.

### Model 3: `phi4:mini` (Ollama)
- Used by: `pod1_pod2_integrated_V2.query_decompose()` via LangChain
- Purpose: breaks the user's query into primary condition + comorbidities
- Must be pulled: `ollama pull phi4:mini`
- Why phi4-mini: it is a small, fast model that handles structured JSON output reliably. Query decomposition is a simple classification task that does not require a large model.
- Fallback: if Ollama is offline, `pipeline._safe_decompose()` splits on "with"/"and" instead

### Model 4: Any LLM in the dropdown (Ollama or cloud)
- Used by: `llm.call_llm()` during `pipeline.add_llm_explanations()` and `ragas_eval._score_faithfulness()`
- Purpose: writes plain-English explanations for each recommended code
- Default: `llama3.2:1b` (fast, 1GB, good for development)
- Better quality: `llama3.2`, `mistral`, or any cloud model
- Selected by: the "AI Model" dropdown in the UI

---

## How to Backtest Different Models

Backtesting means running the system against the NICE DAAR_2025_004 gold-standard code lists and measuring how many of the expert-validated codes the system retrieves.

**Running a backtest:**
```python
from app import process_query
from app_audit import run_backtest

# Test llama3.2:1b
results_small = run_backtest(
    chat_fn=process_query,
    gold_standard_dir="data/gold_standard/",
    model_name="llama3.2:1b",
    ranking_model="Hybrid"
)

# Test a cloud model
import os
os.environ["LLM_MODEL"] = "gpt-4o-mini"
results_cloud = run_backtest(
    chat_fn=process_query,
    gold_standard_dir="data/gold_standard/",
    model_name="gpt-4o-mini",
    ranking_model="Hybrid"
)
```

**What the backtest measures:**
- **Recall**: of all codes the expert chose, what percentage did the system find? Higher is better. Missing codes means patients are not counted.
- **Precision**: of all codes the system recommended, what percentage were actually correct? Lower precision means the analyst has to review many false positives.
- **F1**: the balance between recall and precision. Use this when you want a single number.
- **Patient-weighted recall**: recall weighted by each missed code's national NHS usage count. Missing a code used by 500,000 patients is much worse than missing one used by 50.

**What to compare:**
- Model A vs Model B for explanation quality (which produces better faithfulness scores?)
- Hybrid weights 70/20/10 vs 80/10/10 vs 60/30/10 (which retrieves more gold-standard codes?)
- phi4:mini vs llama3.2 for query decomposition (which finds more relevant sub-queries?)

**Important note:** The embedding model (BGE) and CrossEncoder are fixed. They do not change with the model dropdown. Only the explanation and decomposition steps change with model selection. If you want to test different embedding models, you must rebuild the database with a different model in `ingest_data.py` and re-run the full backtest.

---

## Required Files Summary

To run the final app, you need all of these in the same folder:

```
final_app/
├── app.py                        ← run this
├── pipeline.py                   ← imported by app.py
├── pod1_pod2_integrated_V2.py    ← imported by pipeline.py
├── llm.py                        ← imported by pipeline.py and ragas_eval.py
├── ragas_eval.py                 ← imported by app.py
├── reasoning_eval.py             ← imported by app.py
├── feedback_hitl.py              ← imported by app.py
├── app_audit.py                  ← imported optionally for backtesting
├── ingest_data.py                ← run once to build the DB
├── .env                          ← your API keys and model config
├── nhs-logo-880x4951.jpeg        ← optional logo (fallback text badge if missing)
└── data/
    └── chroma_db/                ← ChromaDB built by ingest_data.py
```

You do NOT need:
- `service.py` — only used by `demo.py` for standalone testing
- `demo.py` — only used for testing without the UI
- `prompts.md` — only used by `service.py` / `demo.py`
