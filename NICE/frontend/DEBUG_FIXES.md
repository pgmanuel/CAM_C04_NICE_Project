# Debug Report: Query Decomposition and Retrieval Errors

## Summary

Three separate bugs were causing the app to fail after startup. The engine loaded correctly (374,827 codes in ChromaDB), but every query failed before returning any results. This document explains exactly what each bug was, why it happened, which line caused it, and what was changed to fix it.

---

## Bug 1 — 404 Error in Query Decomposition (Root Cause of Broken Queries)

### The error
```
[engine] Decomposition failed: 404 page not found (status code: 404)
```

### Which file
`pod1_pod2_integrated_V2.py`, function `query_decompose()`, line 92–93

### What the code was doing
```python
def query_decompose(query: str, model_name: str = "phi4:mini") -> dict:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm = OllamaLLM(model=model_name, base_url=base_url)
```

### Why this caused a 404

There are **two completely different APIs** for talking to Ollama, and two completely different pieces of code in this project that use them. This is the root of the confusion.

**API 1 — OpenAI-compatible API** (used by `llm.py`):
- Endpoint: `http://localhost:11434/v1/chat/completions`
- Used by: the `OpenAI` Python SDK
- The `/v1` suffix is required because this is Ollama pretending to be the OpenAI API
- This is why `OLLAMA_BASE_URL=http://localhost:11434/v1` is correct in `.env`

**API 2 — Ollama Native API** (used by `OllamaLLM` in LangChain):
- Endpoint: `http://localhost:11434/api/generate`
- Used by: `langchain_ollama.OllamaLLM`
- LangChain automatically appends `/api/generate` to whatever `base_url` you provide
- The base URL must be the bare host with **no `/v1`**

When `.env` contains `OLLAMA_BASE_URL=http://localhost:11434/v1`, and `query_decompose()` passes this directly to `OllamaLLM(base_url=base_url)`, LangChain appends `/api/generate` to it:

```
http://localhost:11434/v1  +  /api/generate
=
http://localhost:11434/v1/api/generate   ← this endpoint does not exist → 404
```

The correct URL for `OllamaLLM` would be:
```
http://localhost:11434  +  /api/generate
=
http://localhost:11434/api/generate   ← this is the native Ollama endpoint → 200 OK
```

### Why previous fixes didn't work

Previous attempts changed the `.env` value, but that broke `llm.py` which genuinely needs `/v1` for the OpenAI SDK. The `.env` value is correct for `llm.py` — the problem is that `query_decompose()` was using that same value without stripping the path.

### The fix

Strip the `/v1` suffix inside `query_decompose()` before passing to `OllamaLLM`. This way the `.env` value stays correct for `llm.py`, and `OllamaLLM` gets the bare host it needs.

**File changed:** `pod1_pod2_integrated_V2.py`

```python
# BEFORE (broken):
base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
llm = OllamaLLM(model=model_name, base_url=base_url)

# AFTER (fixed):
raw_url  = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
base_url = raw_url.rstrip("/")       # remove trailing slash
if base_url.endswith("/v1"):
    base_url = base_url[:-3]         # strip /v1 → leaves bare host
llm = OllamaLLM(model=model_name, base_url=base_url)
```

**Why `[:-3]` removes `/v1`:** The string `/v1` is 3 characters long. After `rstrip("/")` ensures no trailing slash, `base_url[:-3]` removes the last 3 characters, turning `http://localhost:11434/v1` into `http://localhost:11434`.

---

## Bug 2 — ChromaDB `ids` Error in Retrieval

### The error
```
[pipeline] Sub-query '...' error: Expected include item to be one of
documents, embeddings, metadatas, distances, uris, data, got ids in query.
```

### Which file
`pipeline.py`, function `retrieve_and_rank()`, line 210

### What the code was doing
```python
res = _engine.collection.query(
    query_embeddings=[emb], n_results=15,
    include=["documents", "metadatas", "distances", "ids"]   ← "ids" here
)
```

### Why this caused an error

ChromaDB changed its API between versions. In the version installed in this environment (≥ 0.4.x), the `include` parameter only accepts these values:

```
documents | embeddings | metadatas | distances | uris | data
```

**`ids` is not in that list.** In newer ChromaDB, code IDs are always returned by default — they do not need to be requested. Explicitly asking for `"ids"` is now treated as an invalid argument and raises a validation error.

In older ChromaDB (< 0.4), you had to ask for IDs explicitly. The code was written for the older API and was not updated when ChromaDB was upgraded.

### The fix

Remove `"ids"` from the `include` list. The code that reads `res["ids"][0]` on the next line still works — ChromaDB returns IDs automatically regardless of whether they are requested.

**File changed:** `pipeline.py`

```python
# BEFORE (broken):
res = _engine.collection.query(
    query_embeddings=[emb], n_results=15,
    include=["documents", "metadatas", "distances", "ids"]
)

# AFTER (fixed):
res = _engine.collection.query(
    query_embeddings=[emb], n_results=15,
    include=["documents", "metadatas", "distances"]   ← "ids" removed
)
# res["ids"][0] still works on the next lines — IDs always come back
```

### Why this was the second error and not the first

Bug 1 (404) happened during query decomposition, which runs first. When decomposition fails, `_safe_decompose()` falls back to splitting the query on "with"/"and". That fallback produces a single sub-query (the whole or split query). That sub-query is then passed to the ChromaDB search, where Bug 2 immediately causes a second crash. So both errors appear in every failed query — Bug 1 always triggers Bug 2.

---

## Bug 3 — Gradio 6.0 CSS Warning

### The warning
```
UserWarning: The parameters have been moved from the Blocks constructor to the
launch() method in Gradio 6.0: css. Please pass these parameters to launch() instead.
```

### Which file
`app.py`, function `_build_interface()`, line 326

### What the code was doing
```python
with gr.Blocks(title="NICE Clinical Code Assistant", css=_CSS) as demo:
```

### Why this appears

In Gradio 6.0, the `css` parameter was moved from `gr.Blocks()` to `launch()`. Passing it to `gr.Blocks()` still works (the CSS is applied) but triggers a deprecation warning. This is not causing query failures but will become an error in a future Gradio version.

### The fix

**File changed:** `app.py`

```python
# BEFORE (deprecated):
with gr.Blocks(title="NICE Clinical Code Assistant", css=_CSS) as demo:
    ...
_build_interface().launch(server_name="127.0.0.1", server_port=7860)

# AFTER (Gradio 6.0 correct):
with gr.Blocks(title="NICE Clinical Code Assistant") as demo:
    ...
_build_interface().launch(server_name="127.0.0.1", server_port=7860, css=_CSS)
```

`_CSS` is a module-level variable in `app.py`, so it is accessible at the `launch()` call without any structural change.

---

## Summary of All Changes

| Bug | Error message | File | Change |
|---|---|---|---|
| 1 | `404 page not found` | `pod1_pod2_integrated_V2.py` | Strip `/v1` from URL before passing to `OllamaLLM` |
| 2 | `got ids in query` | `pipeline.py` | Remove `"ids"` from ChromaDB `include` list |
| 3 | Gradio `UserWarning: css` | `app.py` | Move `css=_CSS` from `gr.Blocks()` to `launch()` |

---

## Why the `.env` Value Is Correct and Should Not Change

The `.env` file contains `OLLAMA_BASE_URL=http://localhost:11434/v1`. This is correct and should stay as-is.

- `llm.py` uses the `OpenAI` Python SDK, which requires the `/v1` path. If you remove `/v1`, `llm.py` will break.
- `pod1_pod2_integrated_V2.py` uses `OllamaLLM` from LangChain, which requires the bare host without `/v1`. The fix strips the suffix inside `query_decompose()` so the same `.env` value serves both.

**Do not change `.env`.** The fix is entirely in `pod1_pod2_integrated_V2.py`.

---

## Expected Terminal Output After Fix

When `app.py` starts successfully you should see:
```
[env] OLLAMA_BASE_URL: http://localhost:11434/v1
[env] LLM_MODEL:       llama3.2:1b
[ChromaDB] Using persistence directory: ...
[pipeline] Engine loaded. ChromaDB: 374,827 codes.
============================================================
Engine:  ✓ Ready
DB:      374,827 codes
```

When you submit a query you should see:
```
[pipeline] Sub-query 'obesity' error:        ← nothing here
[pipeline] Sub-query 'type 2 diabetes' error: ← nothing here
```
And a result in the chat window with ranked codes.

If `query_decompose` still fails for any reason, `_safe_decompose()` in `pipeline.py` catches it and falls back to splitting on "with"/"and" — so the query will still return results, just with less precise sub-query decomposition.
