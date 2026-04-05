# How the App Works — Plain English Guide

> **Who this is for.** Anyone who wants to understand what each piece of code does and how it all fits together — without needing prior Python experience.

---

## The Big Picture in One Paragraph

When a user types a clinical query into the chat window and presses Send, the app does five things in sequence: it searches a large CSV file of clinical codes to find relevant matches, it passes those matches to a ranking model (Pod 1 or Pod 2) to order them by relevance, it sends the matched codes to an AI language model (Llama, GPT, etc.) to write a plain-English explanation for each one, it formats all of that into a readable response, and it displays the result in the chat window. The dropdown menus at the top of the interface control which AI model and which ranking algorithm are used at steps 3 and 2 respectively.

---

## File by File

---

### `llm.py` — The AI Call Engine

**What it is:** The file responsible for sending a message to a language model and getting a response back. Nothing else.

**Functions:**

#### `_client()`
- **What it does:** Looks at your `.env` file and creates a connection to whichever AI provider is configured.
- **Input:** Nothing — it reads from environment variables automatically.
- **Output:** A configured AI client object, ready to make calls.
- **How it decides which provider to use:** It checks three environment variables in order. If `OLLAMA_BASE_URL` is set, it connects to your local Ollama instance. If `OPENROUTER_API_KEY` is set, it connects to OpenRouter. If `OPENAI_API_KEY` is set, it connects to OpenAI directly. Whichever one it finds first wins.
- **The Ollama trick:** All three providers speak the same API language (the OpenAI format), so the same connection code works for all of them. Ollama just runs that same API on your own machine at `localhost:11434`.

#### `call_llm(system, user)`
- **What it does:** Sends two messages to the AI and returns whatever the AI writes back as a raw string of text.
- **Input:** `system` — the standing instructions for how the AI should behave (written in `prompts.md`). `user` — the specific thing you want the AI to process (the clinical codes and query).
- **Output:** A string of text — usually a JSON structure, but raw text at this stage.
- **Why it returns raw text and not a Python dict:** Because its only job is communication. Turning the text into structured data is `service.py`'s responsibility. Keeping these two jobs in separate functions means you can change one without breaking the other.

---

### `service.py` — The Report Generator

**What it is:** The file that orchestrates one complete call from "here is a clinical payload" to "here is a structured JSON report."

**Functions:**

#### `generate_report(payload)`
- **What it does:** Takes a dictionary of clinical information, sends it to the AI with the correct instructions, and returns a structured result.
- **Input:** A Python dictionary with at minimum a `query` field and a `candidates` list (a list of clinical codes to reason about).
- **Output:** A Python dictionary matching the structure defined in `prompts.md` — with an `items` list and a `synthetic_suggestion` block.
- **The three steps inside it:**
  1. It reads `prompts.md` from disk to get the AI's standing instructions.
  2. It converts the input dictionary to JSON text and passes it to `call_llm()`.
  3. It converts the AI's text response back into a Python dictionary and returns it.
- **Why does it convert to JSON and back?** Because AI models receive and return text, not Python objects. `json.dumps()` converts a Python dict to text; `json.loads()` converts the AI's text response back to a dict.

---

### `prompts.md` — The AI Instructions

**What it is:** A plain-text file that tells the AI exactly how to behave. It is not Python code — it is a set of written rules, like a job description for the AI.

**Key rules it contains:**
- The flag system: every code must be labelled exactly as `CANDIDATE_INCLUDE`, `REVIEW`, `STRATIFIER`, or `UNCLASSIFIED`.
- The priority rule: only populate the priority field if the input data contains an explicit priority — never infer it from numeric scores.
- The explanation rule: explanations must cover what the code represents, what supports it, what is uncertain, and what the analyst should verify.
- The output format: the AI must return a specific JSON structure and nothing else.

**Why is this a `.md` file rather than a Python string?**
Because it changes frequently. Keeping it in a separate file means the team can edit the AI's instructions without touching any Python code and without knowing how to program.

---

### `demo.py` — The Test Runner

**What it is:** A standalone script used during development to check that the whole pipeline works. It is not part of the main chatbot.

**Functions:**

#### `main()`
- **What it does:** Creates one example clinical payload, calls `generate_report()` with it, and prints the result to the terminal.
- **Input:** Nothing — the `SAMPLE_PAYLOAD` at the top of the file is hardcoded.
- **Output:** The full JSON report printed to the terminal.
- **Why does it exist?** When you make a change to any file, running `python demo.py` tells you within seconds whether the pipeline still works end-to-end. It is your fast sanity check before using the full chatbot UI.

**The `SAMPLE_PAYLOAD` dictionary:**
This is deliberately complex — it contains three candidates with different field name formats (`MedCodeId`, `concept_id`, `id`) to test that the AI can handle inconsistent real-world input. Real clinical data from different GP systems uses different field names, so testing with inconsistent input is realistic.

---

### `__init__.py` — The Package Door

**What it is:** A two-line file that makes the `justification` folder importable as a Python package.

**What it contains:**
```python
from .service import generate_report
__all__ = ["generate_report"]
```

**Plain English translation:** "This folder is a Python package. The public thing you can use from it is the `generate_report` function. You can import it by writing `from justification import generate_report`."

**Why does this file need to exist?** Python does not automatically treat every folder as something you can import from. The presence of `__init__.py` is Python's signal that "yes, this folder contains code you can import." Without it, writing `from justification import generate_report` would fail.

**Why does it re-export `generate_report`?** So that other files in the project can write the short form `from justification import generate_report` rather than the longer `from justification.service import generate_report`. It is a convenience that also means if you ever move `generate_report` to a different internal file, only `__init__.py` needs updating — all the external code that imports it stays the same.

---

### `app.py` — The Chatbot Interface

**What it is:** The main application file. It builds the browser-based chat window, handles everything the user types, and connects all the other pieces together.

This file is organised into seven sections. Here is what each one does.

---

#### Section A — Ranking Plug-ins

##### `_rank_pod1(candidates, query)`
- **What it does:** Applies the Pod 1 ranking algorithm to a list of candidate codes.
- **Current status:** Placeholder — returns candidates in their original order with a rank number added.
- **Input:** `candidates` — a list of dictionaries, each with a `MedCodeId`, `term`, and `score`. `query` — the user's search string.
- **Output:** The same list of dictionaries, each now also containing `rank` (1 = most relevant), `confidence_score` (0 to 1), and `ranked_by` (a label shown in the UI).
- **How to connect your ranking code:** Replace the body of this function with a call to your ranking module. The input and output shapes above are the contract your ranking module must satisfy.

##### `_rank_pod2(candidates, query)`
- **What it does:** Same as `_rank_pod1` but for the Pod 2 algorithm.
- **Input / Output:** Identical structure to `_rank_pod1`.

##### `_apply_ranking(candidates, query, choice)`
- **What it does:** Reads the ranking dropdown's value and calls the correct ranking function.
- **Input:** The candidates list, the query, and the string "Pod 1" or "Pod 2" from the dropdown.
- **Output:** The ranked candidates list from whichever ranker was selected.
- **Why this function exists:** It is a single routing point. If you add Pod 3, you add one `elif` here — nothing else in the file needs to change.

---

#### Section B — Model Selection

##### `_get_available_models()`
- **What it does:** Builds the list of AI models shown in the dropdown by checking which provider credentials are in the `.env` file.
- **Input:** Nothing — reads environment variables.
- **Output:** A list of model name strings. If Ollama is configured, it includes Ollama models. If OpenRouter is configured, it adds those. If OpenAI is configured, it adds those. If nothing is configured, it returns a single placeholder.
- **Why dynamic rather than hardcoded?** So the dropdown only shows models that will actually work. If you have only Ollama configured, you should not see GPT-4 in the list.

##### `_get_default_model()`
- **What it does:** Decides which model to pre-select when the app starts.
- **Output:** The value of `LLM_MODEL` from `.env` if that model is in the available list; otherwise the first available model.

##### `_apply_model_choice(model_choice)`
- **What it does:** When the user changes the model dropdown, this function updates the environment so that the AI call layer (`llm.py`) uses the new model.
- **Input:** The model name string selected in the dropdown.
- **Output:** Nothing visible — it silently updates `os.environ["LLM_MODEL"]`.
- **Why does updating an environment variable work?** Because `llm.py` reads `LLM_MODEL` using `os.getenv()` at call time, not at import time. So updating it before each call is enough.

---

#### Section C — CSV Code Loading

##### `_resolve_csv_path()`
- **What it does:** Searches a list of likely locations for the `combined_normalized_codes.csv` file and returns the first one it finds.
- **Output:** A file path if found; `None` if not found anywhere.

##### `_normalize_text(value)`
- **What it does:** Converts a string to lowercase, removes punctuation, and collapses extra spaces so that "Type 2 Diabetes!" and "type 2 diabetes" are treated as the same thing.
- **Input:** Any string.
- **Output:** A cleaned, lowercase, punctuation-free string.

##### `_pick_column(headers, candidates)`
- **What it does:** Given a list of column headers from the CSV and a list of acceptable column names to look for, returns the first match it finds.
- **Why this exists:** Different CSV files use different column names (`code`, `MedCodeId`, `SnomedCTConceptId`). This function handles that inconsistency without failing.

##### `_load_codes_from_csv()`
- **What it does:** Reads the entire CSV into memory once and stores it in a global list called `_cached_codes`.
- **Why only once?** Reading a large file from disk on every user message would make the app very slow. Loading it once at the first query and keeping it in memory makes all subsequent searches fast.

##### `_extract_candidates(text)`
- **What it does:** Searches the cached code list for entries that match the user's query and returns them as candidate codes.
- **Input:** The user's plain-English query string.
- **Output:** A list of candidate dictionaries, each with `MedCodeId`, `term`, `status`, and `score`.
- **Two-pass matching:** First pass looks for codes whose description directly overlaps with the query (exact or substring match). If that finds nothing, a second pass looks for codes that share at least two words with the query — this catches partial matches.

---

#### Section D — Report Building and LLM Explanations

##### `_build_report_from_candidates(candidates)`
- **What it does:** Takes the raw candidate list from the CSV search and structures it into the report format used throughout the app.
- **Input:** List of candidate dictionaries.
- **Output:** A dictionary with an `items` key containing structured code records.

##### `_add_llm_explanations(report, query)`
- **What it does:** Sends the matched codes to the AI and asks it to write a one-to-two sentence plain-English explanation for why each code is relevant to the query.
- **Input:** The structured report dictionary and the original query string.
- **Output:** The same report dictionary, with each item's `explanation` field now populated.
- **Critical safety rule:** The AI is only allowed to write explanations. It is explicitly told not to add new codes. The codes themselves always come from the CSV — never from the AI's own knowledge.

---

#### Section E — Format Output

##### `_format_response(report, model_choice, ranking_choice)`
- **What it does:** Converts the structured report dictionary into a Markdown string that Gradio can display in the chat window with formatting.
- **Input:** The report dictionary, the selected model name, and the selected ranker name.
- **Output:** A Markdown string with headers, code names, confidence scores, and explanations.
- **Why include model and ranker in the output?** Provenance — the analyst can always see which AI model and which ranking algorithm produced a given result, which is important for the audit trail.

---

#### Section F — Chat Handler

##### `chat(message, history, model_choice, ranking_choice)`
- **What it does:** The central function that Gradio calls every single time the user sends a message. It orchestrates the full pipeline.
- **Inputs:**
  - `message` — the text the user just typed
  - `history` — every prior pair of (user message, assistant response) in the current session
  - `model_choice` — the value selected in the AI Model dropdown
  - `ranking_choice` — the value selected in the Ranking Model dropdown
- **Output:** A Markdown string shown as the assistant's reply in the chat window.
- **The pipeline it runs:**
  1. Applies the model choice to the environment
  2. Extracts candidate codes from the CSV
  3. Builds a structured report
  4. Applies the selected ranking model
  5. Merges ranking results back into the report and sorts by rank
  6. Adds AI explanations
  7. Formats and returns the Markdown response

---

#### Section G — NHS Interface

##### `_logo_html()`
- **What it does:** Looks for the NHS logo file in the same folder as `app.py`. If found, encodes it as a base64 image and returns an HTML `<img>` tag. If not found, returns a styled text badge instead.
- **Why base64?** Because Gradio's HTML components cannot serve local files directly from the filesystem. Embedding the image as base64 text inside the HTML is the only reliable way to display it.

##### `_build_interface()`
- **What it does:** Constructs the entire browser interface — header, status bar, dropdowns, chat window, and footer — and returns it as a Gradio `Blocks` object ready to launch.
- **Output:** A configured `gr.Blocks` object.
- **Critical fix from previous version:** `theme` and `css` are passed to `gr.Blocks(theme=..., css=...)`. In the old code they were passed to `demo.launch()`, which does not accept them in Gradio version 4 and above — this would have caused a `TypeError` crash at startup.

---

## How Everything Connects at Runtime

When you run `python app.py` and type a query:

```
User types a message and presses Send
            │
            ▼
    chat()  in app.py
            │
            ├── _apply_model_choice()       sets LLM_MODEL in memory
            ├── _extract_candidates()       searches the CSV
            │       └── _load_codes_from_csv()   (loads file once, then cached)
            │
            ├── _build_report_from_candidates()   structures the matches
            │
            ├── _apply_ranking()            routes to Pod 1 or Pod 2
            │       └── _rank_pod1()  or  _rank_pod2()   (plug-in functions)
            │
            ├── _add_llm_explanations()     calls the AI for plain-English reasons
            │       └── call_llm()  in llm.py
            │               └── _client()  in llm.py   (connects to Ollama/OpenAI)
            │                       └── service.generate_report()  (not used here —
            │                           explanations call call_llm() directly)
            │
            └── _format_response()          converts dict to Markdown string
                        │
                        ▼
            Gradio displays the Markdown in the chat window
```

---

## What Changes When You Connect the Real Ranking Functions

Only two things change in the entire codebase:

**Step 1** — At the top of `app.py`, add an import:
```python
from pod1_ranking import rank_candidates as _pod1_ranker
```

**Step 2** — Inside `_rank_pod1()` in Section A, replace the placeholder body with:
```python
return _pod1_ranker(candidates, query)
```

That is the entire change. The rest of the pipeline — the CSV loading, the LLM explanations, the UI, the routing function — is already in place and already calling `_rank_pod1()`. Your ranking function just needs to accept a list of candidate dicts and a query string, and return the same list with `rank`, `confidence_score`, and `ranked_by` fields added to each item.

---

## What Changes When You Switch from Ollama to an API

Only your `.env` file changes:

```bash
# Comment out Ollama:
# OLLAMA_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=llama3.2:1b

# Add your API key:
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
```

No Python files need to change. The model dropdown will automatically update to show OpenAI models on the next restart, because `_get_available_models()` reads from the environment at startup.
