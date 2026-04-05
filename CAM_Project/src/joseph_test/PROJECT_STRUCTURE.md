# Project Structure: What Every File Does and Why

> **Who this is for.** Anyone on the team who wants to understand why the project is laid out the way it is — including why we have a folder called `src`, why there is a file called `__init__.py`, and what each file's single job is.

---

## The Folder Structure

```
CAM_Project/
│
├── src/                          ← all reusable Python code lives here
│   └── justification/            ← one module within src
│       ├── __init__.py           ← makes this folder a Python package
│       ├── llm.py                ← knows how to talk to the LLM
│       ├── service.py            ← knows how to build a report
│       ├── prompts.md            ← the instructions we give the LLM
│       ├── demo.py               ← runs a single example (for testing)
│       ├── app.py                ← the local chatbot frontend (NEW)
│       ├── requirements.txt      ← Python packages this module needs
│       └── .env.example          ← template for your environment settings
│
└── (other folders for data, notebooks, etc. go alongside src/)
```

---

## Why Is There a Folder Called `src`?

`src` stands for **source** — it is the conventional name for the folder that contains your project's actual Python code as opposed to configuration files, data files, notebooks, or documentation.

The reason you separate source code from everything else is that a real project contains many different types of files, and they need different treatment:

- **Source code** (`src/`) is run, imported, tested, and version-controlled carefully.
- **Data** (`data/`) is large, often sensitive, and never committed to Git.
- **Notebooks** (`notebooks/`) are exploratory and experimental — they import from `src/` rather than containing logic themselves.
- **Documentation** (`docs/`) is read by humans, not executed.

If you put everything in one flat folder, it becomes impossible to tell which files are core logic and which are scratch work. `src/` is a clear signal: *this is the real code*.

The further benefit is that anything inside `src/` can be imported by any other file in the project. A notebook in `notebooks/` can do `from src.justification import generate_report` without any path tricks, because Python knows to look in `src/`.

---

## Why Is There a File Called `__init__.py`?

This is one of the most confusing Python concepts for beginners, so let us be precise.

**The short answer:** `__init__.py` makes a folder into a Python package — a folder of code that can be imported like a module.

**The longer explanation:**

Python does not automatically treat every folder as something you can import from. If you have a folder called `justification` and you try to write `from justification import generate_report`, Python will raise an error unless it knows that folder is a package. The presence of `__init__.py` is what tells Python "yes, this folder is a package you can import from."

Think of it like a front door. The folder is a building, and `__init__.py` is the front door that makes it accessible from outside.

**What goes inside `__init__.py`?**

In this project, `__init__.py` contains:

```python
from .service import generate_report

__all__ = ["generate_report"]
```

The `.service` (note the dot) means "import from the `service.py` file that is in the same folder as this `__init__.py`". The `__all__` list declares which names are publicly available when someone writes `from justification import *`.

The practical effect is that any other file in the project can write:

```python
from justification import generate_report
```

instead of the longer:

```python
from justification.service import generate_report
```

This is a minor convenience but it also means you control what the public interface of your module is. If you later refactor and move `generate_report` into a different internal file, the import in `__init__.py` is the only thing that needs changing — all external code continues to work unchanged.

---

## Each File Explained

---

### `llm.py` — The LLM Client

**Single job:** Create a configured API client and make one call to the language model.

**What it does not do:** Parse responses, apply business logic, or know anything about clinical codes.

This file contains two functions:

`_client()` builds an `OpenAI` client object pointed at the right provider. The underscore prefix is a Python convention meaning "this is internal — don't call it directly from outside this file." All three providers (Ollama, OpenRouter, OpenAI) speak the same API format, so the same client code works for all three. The function checks which environment variables are set and returns the appropriate client.

`call_llm(system, user)` sends two messages to the model and returns the raw response as a string. It takes `system` (the standing instructions) and `user` (the specific request) because this is the format all modern LLM APIs use. The `system` message sets the model's behaviour; the `user` message is the actual query.

**Why does it return a raw string rather than a parsed dict?**

Because `llm.py`'s only job is communication. Parsing the response into structured data is `service.py`'s job. Keeping these concerns separate means you can change how parsing works without touching the LLM call, and change the LLM provider without touching the parsing logic.

---

### `service.py` — The Report Generator

**Single job:** Take a structured payload dict, call the LLM, and return a structured result dict.

This file contains one function, `generate_report(payload)`, which does three things in sequence:

1. Reads the system prompt from `prompts.md`
2. Converts the payload to a JSON string to use as the user message
3. Calls `call_llm()` and parses the response back into a dict

**Why does it read the prompt from a file rather than hardcoding it?**

Because the prompt will change frequently as you iterate. Reading it from a `.md` file means you can edit the prompt and re-run without changing any Python. It also means the prompt is human-readable in any text editor, which matters when non-technical team members want to review or adjust the instructions.

**Why is the payload serialised with `json.dumps()`?**

The LLM receives text, not Python objects. `json.dumps(payload)` converts the Python dict into a JSON string that the LLM can read. `default=str` handles any values in the payload that are not JSON-serialisable (such as datetime objects), converting them to their string representation rather than raising an error.

---

### `prompts.md` — The LLM Instructions

**Single job:** Contain the system prompt that tells the LLM exactly how to behave.

This is a Markdown file, not a Python file. It is human-readable, which is intentional — the prompt is documentation as much as it is code.

The prompt defines four categories of rules:

**Flag rules** — the LLM must assign exactly one of four values (`CANDIDATE_INCLUDE`, `REVIEW`, `STRATIFIER`, `UNCLASSIFIED`) to each code. This prevents the LLM from inventing its own categories or using vague language. The rules also explicitly prohibit copying raw status values from the payload into the flag field — a specific failure mode that was observed during development.

**Priority rules** — the LLM is only allowed to set priority if the payload contains an explicit priority field. It must not infer priority from score values. This prevents the LLM from making decisions the system was not designed to make.

**Explanation rules** — the LLM must write analyst-facing explanations covering what the code represents, what evidence supports it, what uncertainty remains, and what the analyst should verify. These are not internal reasoning notes; they are the output the NICE analyst reads.

**Evidence rules** — each evidence item must be an object with `text` and `source` keys, where `source` indicates where in the payload the evidence came from. This creates the audit trail.

The JSON output schema at the bottom of the file is the single most important part of the prompt. The LLM is told to return *exactly* this structure, no more and no less. This is what makes the response parseable by `service.py`.

---

### `demo.py` — The Test Runner

**Single job:** Run a single example through the full pipeline so you can check that everything is wired up correctly.

`demo.py` is not part of the production system. It is a development tool. When you make a change to any file, running `python demo.py` tells you within a few seconds whether the change broke anything.

`SAMPLE_PAYLOAD` at the top of the file is deliberately complex — it contains three candidates with different field names and structures (`MedCodeId`, `concept_id`, `id`). This tests that the LLM can handle inconsistent input without failing. In real clinical data, field names are not consistent across sources, so the test payload reflects that reality.

---

### `app.py` — The Local Chatbot Frontend (New)

**Single job:** Provide a browser-based chat interface that wraps the existing `service.py` pipeline.

This file uses Gradio to create a chat window at `http://localhost:7860`. It does not replace any existing code — it adds a human-friendly layer on top.

The file has four logical sections:

`_build_payload_from_conversation()` converts a plain-English chat message into the structured dict that `service.py` expects. It extracts clinical terms from the message and packages them as candidates. This is intentionally simple for now — the real pipeline would use the QOF lookup and semantic search tools from Phase 2.

`_extract_candidates()` scans the message for known clinical terms and maps them to approximate SNOMED codes. This is a placeholder that gives the LLM something to reason about even before the full retrieval pipeline is connected.

`_format_response()` converts the JSON report from `service.py` into human-readable Markdown. Gradio renders Markdown natively, so bold text, emojis, and headers appear properly formatted in the chat window.

`chat()` is the function Gradio calls every time the user submits a message. It orchestrates the three functions above and handles errors gracefully — returning plain-English error messages rather than Python tracebacks.

`_build_interface()` creates the Gradio `ChatInterface` with example queries, descriptions, and styling. The example queries appear as clickable buttons under the input box, which is useful for demonstrations.

---

### `requirements.txt` — Python Dependencies

**Single job:** List every Python package this module needs.

Currently contains three packages:

`openai` — the Python SDK for the OpenAI API. Used by `llm.py`. Also works with Ollama and OpenRouter because all three use the same API format.

`python-dotenv` — reads the `.env` file and makes its values available as environment variables via `os.getenv()`. Without this, you would have to set environment variables in the terminal before every run, which is error-prone and not reproducible.

`gradio` — the web framework used by `app.py` to create the chat interface. Only needed if you are running the chatbot frontend; `demo.py` and `service.py` do not require it.

---

### `.env.example` — Environment Variable Template

**Single job:** Document every configuration option and provide sensible defaults.

`.env` files are how Python applications receive configuration that should not be hardcoded — API keys, model names, timeouts. The `.example` suffix means this file is committed to Git as a template; the actual `.env` file (which contains real secrets) is in `.gitignore` and never committed.

The three sections correspond to the three providers in `llm.py`. You uncomment the section for your chosen provider and comment out the others. This is deliberate — having all three visible in the template makes it easy to switch providers without having to look up the environment variable names.

---

## How the Files Connect at Runtime

When you run `python app.py` and type a message:

```
app.py              receives the message
  │
  ├── _extract_candidates()    identifies clinical terms
  ├── _build_payload_from_conversation()   packages them
  │
  └── service.generate_report(payload)
          │
          ├── reads prompts.md       loads the system prompt
          └── llm.call_llm()         sends to the LLM
                  │
                  ├── checks .env    finds OLLAMA_BASE_URL or API key
                  └── OpenAI client  makes the API call
                          │
                          └── returns JSON string
          │
          └── json.loads()    parses back to dict
  │
  └── _format_response()      converts to Markdown
          │
          └── Gradio           renders in the browser
```

Every arrow in this diagram crosses a file boundary. Each file has one job and knows nothing about the files on either side of it except through the function signatures they expose. This is why adding the Gradio frontend required only one new file (`app.py`) and one small change to `llm.py` — because the other files were already doing clean, separated work.

---

## Quick Start

```bash
# 1. Navigate to the joseph_test module
cd CAM_Project/src/joseph_test

# 2. Create a virtual environment (keeps dependencies isolated)
python3 -m venv .venv
source .venv/bin/activate        # Mac/Linux
# .venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your environment
cp .env.example .env
# Open .env and set LLM_MODEL and your chosen provider key

# 5a. For local testing with Ollama (no API key needed):
#     Install Ollama from https://ollama.com
#     Then run: ollama pull llama3.2:1b
#     Make sure Ollama is running, then:
python app.py

# 5b. To test with a single example without the UI:
python demo.py
```

Open `http://localhost:7860` in your browser. The chatbot is ready.

---

## Switching From Local to API

When you are ready to use a cloud API instead of Ollama, the only change is in `.env`:

```bash
# Comment out the Ollama lines:
# OLLAMA_BASE_URL=http://localhost:11434/v1
# LLM_MODEL=llama3.2:1b

# Uncomment and fill in your API key:
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4o-mini
```

No Python files need to change. This is the benefit of putting all provider logic in one place (`llm.py`) and driving configuration through environment variables.
 

## Troubleshooting running Ollama 

Found it. There are two separate problems, one hiding the other. Let me show you both clearly before fixing them.
Problem 1 — the BOM character. Look at line 1 of your .env file:
﻿OLLAMA_BASE_URL=http://localhost:11434/v1
That ﻿ at the very start is a BOM (Byte Order Mark) — an invisible UTF-8 signature that Windows Notepad/VS Code adds when saving files. Python reads the first key as ﻿OLLAMA_BASE_URL (with the invisible character stuck to the front) instead of OLLAMA_BASE_URL, so os.getenv('OLLAMA_BASE_URL') returns None. This is why the model loads but the provider never shows. It is the root cause.
Problem 2 — the duplicate load_dotenv(). Lines 39 and 42 both call load_dotenv(). The second call at line 42 has no path argument so it searches from wherever PowerShell's working directory is — which may not be your script folder. Even without the BOM issue, this second call could load a different .env or nothing at all.
The fix is in two places: app.py (remove the duplicate call, add encoding='utf-8-sig' to strip the BOM automatically) and the .env file itself (resave without BOM).Demonstrate the BOM problem exactlyDemonstrate the BOM problem exactlyConfirmed. Now fixing both files:Created a file, ran a commandCreated a file, ran a commandNow write the fixed .env without BOM:Ran a command, read a fileRan a command, read a fileTwo files to replace. Here is a precise breakdown of what was wrong and every change made.

Root Cause Explained
Your troubleshooting correctly identified the duplicate load_dotenv() as suspicious, but that was actually the secondary issue. The primary cause is visible right at the top of the .env file you uploaded — look at line 1 in the raw file view:
﻿OLLAMA_BASE_URL=http://localhost:11434/v1
That ﻿ character at the very front is a BOM (Byte Order Mark, Unicode codepoint U+FEFF). Windows Notepad — and some VS Code save modes on Windows — silently prepend three bytes (\xef\xbb\xbf) to the start of any UTF-8 file. When Python reads the file with the default utf-8 encoding, it cannot distinguish the BOM from real content, so it treats it as part of the first key name. The key gets stored in memory as ﻿OLLAMA_BASE_URL (with that invisible character fused to the front). When llm.py then calls os.getenv('OLLAMA_BASE_URL') — asking for the clean name — it gets None. The URL was there all along; Python just could not find it under the name you were looking for.
The second issue was the duplicate load_dotenv() on lines 39 and 42. Both calls would have failed to set OLLAMA_BASE_URL because of the BOM, but even without that bug, the bare load_dotenv() on line 42 searches from PowerShell's current working directory, not the script folder. When you run a script with its full path (as you did: c:/Users/.../app.py), those two directories are often different.

Every Change in app.py
Line removed: load_dotenv() at old line 42. The bare call with no arguments is gone entirely. One explicit call is enough and is always unambiguous.
load_dotenv() call updated with encoding="utf-8-sig". The utf-8-sig codec is specifically designed to strip a BOM before parsing. If the file has a BOM, it removes it silently. If the file has no BOM, it works identically to plain utf-8. This one argument fixes the root cause.
override=True added. If OLLAMA_BASE_URL happened to exist as a system environment variable from a previous session, the old code would silently ignore your .env file. override=True makes the .env always win.
_env_path = Path(__file__).parent / ".env" made explicit. Using pathlib.Path instead of os.path.join is cleaner and easier to read, and it is stored in a variable so the startup diagnostics can print it and check whether the file actually exists.
Startup diagnostics expanded. When Provider shows as NONE, the app now prints the exact .env path it checked, whether the file exists, whether a BOM was detected, and the first 60 characters of the file content. This means you can see exactly what went wrong from the terminal output, without having to read source code.
from datetime import datetime, timezone moved to the top. It was at line 50 in the original, after the service import. Imports should always be at the top of the file — this is a Python style convention but also prevents confusing NameError bugs.

The .env File
The new .env file was written by Python with encoding='utf-8' explicitly — no BOM. I also changed USE_JSON_RESPONSE_FORMAT from true to false for local Ollama testing. The llama3.2:1b model is a small model and it sometimes returns a JSON parsing error when the response_format={"type": "json_object"} parameter is set, because not all small models support that API parameter. With false, the service relies on the prompt instruction ("Return only valid JSON") instead, which small models handle more reliably.
If you need to recreate the .env yourself in VS Code to avoid the BOM issue in future, open the Command Palette (Ctrl+Shift+P), type Change File Encoding, and choose Save with Encoding → UTF-8 (not UTF-8 with BOM).


## Explanation of Structure

Analysis of app.py — Step by Step
Entry Point (Lines 442-460): if __name__ == "__main__":

This is Python's standard way to run code only when the script is executed directly (not imported as a module).

if __name__ == "__main__":
    # Only runs when: python app.py
    # Does NOT run when: from app import chat

Then it:

Prints the welcome banner
Calls _build_interface() to create the Gradio UI
Calls .launch() to start the web server on http://127.0.0.1:7860
Full Pipeline Breakdown
STEP 1: Load Environment (Lines 24-44)

Reads .env file (with UTF-8 BOM handling)
Sets OLLAMA_BASE_URL, LLM_MODEL, etc.
Imports service.py AFTER env vars are set
STEP 2: Extract Candidates from CSV (Lines 150-200)

_extract_candidates(text) searches _cached_codes for matching terms
Returns list of real SNOMED codes from your combined_normalized_codes.csv
Example: User types "obesity" → finds all obesity-related codes in CSV
STEP 3: Build Report (Lines 207-222)

Takes extracted candidates (real codes from CSV)
Wraps each in a structured dict with code, term, priority, explanation
No synthetic code generation — only real CSV data
STEP 4: Add LLM Explanations (Lines 229-270)

Calls LLM to explain WHY each code matches the query
LLM does NOT generate codes — only explains existing ones
Fallback: "Matched from clinical terminology dataset"
STEP 5: Format Output (Lines 277-293)

Converts report to readable Markdown
Shows: ✅ **Term** code | Priority | Explanation
STEP 6: Main Chat Handler (Lines 300-318)

Receives user message
Runs steps 1-5 in sequence
Returns formatted response to Gradio UI
STEP 7: Build & Launch UI (Lines 325-357)

Creates Gradio ChatInterface
Shows model name + provider (Ollama/OpenAI/OpenRouter)
Defines example prompts

Why This Design Works
User types: "obesity with type 2 diabetes"
    ↓
_extract_candidates() → searches CSV for "obesity", "diabetes"
    ↓
_build_report_from_candidates() → wraps found codes
    ↓
_add_llm_explanations() → LLM explains relevance
    ↓
_format_response() → pretty print to UI
    ↓
User sees: Real SNOMED codes + explanations
No synthetic code generation — only CSV data + LLM explanations.

