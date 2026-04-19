# NICE Clinical Code Assistant — Change Report & Integration Notes

> **Audience:** This document is written as a teacher explaining to a student *why* each change was made, *how* it works, and *what problem* it solves. Keep it. It is your long-term reference.

---

## 1. File Relationship Audit — Are All Files Related to `app.py`?

The first question to answer is: of the nine files you uploaded, which ones actually connect to `app.py`?

### Direct imports in `app.py`

```python
import pipeline        # retrieval + hybrid ranking (not in your upload — separate file)
import ragas_eval      # ← uploaded ✅
import reasoning_eval  # ← uploaded ✅
import feedback_hitl   # ← uploaded ✅
import app_audit       # ← uploaded ✅ — but was NOT imported before our change
```

### Files that connect to `app.py` via a shared dependency (`llm.py`)

| File | Connects how? | Status |
|------|--------------|--------|
| `ragas_eval.py` | `from llm import call_llm` inside `_score_faithfulness()` | ✅ Used |
| `reasoning_eval.py` | Standalone — no LLM import, only reads report dicts | ✅ Used |
| `feedback_hitl.py` | Standalone — writes CSV/JSON to disk | ✅ Used |
| `app_audit.py` | Standalone — writes JSON to disk | ✅ Integrated by our change |
| `llm.py` | Called by `ragas_eval.py` (and by `pipeline.py` indirectly) | ✅ Indirect |
| `service.py` | Calls `llm.py` + reads `prompts.md` | ⚠️ Only called by `demo.py` |
| `demo.py` | Calls `service.py` as a CLI test script | ⚠️ Standalone test — not linked to app.py |
| `prompts.md` | Read by `service.py` | ⚠️ See Section 2 |

### Summary

Seven of the nine files are meaningfully related to `app.py`. The two exceptions are:

- **`demo.py`** — a standalone test runner that calls `service.py` directly. It is useful for quick sanity tests but has no connection to the Gradio UI. Think of it as a "run a single query from the command line" tool.
- **`service.py` + `prompts.md`** — these form their own mini-pipeline (`demo.py → service.py → llm.py`) that is separate from the main `pipeline.py → llm.py` flow used by `app.py`. See Section 2.

---

## 2. Is `prompts.md` Being Used by `app.py`?

**Short answer: No — not through `app.py`'s main pipeline.**

Here is the data flow for both paths, side by side:

```
Path A (app.py):
  app.py
    └── pipeline.retrieve_and_rank()
          └── pipeline.add_llm_explanations()
                └── llm.call_llm(system_prompt, user_prompt)
                      ↑
                      system_prompt is defined INSIDE pipeline.py
                      (prompts.md is NOT read here)

Path B (demo.py only):
  demo.py
    └── service.generate_report(payload)
          └── (Path(__file__).parent / "prompts.md").read_text()   ← reads prompts.md
                └── llm.call_llm(system_prompt, user_prompt)
```

### Why this matters

`prompts.md` contains important clinical governance rules:

```
Flag rules — `flag` must be exactly one of: CANDIDATE_INCLUDE | REVIEW | STRATIFIER | UNCLASSIFIED
...
- Do NOT hallucinate codes, terms, or clinical facts.
- Return only valid JSON. No prose outside the JSON.
```

These rules are currently **only applied when you run `demo.py`**. When analysts use the Gradio app, `pipeline.add_llm_explanations()` uses whatever system prompt is hardcoded inside `pipeline.py`.

### What you should do (recommended next step — not in this PR)

In `pipeline.py`, wherever `add_llm_explanations()` builds its system prompt, replace the hardcoded string with:

```python
# In pipeline.py — recommended future change
from pathlib import Path

_SYSTEM_PROMPT = (Path(__file__).parent / "prompts.md").read_text(encoding="utf-8")

def add_llm_explanations(report: dict, query: str) -> dict:
    ...
    raw = call_llm(system=_SYSTEM_PROMPT, user=json.dumps(payload))
```

This one change would bring `prompts.md`'s governance rules into the live app without touching any other file.

---

## 3. Module Integration Audit — `app_audit.py`, `ragas_eval.py`, `reasoning_eval.py`

### 3a. `ragas_eval.py` — Was it integrated?

**Yes — already working before our changes.**

In the original `app.py`, `process_query()` already called it:

```python
# Original app.py — already present
try:
    metrics       = ragas_eval.evaluate(report, message)
    eval_panel_md = ragas_eval.format_eval_panel(metrics)
except Exception as e:
    eval_panel_md = f"*Evaluation unavailable: {e}*"
```

And the UI already had the accordion:

```python
with gr.Accordion("📊 Evaluation Metrics (RAGAS-style)", open=False):
    eval_panel = gr.Markdown(...)
```

✅ **Status: Fully integrated. No changes needed.**

### 3b. `reasoning_eval.py` — Was it integrated?

**Yes — already working before our changes.**

```python
# Original app.py — already present
try:
    reasoning_md = reasoning_eval.generate_reasoning_trace(report, message)
except Exception as e:
    reasoning_md = f"*Reasoning trace unavailable: {e}*"
```

And the UI already had:

```python
with gr.Accordion("🔍 Reasoning Trace", open=False):
    reasoning_panel = gr.Markdown(...)
```

✅ **Status: Fully integrated. No changes needed.**

### 3c. `app_audit.py` — Was it integrated?

**No — this was the key missing piece.**

`app_audit.py` was uploaded and available, but `app.py` never imported it. The audit module was completely disconnected from the running application. All of its valuable features (validation flags, code provenance, run comparison, backtesting) existed only on disk, never accessible through the UI.

---

## 4. Changes Made to `app.py`

### Change 1 — Import `app_audit` and create a singleton logger

**Where:** Top of `app.py`, after the other imports.

**Before:**
```python
import pipeline
import ragas_eval
import reasoning_eval
import feedback_hitl
# app_audit was never imported
```

**After:**
```python
import pipeline
import ragas_eval
import reasoning_eval
import feedback_hitl
import app_audit  # ← NEW

# One shared AuditLogger instance for the life of the app process.
# "Singleton" means we create it once at startup, not once per query.
# All queries share the same output directory: outputs/run_logs/
_audit = app_audit.AuditLogger(output_dir="outputs/run_logs")
```

**Why a singleton?** If you created a new `AuditLogger()` inside `process_query()`, each call would create a new in-memory store and the records from previous calls would be unreachable. One shared instance means `finish_run()` can always find the record that `start_run()` created, no matter how much time passed between them.

---

### Change 2 — Replace manual `run_id` with `_audit.start_run()`

**Where:** Inside `process_query()`.

**Before:**
```python
run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
# This generated an ID but never told app_audit about it.
# The audit logger had no record of this run.
```

**After:**
```python
# _audit.start_run() generates the run_id AND creates an in-memory
# RunRecord so that finish_run() and save() can populate it later.
run_id = _audit.start_run(message, model_choice, "ChromaDB Hybrid")
```

**Why this matters:** Previously the `run_id` was just a label attached to the feedback record (so analysts could correlate feedback with runs). Now it is a real key into the audit logger's internal dictionary. The audit logger can answer: "show me the codes and validation flags for this exact run."

---

### Change 3 — Call `finish_run()` and `save()` after the report is built

**Where:** Inside `process_query()`, after `pipeline.add_llm_explanations()`.

**Before:**
```python
report      = pipeline.add_llm_explanations(report, message)
response_md = _format_codes_response(report, model_choice)
# Nothing — audit logger never heard about this run's results.
```

**After:**
```python
report      = pipeline.add_llm_explanations(report, message)
response_md = _format_codes_response(report, model_choice)

try:
    _audit.finish_run(run_id, report.get("items", []))
    _audit.save(run_id)
    audit_md = _format_audit_panel(run_id)
except Exception as e:
    audit_md = f"*Audit trail unavailable: {e}*"
```

**What `finish_run()` does (from `app_audit.py`):**

```python
def finish_run(self, run_id: str, report_items: list[dict]) -> None:
    # 1. Records each returned code as a CodeProvenance dataclass
    for item in report_items:
        cp = CodeProvenance(
            snomed_code=      item.get("code", "UNKNOWN"),
            confidence_score= float(item.get("confidence_score", 0.0)),
            ...
        )
        rec.codes.append(cp)

    # 2. Runs automatic validation checks:
    #    - MISSING_EXPLANATION: code has no plain-English reason
    #    - POSSIBLE_HALLUCINATION: code.snomed_code == "UNKNOWN"
    #    - CONFIDENCE_MISMATCH: high score but low initial match
    self._validate(rec)
```

**The try/except wrapper** is deliberate. Audit logging is a secondary concern — if it fails (e.g. disk is full), the analyst still sees their codes. The app degrades gracefully, logging an error rather than crashing.

---

### Change 4 — Add `_format_audit_panel()` helper function

**Where:** New function in `app.py`, Section B2.

This function reads the completed `RunRecord` from the audit logger's memory and formats it as Markdown for the UI:

```python
def _format_audit_panel(run_id: str) -> str:
    rec = _audit._active.get(run_id)
    if not rec:
        return "*No audit record available for this run.*"

    # Show run metadata
    lines = [
        f"**Run ID:** `{rec.run_id}`",
        f"**Model:** `{rec.model_name}`  **Ranker:** `{rec.ranking_model}`",
        ...
    ]

    # Show validation flags with severity icons
    severity_icon = {"CRITICAL": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
    for flag in rec.validation_flags:
        icon = severity_icon.get(flag["severity"], "⚪")
        lines.append(f"{icon} **[{flag['severity']}]** `{flag['type']}` — {flag['message']}")

    # Show code provenance table (top 5)
    lines += ["| Rank | Code | Term | Score | Source |", ...]
    for cp in rec.codes[:5]:
        lines.append(f"| #{cp.rank} | `{cp.snomed_code}` | {cp.term} | ... |")

    return "\n".join(lines)
```

**Why format it in `app.py` rather than `app_audit.py`?** Separation of concerns. `app_audit.py` is responsible for *recording* audit data; `app.py` is responsible for *displaying* things in the UI. Mixing Gradio-flavoured Markdown formatting into `app_audit.py` would couple the audit module to the UI, making it harder to reuse in other contexts (e.g. an API endpoint or a scheduled backtest job).

---

### Change 5 — Add Audit Trail accordion to the UI

**Where:** `_build_interface()`, after the Reasoning Trace accordion.

**Before:**
```python
with gr.Accordion("🔍 Reasoning Trace", open=False):
    reasoning_panel = gr.Markdown(...)
# audit panel did not exist
with gr.Accordion("👍👎 Analyst Feedback Summary", open=False):
    fb_summary = gr.Markdown(...)
```

**After:**
```python
with gr.Accordion("🔍 Reasoning Trace", open=False):
    reasoning_panel = gr.Markdown(...)

# NEW ACCORDION
with gr.Accordion("🗂️ Audit Trail & Validation", open=False):
    audit_panel = gr.Markdown(value="*Run a query to see the audit trail.*", ...)

with gr.Accordion("👍👎 Analyst Feedback Summary", open=False):
    fb_summary = gr.Markdown(...)
```

---

### Change 6 — Update `_outputs` and `process_query` return signature

Because we added a 4th output (`audit_panel`), we needed to update both the output list and the function's return statement.

**Before (3 panels + run_id = 4 outputs):**
```python
_outputs = [chatbot, eval_panel, reasoning_panel, run_id_state]

def process_query(...) -> tuple:
    ...
    return new_history, eval_panel_md, reasoning_md, run_id
```

**After (4 panels + run_id = 5 outputs):**
```python
_outputs = [chatbot, eval_panel, reasoning_panel, audit_panel, run_id_state]

def process_query(...) -> tuple:
    ...
    return new_history, eval_panel_md, reasoning_md, audit_md, run_id
```

**Why must these match exactly?** Gradio maps return values to outputs positionally. If you return 5 values but only list 4 outputs, Gradio raises a `ValueError` at runtime. Every early-return path inside `process_query` was also updated to return 5 values (using `""` as the empty placeholder for the new audit panel).

---

### Change 7 — Fix chat bubble formatting (remove `###` headings)

**Where:** `_format_codes_response()`.

**Before:**
```python
lines += [f"### 📋 Code Recommendations ({len(items)} found)\n", "---"]
```

**After:**
```python
lines += [f"**📋 Code Recommendations ({len(items)} found)**\n", "---"]
```

**Why `###` was wrong here:**

Markdown heading levels (`#`, `##`, `###`) in a Gradio chatbot bubble render using the browser's default heading styles. `###` (H3) typically renders at around 1.25×–1.5× the base font size. The rest of the bubble content is set to `font-size: 1rem` by the CSS rule:

```css
.chatbot .message.bot > div {
    font-size: 1rem !important;
    font-family: Arial, sans-serif;  /* inherited from body */
}
```

So the `###` heading would appear noticeably larger than every other line, breaking visual consistency. It also has a heavier default font weight from the browser's UA stylesheet, which compounds the size difference.

**Using `**bold**` instead** means the text stays at exactly `1rem` — same size, same font, same family as everything else — but with `font-weight: bold`. The visual hierarchy is preserved (the section title looks more important) without the distracting size jump.

**The rule to remember:** Inside a Gradio chat bubble, use `**bold**` for titles. Reserve `###` for standalone Markdown panels (like the RAGAS eval panel) where heading size is not a concern because there is no surrounding 1rem body text for the heading to clash with.

---

## 5. What `app_audit.py` Now Provides (accessible through the UI)

After these changes, every query through the app now:

| Feature | How it works |
|---------|-------------|
| **Run logging** | Each query produces a `outputs/run_logs/{run_id}.json` file containing the full run record |
| **Code provenance** | Every returned code is recorded with its score, rank, source type, and explanation |
| **Validation flags** | Three automatic checks run after every query: missing explanations, possible hallucinations, confidence mismatches |
| **UI visibility** | The "🗂️ Audit Trail & Validation" accordion shows flags and the top-5 code provenance table inline |
| **Backtesting** | `app_audit.run_backtest()` can be called from the command line against NICE gold-standard files |
| **Run comparison** | `app_audit.compare_runs(path_a, path_b)` diffs two saved run JSONs |

### How to trigger a backtest from the command line

```bash
python -c "
from app import process_query          # the live chat function
from app_audit import run_backtest
run_backtest(
    chat_fn=process_query,
    gold_standard_dir='data/gold_standard/',
    model_name='llama3.2:1b',
    ranking_model='ChromaDB Hybrid',
    output_dir='outputs/backtest/',
)
"
```

---

## 6. Reasoning Engine Logic — How the Three Eval Modules Complement Each Other

Think of the three panels as three different perspectives on the same output:

```
User query
    │
    ▼
pipeline.retrieve_and_rank()   ← finds codes from ChromaDB
    │
    ├──► reasoning_eval.generate_reasoning_trace()
    │       "How did we get here?"
    │       Explains the mechanics: decomposition → embedding →
    │       cross-encoder → hybrid scoring → LLM explanation.
    │       Audience: clinical analyst who wants to understand the process.
    │
    ├──► ragas_eval.evaluate()
    │       "How good is this result?"
    │       Measures faithfulness, relevancy, and coverage.
    │       Audience: quality assurance / governance team.
    │
    └──► app_audit.finish_run() + save()
            "What happened, and are there any problems?"
            Records provenance and flags anomalies.
            Audience: clinical coder review and audit trail.
```

Each module answers a different question. Together they provide a complete picture of not just *what* the system returned, but *why*, *how good it is*, and *whether anything looks wrong*.

---

## 7. Data Flow Summary (post-changes)

```
app.py: process_query(message, history, model_choice)
    │
    ├─ [1] _audit.start_run(message, model_choice, "ChromaDB Hybrid")
    │       → creates RunRecord in memory, returns run_id
    │
    ├─ [2] pipeline.retrieve_and_rank(query, model_choice, top_k=10)
    │       → returns report dict {items, sub_queries, primary_condition, ...}
    │
    ├─ [3] pipeline.add_llm_explanations(report, message)
    │       → calls llm.call_llm() to add explanation strings to each item
    │       → NOTE: uses pipeline.py's internal prompt, NOT prompts.md
    │
    ├─ [4] _format_codes_response(report, model_choice)
    │       → builds Markdown string for the chat bubble
    │       → uses **bold** not ### for section titles (formatting fix)
    │
    ├─ [5] _audit.finish_run(run_id, report["items"])
    │       → records CodeProvenance for each item
    │       → runs validation checks (3 types)
    │
    ├─ [6] _audit.save(run_id)
    │       → writes outputs/run_logs/{run_id}.json
    │
    ├─ [7] _format_audit_panel(run_id)   ← NEW
    │       → reads RunRecord, formats as Markdown for UI
    │
    ├─ [8] ragas_eval.evaluate(report, message)   ← was already here
    │       → faithfulness, answer_relevancy, context_recall
    │
    └─ [9] reasoning_eval.generate_reasoning_trace(report, message)   ← was already here
            → step-by-step Markdown trace
```

---

## 8. Files Changed and Their Status

| File | Status | What changed |
|------|--------|-------------|
| `app.py` | ✅ **Modified** | Added `import app_audit`, `_audit` singleton, `_format_audit_panel()`, audit panel accordion, updated `_outputs` list and `process_query` return, fixed `###` → `**bold**` in `_format_codes_response` |
| `app_audit.py` | ✅ **Unchanged** | No modifications needed — the existing API (`start_run`, `finish_run`, `save`) was sufficient |
| `ragas_eval.py` | ✅ **Unchanged** | Already integrated |
| `reasoning_eval.py` | ✅ **Unchanged** | Already integrated |
| `feedback_hitl.py` | ✅ **Unchanged** | Already integrated |
| `llm.py` | ✅ **Unchanged** | Works as-is |
| `service.py` | ⚠️ **Not changed** | Only used by `demo.py`; see Section 2 for recommended future change |
| `prompts.md` | ⚠️ **Not changed** | Correct content, but only reaches the LLM via `demo.py`; see Section 2 |
| `demo.py` | ✅ **Unchanged** | Standalone test script; unrelated to app.py |

---

*Report generated for NICE Clinical Code Assistant — Group 5, NICE Employer Project 2025.*
