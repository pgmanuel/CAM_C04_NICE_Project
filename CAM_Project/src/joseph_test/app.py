"""
app.py
------
Local chatbot frontend using Gradio.

Run with:
    python app.py

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

import base64
import json
import os
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path
import csv
import gradio as gr
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────
# LOAD ENVIRONMENT VARIABLES — ONE call only, with two fixes applied.
#
# FIX 1 — encoding='utf-8-sig'
#   Windows Notepad and VS Code (on Windows) often save files with a
#   BOM (Byte Order Mark): three invisible bytes at the start of the
#   file. When Python reads the .env as plain utf-8, the BOM sticks
#   to the first variable name, turning:
#       OLLAMA_BASE_URL=...
#   into:
#       ﻿OLLAMA_BASE_URL=...   (with invisible char at front)
#   So os.getenv('OLLAMA_BASE_URL') returns None — the key was stored
#   under a different name. 'utf-8-sig' strips the BOM automatically.
#   This was the root cause of Provider showing as NONE.
#
# FIX 2 — single call, explicit path
#   The previous version called load_dotenv() twice: once with a path
#   and once without. The second no-argument call searched from
#   PowerShell's working directory (not necessarily the script folder).
#   One call with an explicit path is always unambiguous.
#
# FIX 3 — override=True
#   If OLLAMA_BASE_URL was set as a system env var in a prior session,
#   load_dotenv() without override=True silently skips it. override=True
#   makes your .env file always take precedence during development.
# ─────────────────────────────────────────────────────────────────

_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, encoding="utf-8-sig", override=True)

# Diagnostic prints — confirm the env loaded before Gradio starts.
# Comment these out once everything is working.
print(f"[env] .env path:       {_env_path}")
print(f"[env] OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL')}")
print(f"[env] LLM_MODEL:       {os.getenv('LLM_MODEL')}")

# Import service AFTER load_dotenv so llm.py sees the env vars.
from service import generate_report


# ─────────────────────────────────────────────────────────────────
# STEP 1: Convert chat input to the structured payload service.py expects
# ─────────────────────────────────────────────────────────────────

# Global cache so we only load the CSV once
CSV_FILENAME = "combined_normalized_codes.csv"
_cached_codes: list[dict] = []
_csv_loaded = False


def _resolve_csv_path() -> Path | None:
    """Find the CSV in the project tree or nearby working directories."""
    script_dir = Path(__file__).resolve().parent
    project_data_dir = script_dir.parent.parent.parent / "data"

    candidates = [
        project_data_dir / CSV_FILENAME,
        script_dir / CSV_FILENAME,
        script_dir.parent / CSV_FILENAME,
        Path.cwd() / "data" / CSV_FILENAME,
        Path.cwd() / CSV_FILENAME,
    ]

    print("[csv] path search:")
    for candidate in candidates:
        print(f"[csv]   trying: {candidate}")
        if candidate.exists():
            print(f"[csv]   found:  {candidate}")
            return candidate

    return None


def _normalize_text(value: str) -> str:
    """Normalize text for matching."""
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _pick_column(headers: list[str], candidates: list[str]) -> str | None:
    """Return the first matching header name from a list of candidates."""
    header_map = {h.lower().strip(): h for h in headers}
    for candidate in candidates:
        if candidate in header_map:
            return header_map[candidate]
    return None


def _load_codes_from_csv() -> None:
    """Load clinical codes from the CSV file into memory."""
    global _cached_codes, _csv_loaded

    if _csv_loaded:
        return

    _csv_loaded = True

    csv_path = _resolve_csv_path()
    if not csv_path:
        print("[csv] ERROR: file does not exist in any known location")
        return

    print(f"[csv] loading file: {csv_path.resolve()}")

    try:
        with open(csv_path, mode="r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            print(f"[csv] headers ({len(headers)}): {headers}")

            code_col = _pick_column(headers, ["code", "medcodeid", "snomedctconceptid", "cleansedreadcode", "originalreadcode"])
            term_col = _pick_column(headers, ["term", "description", "name", "label", "medicine", "drugsubstancename"])

            if not code_col or not term_col:
                print("[csv] ERROR: could not detect usable code/term columns")
                return

            rows = 0
            for row in reader:
                rows += 1

                code_value = (
                    str(row.get(code_col, "")).strip()
                    or str(row.get("Code", "")).strip()
                    or str(row.get("SnomedCTConceptId", "")).strip()
                    or str(row.get("CleansedReadCode", "")).strip()
                    or str(row.get("OriginalReadCode", "")).strip()
                )
                term_value = (
                    str(row.get(term_col, "")).strip()
                    or str(row.get("Term", "")).strip()
                    or str(row.get("medicine", "")).strip()
                    or str(row.get("drugsubstancename", "")).strip()
                )

                if code_value and term_value:
                    _cached_codes.append(
                        {
                            "code": code_value,
                            "term": term_value,
                            "search_key": _normalize_text(term_value),
                            "row": row,
                        }
                    )

            print(f"[csv] rows read: {rows}")
            print(f"[csv] records cached: {len(_cached_codes)}")
            if _cached_codes:
                print(f"[csv] sample[0]: code={_cached_codes[0]['code']} | term={_cached_codes[0]['term']}")
                if len(_cached_codes) > 1:
                    print(f"[csv] sample[1]: code={_cached_codes[1]['code']} | term={_cached_codes[1]['term']}")
            else:
                print("[csv] ERROR: no usable records were cached from the CSV")

    except Exception as e:
        print(f"[csv] ERROR: failed to load CSV: {type(e).__name__}: {e}")


def _extract_candidates(text: str) -> list[dict]:
    """Match clinical terms from the CSV to seed the payload."""
    _load_codes_from_csv()

    text_norm = _normalize_text(text)
    candidates = []
    seen_codes = set()

    if _cached_codes:
        for item in _cached_codes:
            term_norm = item["search_key"]
            if not term_norm:
                continue

            if term_norm == text_norm or term_norm in text_norm or text_norm in term_norm:
                if item["code"] not in seen_codes:
                    candidates.append(
                        {
                            "MedCodeId": item["code"],
                            "term": item["term"],
                            "status": "candidate",
                            "score": 0.95 if term_norm == text_norm else 0.8,
                        }
                    )
                    seen_codes.add(item["code"])

        if not candidates:
            text_tokens = set(text_norm.split())
            for item in _cached_codes:
                term_tokens = set(item["search_key"].split())
                if len(term_tokens) < 2:
                    continue

                overlap = len(text_tokens & term_tokens)
                if overlap >= 2 or (overlap >= 1 and len(term_tokens) <= 4):
                    if item["code"] not in seen_codes:
                        candidates.append(
                            {
                                "MedCodeId": item["code"],
                                "term": item["term"],
                                "status": "candidate",
                                "score": 0.7,
                            }
                        )
                        seen_codes.add(item["code"])
                        if len(candidates) >= 10:
                            break

    if not candidates:
        candidates.append(
            {
                "MedCodeId": "UNKNOWN",
                "term": text[:100],
                "status": "unrecognised",
                "score": 0.0,
            }
        )

    return candidates


# def _build_payload_from_conversation(
#     user_message: str,
#     history: list[tuple[str, str]],
# ) -> dict:
#     """Convert a plain-English chat message into the structured payload that service.generate_report() expects."""
#     candidates = _extract_candidates(user_message)

#     conversation_context = ""
#     if history:
#         prior = "\n".join(
#             f"User: {h[0]}\nAssistant: {h[1]}" for h in history[-3:]
#         )
#         conversation_context = f"Prior conversation:\n{prior}\n\n"

#     return {
#         "query": user_message,
#         "metadata": {
#             "source": "chatbot_session",
#             "reviewer": "local_dev",
#             "timestamp": datetime.now(timezone.utc).isoformat(),
#             "confidence_threshold": 0.75,
#             "conversation_turns": len(history),
#         },
#         "candidates": candidates,
#         "notes": conversation_context + f"User query: {user_message}",
#     }

# ─────────────────────────────────────────────────────────────────
# STEP 2: Build structured report DIRECTLY from CSV candidates
# ─────────────────────────────────────────────────────────────────

def _build_report_from_candidates(candidates: list[dict]) -> dict:
    """Convert extracted candidates into structured report items."""
    
    items = []

    for i, c in enumerate(candidates):
        if c["MedCodeId"] == "UNKNOWN":
            continue

        items.append({
            "code": c["MedCodeId"],
            "term": c["term"],
            "flag": "CANDIDATE_INCLUDE",
            "priority": i + 1,
            "explanation": "",  # to be filled optionally by LLM
            "evidence": [],
        })

    return {"items": items}


# ─────────────────────────────────────────────────────────────────
# STEP 3: Optional LLM explanation (SAFE — no code generation)
# ─────────────────────────────────────────────────────────────────

def _add_llm_explanations(report: dict, query: str) -> dict:
    """Use LLM ONLY to explain codes — never to generate them."""

    try:
        from llm import call_llm

        codes_text = "\n".join(
            f"- {item['term']} ({item['code']})"
            for item in report["items"]
        )

        system_prompt = (
            "You are a clinical coding assistant.\n"
            "Explain why each code is relevant to the query.\n"
            "DO NOT add new codes.\n"
            "Return JSON list with: code, explanation."
        )

        user_prompt = f"""
Query:
{query}

Codes:
{codes_text}
"""

        response = call_llm(system_prompt, user_prompt)

        parsed = json.loads(response)

        explanation_map = {
            item["code"]: item["explanation"]
            for item in parsed
        }

        for item in report["items"]:
            item["explanation"] = explanation_map.get(
                item["code"],
                "Relevant based on matched clinical term."
            )

    except Exception as e:
        print(f"[warn] LLM explanation failed: {e}")
        for item in report["items"]:
            item["explanation"] = "Matched from clinical terminology dataset."

    return report


# ─────────────────────────────────────────────────────────────────
# STEP 4: Format output (NO synthetic sections)
# ─────────────────────────────────────────────────────────────────

def _format_response(report: dict) -> str:
    """Convert structured report into Markdown."""

    items = report.get("items", [])
    if not items:
        return "No clinical codes matched your query."

    lines = []
    lines.append(f"### 📋 Code Recommendations ({len(items)} found)\n")

    for item in items:
        lines.append(f"✅ **{item['term']}** `{item['code']}`")
        lines.append(f"*Priority:* {item['priority']}")
        lines.append(f"\n{item['explanation']}\n")
        lines.append("---")

    return "\n".join(lines)

# ─────────────────────────────────────────────────────────────────
# STEP 5: Main chat handler (Now with TWO dropdowns)
# ─────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────
# STEP 5: Main chat handler (Accepted TWO dropdowns)
# ─────────────────────────────────────────────────────────────────

def chat(message: str, history: list[tuple[str, str]], model_choice: str, ranking_choice: str) -> str:
    if not message.strip():
        return "Please enter a clinical query."

    # Sync model choice with environment for llm.py
    if model_choice:
        os.environ["LLM_MODEL"] = model_choice
    
    print(f"[debug] AI: {model_choice} | Ranker: {ranking_choice}")

    try:
        candidates = _extract_candidates(message)
        report = _build_report_from_candidates(candidates)
        report = _add_llm_explanations(report, message)
        return _format_response(report)
    except Exception as e:
        return f"⚠️ Error: {type(e).__name__}: {e}"

# ─────────────────────────────────────────────────────────────────
# STEP 6: Build the Professional Interface
# ─────────────────────────────────────────────────────────────────

def _build_interface() -> tuple:
    provider = (
        "Ollama (local)"     if os.getenv("OLLAMA_BASE_URL")    else
        "OpenRouter (cloud)" if os.getenv("OPENROUTER_API_KEY") else
        "OpenAI (cloud)"     if os.getenv("OPENAI_API_KEY")     else
        "⚠️ No provider"
    )

    # 1. Handle the local image path
    current_dir = Path(__file__).parent
    logo_filename = "nhs-logo-880x4951.jpeg"
    logo_path = current_dir / logo_filename

    # 2. Convert image to Base64
    logo_html = ""
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            encoded_string = base64.b64encode(f.read()).decode()
            logo_html = f'<img src="data:image/jpeg;base64,{encoded_string}" class="nhs-logo-img">'
    else:
        logo_html = "<span style='color:red;'>Logo Not Found</span>"

    custom_css = """
    /* Header: Logo Far Right */
    .header-row { 
        display: flex !important; 
        justify-content: space-between !important; 
        align-items: center !important; 
        padding: 10px 25px; 
        background: var(--background-fill-secondary); 
        border-radius: 10px; 
        margin-bottom: 0px !important; /* NO SPACE BELOW HEADER */
        border: 2px solid black; 
    }
    
    .team-name { font-weight: 800; color: #005EB8; margin: 0; }
    
    .logo-container {
        margin-left: auto !important; 
        display: flex;
        justify-content: flex-end;
    }

    .nhs-logo-img { height: 45px; width: auto; }
    
    /* Dropdowns and Chatbot Container */
    #chat-container { 
        border: 2px solid black !important; 
        border-radius: 12px; 
        padding: 20px;
        margin-top: 0px !important; /* NO SPACE ABOVE CHATBOT */
    }

    /* THE ONLY GAP: Between Chatbot and Examples */
    /* Target the container holding the example buttons */
    .examples, .examples-container, .gr-samples {
        margin-top: 250px !important; /* MASSIVE SPACE HERE */
        padding-top: 20px !important;
        border-top: 3px solid black !important;
        display: block !important;
        clear: both !important;
    }

    /* High Contrast Borders */
    .gradio-container .gr-input, 
    .gradio-container .gr-box, 
    .gradio-container .secondary, 
    .gradio-container button.secondary,
    .gradio-container .gr-button,
    .gradio-container .gr-dropdown,
    .gradio-container select,
    .gradio-container textarea,
    .gradio-container .block {
        border: 2px solid black !important;
    }

    /* Grey Textbox Area */
    .gradio-container textarea {
        background-color: #f0f0f0 !important;
        border: 2px solid black !important;
    }

    /* Remove default Gradio gap between blocks */
    .gradio-container .gap {
        gap: 0px !important;
    }
    """

    theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

    with gr.Blocks(title="NICE Clinical Assistant") as demo:
        # 1. Header (Logo on right)
        with gr.Row(elem_classes=["header-row"]):
            gr.HTML(f"<div class='team-name'>Group 5 - NICE Project</div>")
            gr.HTML(f"<div class='logo-container'>{logo_html}</div>")

        # Minimal Markdown - no extra spacing
        gr.Markdown(f"### 🏥 NICE Clinical Code Assistant", elem_id="main-title")

        # 2. Selectors Row - sits right above chat
        with gr.Row():
            with gr.Column(scale=1):
                model_dropdown = gr.Dropdown(
                    choices=["gpt-4o-mini", "gpt-4o", "llama3.2:1b"],
                    value="llama3.2:1b", 
                    label="AI Model",
                    interactive=True,
                )
            with gr.Column(scale=1):
                ranking_dropdown = gr.Dropdown(
                    choices=["Pod 1", "Pod 2"],
                    value="Pod 1",
                    label="Ranking Model",
                    interactive=True,
                )

        # 3. Chat Interface (Grouped to maintain border)
        with gr.Group(elem_id="chat-container"):
            gr.ChatInterface(
                fn=chat, 
                additional_inputs=[model_dropdown, ranking_dropdown],
                examples=[
                    ["obesity with type 2 diabetes", "llama3.2:1b", "Pod 1"],
                    ["hypertension and chronic kidney disease", "llama3.2:1b", "Pod 1"],
                    ["T2DM with high cholesterol", "llama3.2:1b", "Pod 1"],
                ],
                textbox=gr.Textbox(
                    placeholder="Type clinical query here...", 
                    container=False, 
                    scale=7
                ),
            )

    return demo, theme, custom_css

# ─────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────

# Update the launch section at the bottom of your app.py
if __name__ == "__main__":
    print("\n" + "="*60)
    print("NICE Clinical Code Assistant — CSV Driven")
    print("="*60)

    # Unpack the three items now returned by the function
    interface, theme, css = _build_interface()

    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        theme=theme,
        css=css
    )