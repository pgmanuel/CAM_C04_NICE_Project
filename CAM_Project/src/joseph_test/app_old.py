"""
app.py
------
Local chatbot frontend using Gradio.

Run with:
    python app.py

Then open http://localhost:7860 in your browser.
"""

from __future__ import annotations

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

# Relative path: Goes up 3 levels from app.py (src/justification/app.py -> CAM_Project) 
# then into the data folder. This works on any machine.
CSV_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "combined_normalized_codes.csv"

# Global cache so we only load the CSV once
_cached_codes = []

def _load_codes_from_csv():
    """Load clinical codes from the CSV file into memory."""
    global _cached_codes
    if _cached_codes:
        return  # Already loaded

    if not CSV_DATA_PATH.exists():
        print(f"⚠️ [warning] Could not find CSV at {CSV_DATA_PATH.resolve()}. Using fallback dict.")
        return

    try:
        with open(CSV_DATA_PATH, mode="r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []

            # Auto-detect the code and term columns based on common names
            code_col = next((h for h in headers if h.lower() in ["code", "concept_id", "medcodeid", "id"]), headers[0] if headers else None)
            term_col = next((h for h in headers if h.lower() in ["term", "description", "name", "label"]), headers[1] if len(headers) > 1 else None)

            if not code_col or not term_col:
                print("⚠️ [warning] Could not detect code/term columns in CSV.")
                return

            for row in reader:
                code = row.get(code_col, "").strip()
                term = row.get(term_col, "").strip()
                if code and term:
                    # Store the lowercase term for faster substring searching later
                    _cached_codes.append({
                        "code": code, 
                        "term": term, 
                        "search_key": term.lower()
                    })

        print(f"[info] Successfully loaded {len(_cached_codes)} clinical codes from CSV.")
    except Exception as e:
        print(f"⚠️ [error] Failed to load CSV: {e}")

def _extract_candidates(text: str) -> list[dict]:
    """
    Pattern-match clinical terms from the CSV to seed the payload.
    """
    # Ensure the CSV data is loaded
    _load_codes_from_csv()

    text_lower = text.lower()
    candidates = []
    seen_codes = set()

    # 1. Search the CSV data if it loaded successfully
    if _cached_codes:
        for item in _cached_codes:
            # If the CSV term appears in the user's message
            if item["search_key"] in text_lower and item["code"] not in seen_codes:
                candidates.append({
                    "MedCodeId": item["code"],
                    "term": item["term"],
                    "status": "candidate",
                    "score": 0.8,
                })
                seen_codes.add(item["code"])
    
    # 2. Fallback to the original hardcoded list if the CSV is missing/empty
    else:
        known_conditions = {
            "type 2 diabetes":    ("44054006",  "Type 2 diabetes mellitus"),
            "t2dm":               ("44054006",  "Type 2 diabetes mellitus"),
            "diabetes":           ("73211009",  "Diabetes mellitus"),
            "hypertension":       ("38341003",  "Hypertension"),
            "high blood pressure":("38341003",  "Hypertension"),
            "obesity":            ("414916001", "Obesity (BMI 30+)"),
            "obese":              ("414916001", "Obesity (BMI 30+)"),
            "ckd":                ("709044004", "Chronic kidney disease"),
            "chronic kidney":     ("709044004", "Chronic kidney disease"),
            "dyslipidaemia":      ("55822004",  "Hyperlipidaemia"),
            "cholesterol":        ("55822004",  "Hyperlipidaemia"),
            "sleep apnoea":       ("73430006",  "Sleep apnoea syndrome"),
            "osa":                ("73430006",  "Sleep apnoea syndrome"),
        }
        for phrase, (code, term) in known_conditions.items():
            if phrase in text_lower and code not in seen_codes:
                candidates.append({
                    "MedCodeId": code,
                    "term": term,
                    "status": "candidate",
                    "score": 0.8,
                })
                seen_codes.add(code)

    # 3. If nothing matched at all, return an unknown placeholder
    if not candidates:
        candidates.append({
            "MedCodeId": "UNKNOWN",
            "term": text[:100],
            "status": "unrecognised",
            "score": 0.5,
        })

    return candidates


def _build_payload_from_conversation(
    user_message: str,
    history: list[tuple[str, str]],
) -> dict:
    """
    Convert a plain-English chat message into the structured payload
    that service.generate_report() expects.
    """
    # ✨ This is where _extract_candidates is called ✨
    candidates = _extract_candidates(user_message)

    conversation_context = ""
    if history:
        prior = "\n".join(
            f"User: {h[0]}\nAssistant: {h[1]}" for h in history[-3:]
        )
        conversation_context = f"Prior conversation:\n{prior}\n\n"

    from datetime import datetime, timezone
    return {
        "query": user_message,
        "metadata": {
            "source": "chatbot_session",
            "reviewer": "local_dev",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence_threshold": 0.75,
            "conversation_turns": len(history),
        },
        "candidates": candidates,
        "notes": conversation_context + f"User query: {user_message}",
    }


# ─────────────────────────────────────────────────────────────────
# STEP 2: Format JSON report as readable Markdown for the chat window
# ─────────────────────────────────────────────────────────────────

def _format_response(report: dict) -> str:
    """Convert the JSON report into readable Markdown text."""
    lines = []

    items = report.get("items", [])
    if not items:
        return "No items were returned. Try rephrasing your question."

    lines.append(f"### 📋 Code Recommendations ({len(items)} found)\n")

    for item in items:
        flag = item.get("flag", "UNCLASSIFIED")
        icons = {
            "CANDIDATE_INCLUDE": "✅",
            "REVIEW":            "⚠️",
            "STRATIFIER":        "🔀",
            "UNCLASSIFIED":      "❓",
        }
        icon = icons.get(flag, "❓")

        code = item.get("code", "N/A")
        term = item.get("term", "N/A")
        explanation = item.get("explanation", "No explanation provided.")
        priority = item.get("priority")
        evidence_list = item.get("evidence", [])

        lines.append(f"{icon} **{term}** `{code}`")
        lines.append(f"*Flag:* `{flag}`")
        if priority:
            lines.append(f"*Priority:* {priority}")
        lines.append(f"\n{explanation}\n")

        if evidence_list:
            lines.append("**Evidence:**")
            for ev in evidence_list:
                source = ev.get("source", "unknown")
                text   = ev.get("text", "")
                lines.append(f"- *{source}:* {text}")

        lines.append("---")

    synth = report.get("synthetic_suggestion", {})
    if synth and synth.get("synthetic_code"):
        lines.append(f"\n### 🔬 Composite Suggestion: `{synth['synthetic_code']}`")
        lines.append(f"*Confidence:* {synth.get('confidence','?')} | *Authoritative:* No")
        lines.append(f"\n{synth.get('rationale','')}")
        if synth.get("warning"):
            lines.append(f"\n⚠️ {synth['warning']}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# STEP 3: Main chat handler — Gradio calls this on every message
# ─────────────────────────────────────────────────────────────────

def chat(message: str, history: list[tuple[str, str]]) -> str:
    """Main chat handler called by Gradio on every submitted message."""

    if not message.strip():
        return "Please enter a clinical query to get started."

    if message.strip().lower() in {"hi", "hello", "hey", "help", "?"}:
        return (
            "Hi! I'm your NICE clinical code assistant.\n\n"
            "You can ask me things like:\n"
            "- *\"Build a code list for obesity with type 2 diabetes\"*\n"
            "- *\"What codes cover hypertension and CKD?\"*\n"
            "- *\"Suggest codes for a patient with T2DM and high cholesterol\"*\n\n"
            "I'll return a structured list with confidence flags and evidence."
        )

    try:
        payload = _build_payload_from_conversation(message, history)
        report  = generate_report(payload)
        return _format_response(report)

    except RuntimeError as e:
        if "No LLM provider" in str(e) or "No API key" in str(e):
            return (
                "⚠️ **No LLM provider found.**\n\n"
                "Check your `.env` file contains both of these lines:\n"
                "```\n"
                "OLLAMA_BASE_URL=http://localhost:11434/v1\n"
                "LLM_MODEL=llama3.2:1b\n"
                "```\n\n"
                "Also make sure Ollama is actually running "
                "(open a separate terminal and run `ollama serve`).\n\n"
                f"Detail: {e}"
            )
        return f"⚠️ **Runtime error:** {e}"

    except json.JSONDecodeError as e:
        return (
            "⚠️ **Model returned invalid JSON.**\n\n"
            "Common with smaller local models. Try:\n"
            "1. Set `USE_JSON_RESPONSE_FORMAT=false` in your `.env`\n"
            "2. Restart the app\n\n"
            f"Detail: {e}"
        )

    except Exception as e:
        return f"⚠️ **Unexpected error:** `{type(e).__name__}: {e}`"


# ─────────────────────────────────────────────────────────────────
# STEP 4: Build and launch the Gradio interface
# ─────────────────────────────────────────────────────────────────

def _build_interface() -> gr.ChatInterface:
    """Build the Gradio ChatInterface."""

    model_name = os.getenv("LLM_MODEL", "not configured")
    provider = (
        "Ollama (local)"     if os.getenv("OLLAMA_BASE_URL")    else
        "OpenRouter (cloud)" if os.getenv("OPENROUTER_API_KEY") else
        "OpenAI (cloud)"     if os.getenv("OPENAI_API_KEY")     else
        "⚠️ No provider set — check .env"
    )

    return gr.ChatInterface(
        fn=chat,
        title="🏥 NICE Clinical Code Assistant",
        description=(
            f"**Model:** `{model_name}` &nbsp;|&nbsp; "
            f"**Provider:** {provider}\n\n"
            "Ask a clinical question in plain English. "
            "The assistant will suggest relevant SNOMED codes with "
            "confidence flags and supporting evidence.\n\n"
            "*Example: 'Build a code list for obesity with type 2 diabetes'*"
        ),
        examples=[
            "Build a code list for obesity with type 2 diabetes",
            "What codes cover hypertension and chronic kidney disease?",
            "Suggest codes for a patient with T2DM, hypertension and dyslipidaemia",
            "Codes for obese patients with sleep apnoea",
        ],
        textbox=gr.Textbox(
            placeholder="e.g. 'Codes for obesity with type 2 diabetes and hypertension'",
            container=False,
        ),
    )


# ─────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NICE Clinical Code Assistant — Local Chatbot")
    print("="*60)

    model = os.getenv("LLM_MODEL", "NOT SET")
    provider_check = {
        "OLLAMA_BASE_URL":    os.getenv("OLLAMA_BASE_URL"),
        "OPENROUTER_API_KEY": os.getenv("OPENROUTER_API_KEY"),
        "OPENAI_API_KEY":     os.getenv("OPENAI_API_KEY"),
    }
    active = next((k for k, v in provider_check.items() if v), None)

    print(f"Model:    {model}")
    print(f"Provider: {active or 'NONE'}")

    if not active:
        print()
        print("⚠️  TROUBLESHOOTING — Provider is NONE after load_dotenv()")
        print(f"   .env path: {_env_path}")
        print(f"   File exists: {_env_path.exists()}")
        if _env_path.exists():
            raw = _env_path.read_bytes()[:6]
            has_bom = raw[:3] == b'\xef\xbb\xbf'
            print(f"   BOM detected: {has_bom}")
            print("   (BOM is handled by encoding='utf-8-sig' — should not be the issue now)")
            print("   First 60 chars of .env:")
            print("  ", _env_path.read_text(encoding="utf-8-sig")[:60])

    print("\nOpen in browser: http://localhost:7860")
    print("Stop with: Ctrl+C")
    print("="*60 + "\n")

    interface = _build_interface()
    interface.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True,
    )
