"""
app.py
------
NICE Clinical Code Assistant — NHS-styled chatbot interface.

Run with:   python app.py
Then open:  http://localhost:7860
"""

from __future__ import annotations

import base64
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# ─── ENV — single BOM-safe load ───────────────────────────────────
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, encoding="utf-8-sig", override=True)

print(f"[env] OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL')}")
print(f"[env] LLM_MODEL:       {os.getenv('LLM_MODEL')}")

from service import generate_report  # after load_dotenv


# ═══════════════════════════════════════════════════════════════════
# SECTION A — RANKING PLUG-INS
#
# These are ready to receive your Pod 1 / Pod 2 ranking code.
# The pipeline already calls them. You only need to replace the
# body of each function when your ranking model is ready.
#
# HOW TO CONNECT (when ready):
#   1. At the top of this file add:
#        from pod1_ranking import rank_candidates as _pod1_ranker
#   2. Inside _rank_pod1() replace the body with:
#        return _pod1_ranker(candidates, query)
# ═══════════════════════════════════════════════════════════════════

def _rank_pod1(candidates: list[dict], query: str) -> list[dict]:
    """
    PLUG-IN: Pod 1 Ranking Model.
    Status: placeholder — returns candidates in original order.

    Inputs:
        candidates  list[dict]  each dict has MedCodeId, term, score (0-1)
        query       str         the user's plain-English query

    Output:
        list[dict]  same items, each now with:
            rank              int    1 = most relevant
            confidence_score  float  0-1
            ranked_by         str    label shown in UI

    ─── REPLACE BODY BELOW WHEN READY ────────────────────────────
    """
    return [
        {**c, "rank": i + 1,
         "confidence_score": round(c.get("score", 0.8), 3),
         "ranked_by": "pod1_placeholder"}
        for i, c in enumerate(candidates)
    ]


def _rank_pod2(candidates: list[dict], query: str) -> list[dict]:
    """
    PLUG-IN: Pod 2 Ranking Model.
    Status: placeholder — returns candidates in original order.
    Inputs/outputs identical to _rank_pod1.

    ─── REPLACE BODY BELOW WHEN READY ────────────────────────────
    """
    return [
        {**c, "rank": i + 1,
         "confidence_score": round(c.get("score", 0.8), 3),
         "ranked_by": "pod2_placeholder"}
        for i, c in enumerate(candidates)
    ]


def _apply_ranking(candidates: list[dict], query: str, choice: str) -> list[dict]:
    """Route to the right ranker. Add new elif blocks for Pod 3, Pod 4 etc."""
    if choice == "Pod 2":
        return _rank_pod2(candidates, query)
    return _rank_pod1(candidates, query)  # default


# ═══════════════════════════════════════════════════════════════════
# SECTION B — MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════

_PROVIDER_MODELS = {
    "ollama":     ["llama3.2:1b", "llama3.2", "mistral", "phi3:mini"],
    "openai":     ["gpt-5-mini", "gpt-4o", "gpt-5.4"],
    "openrouter": [
        "openai/gpt-4o-mini", "openai/gpt-4o",
        "anthropic/claude-3-haiku", "mistralai/mistral-7b-instruct",
    ],
}


def _get_available_models() -> list[str]:
    """Return models for all configured providers, in priority order."""
    m: list[str] = []
    if os.getenv("OLLAMA_BASE_URL"):
        m += _PROVIDER_MODELS["ollama"]
    if os.getenv("OPENROUTER_API_KEY"):
        m += _PROVIDER_MODELS["openrouter"]
    if os.getenv("OPENAI_API_KEY"):
        m += _PROVIDER_MODELS["openai"]
    return m or ["no-provider-configured"]


def _get_default_model() -> str:
    env_model = os.getenv("LLM_MODEL", "")
    available = _get_available_models()
    return env_model if env_model in available else (available[0] if available else "")


def _apply_model_choice(model_choice: str) -> None:
    """Write chosen model into os.environ so llm.py picks it up at call time."""
    if model_choice and model_choice != "no-provider-configured":
        os.environ["LLM_MODEL"] = model_choice


# ═══════════════════════════════════════════════════════════════════
# SECTION C — CSV CODE LOADING  (unchanged logic)
# ═══════════════════════════════════════════════════════════════════

CSV_FILENAME  = "combined_normalized_codes.csv"
_cached_codes: list[dict] = []
_csv_loaded   = False


def _resolve_csv_path() -> Path | None:
    script_dir = Path(__file__).resolve().parent
    for p in [
        script_dir.parent.parent.parent / "data" / CSV_FILENAME,
        script_dir / CSV_FILENAME,
        script_dir.parent / CSV_FILENAME,
        Path.cwd() / "data" / CSV_FILENAME,
        Path.cwd() / CSV_FILENAME,
    ]:
        if p.exists():
            print(f"[csv] found: {p}")
            return p
    return None


def _normalize_text(v: str) -> str:
    v = re.sub(r"[^a-z0-9]+", " ", v.lower().strip())
    return re.sub(r"\s+", " ", v).strip()


def _pick_column(headers: list[str], candidates: list[str]) -> str | None:
    m = {h.lower().strip(): h for h in headers}
    for c in candidates:
        if c in m:
            return m[c]
    return None


def _load_codes_from_csv() -> None:
    global _cached_codes, _csv_loaded
    if _csv_loaded:
        return
    _csv_loaded = True
    path = _resolve_csv_path()
    if not path:
        print("[csv] ERROR: file not found")
        return
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader  = csv.DictReader(f)
            headers = reader.fieldnames or []
            code_col = _pick_column(headers, ["code","medcodeid","snomedctconceptid","cleansedreadcode"])
            term_col = _pick_column(headers, ["term","description","name","label","medicine","drugsubstancename"])
            if not code_col or not term_col:
                print("[csv] ERROR: usable columns not found")
                return
            for row in reader:
                c = str(row.get(code_col,"")).strip()
                t = str(row.get(term_col,"")).strip()
                if c and t:
                    _cached_codes.append({"code":c,"term":t,"search_key":_normalize_text(t),"row":row})
        print(f"[csv] loaded {len(_cached_codes)} records")
    except Exception as e:
        print(f"[csv] ERROR: {type(e).__name__}: {e}")


def _extract_candidates(text: str) -> list[dict]:
    _load_codes_from_csv()
    tn = _normalize_text(text)
    candidates: list[dict] = []
    seen: set[str] = set()

    if _cached_codes:
        for item in _cached_codes:
            sk = item["search_key"]
            if not sk:
                continue
            if sk == tn or sk in tn or tn in sk:
                if item["code"] not in seen:
                    candidates.append({"MedCodeId":item["code"],"term":item["term"],"status":"candidate","score":0.95 if sk==tn else 0.8})
                    seen.add(item["code"])

        if not candidates:
            tt = set(tn.split())
            for item in _cached_codes:
                it = set(item["search_key"].split())
                ov = len(tt & it)
                if (ov>=2 or (ov>=1 and len(it)<=4)) and item["code"] not in seen:
                    candidates.append({"MedCodeId":item["code"],"term":item["term"],"status":"candidate","score":0.7})
                    seen.add(item["code"])
                    if len(candidates) >= 10:
                        break

    if not candidates:
        candidates.append({"MedCodeId":"UNKNOWN","term":text[:100],"status":"unrecognised","score":0.0})
    return candidates


# ═══════════════════════════════════════════════════════════════════
# SECTION D — REPORT BUILDING  (unchanged logic)
# ═══════════════════════════════════════════════════════════════════

def _build_report_from_candidates(candidates: list[dict]) -> dict:
    items = []
    for i, c in enumerate(candidates):
        if c["MedCodeId"] == "UNKNOWN":
            continue
        items.append({"code":c["MedCodeId"],"term":c["term"],"flag":"CANDIDATE_INCLUDE",
                      "priority":i+1,"score":c.get("score",0.8),"explanation":"","evidence":[]})
    return {"items": items}


def _add_llm_explanations(report: dict, query: str) -> dict:
    try:
        from llm import call_llm
        codes_text = "\n".join(f"- {i['term']} ({i['code']})" for i in report["items"])
        system = ("You are a clinical coding assistant.\n"
                  "Explain why each code is relevant to the query.\n"
                  "DO NOT add new codes.\n"
                  "Return JSON list with: code, explanation.")
        user = f"Query:\n{query}\n\nCodes:\n{codes_text}"
        parsed = json.loads(call_llm(system, user))
        em = {p["code"]: p["explanation"] for p in parsed}
        for item in report["items"]:
            item["explanation"] = em.get(item["code"], "Relevant based on matched clinical term.")
    except Exception as e:
        print(f"[warn] LLM explanation failed: {e}")
        for item in report["items"]:
            item["explanation"] = "Matched from clinical terminology dataset."
    return report


# ═══════════════════════════════════════════════════════════════════
# SECTION E — FORMAT OUTPUT
# ═══════════════════════════════════════════════════════════════════

def _format_response(report: dict, model_choice: str, ranking_choice: str) -> str:
    items = report.get("items", [])
    if not items:
        return "No clinical codes matched your query. Try different search terms."

    lines = [
        f"> **AI Model:** `{model_choice}` &nbsp;·&nbsp; **Ranking:** `{ranking_choice}`\n",
        f"### 📋 Code Recommendations ({len(items)} found)\n",
        "---",
    ]
    for item in items:
        rank  = item.get("rank", item.get("priority","—"))
        score = item.get("confidence_score", item.get("score","—"))
        sd    = f"{score:.0%}" if isinstance(score, float) else str(score)
        lines += [
            f"**#{rank} &nbsp; {item['term']}**",
            f"`{item['code']}` &nbsp;·&nbsp; Confidence: **{sd}**",
            f"*Ranked by:* {item.get('ranked_by', ranking_choice)}",
            f"\n{item['explanation']}\n",
            "---",
        ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# SECTION F — CHAT HANDLER
# ═══════════════════════════════════════════════════════════════════

def chat(message: str, history: list[tuple[str,str]], model_choice: str, ranking_choice: str) -> str:
    """
    Main handler — called by Gradio on every message.

    Inputs
    ------
    message        str   the user's query
    history        list  prior (user, assistant) pairs
    model_choice   str   selected from AI Model dropdown
    ranking_choice str   selected from Ranking Model dropdown

    Output
    ------
    str  Markdown displayed in the chat window

    Pipeline
    --------
    message
      → _extract_candidates()           match CSV codes
      → _build_report_from_candidates() structure matches
      → _apply_ranking()                Pod 1 or Pod 2 ranker
      → _add_llm_explanations()         LLM writes reasons
      → _format_response()              render Markdown
    """
    if not message.strip():
        return "Please enter a clinical query to get started."

    if message.strip().lower() in {"hi","hello","hey","help","?"}:
        return (
            "👋 **Welcome to the NICE Clinical Code Assistant.**\n\n"
            "Enter a plain-English clinical query to search the NICE "
            "SNOMED dataset. Results are ranked and explained by AI.\n\n"
            "**Try:**\n- *Obesity with type 2 diabetes*\n"
            "- *Hypertension and chronic kidney disease*"
        )

    _apply_model_choice(model_choice)
    print(f"[chat] model={model_choice} | ranker={ranking_choice} | query='{message[:60]}'")

    try:
        candidates = _extract_candidates(message)
        report     = _build_report_from_candidates(candidates)

        if not report["items"]:
            return ("⚠️ No matching codes found.\n\n"
                    "Try more specific medical terminology, e.g. *Type 2 diabetes mellitus*.")

        # Apply chosen ranker — plug-in functions in Section A
        ranked = _apply_ranking(
            [{"MedCodeId":i["code"],"term":i["term"],"score":i["score"]} for i in report["items"]],
            message, ranking_choice,
        )
        rm = {c["MedCodeId"]: c for c in ranked}
        for item in report["items"]:
            r = rm.get(item["code"], {})
            item["rank"]             = r.get("rank", item["priority"])
            item["confidence_score"] = r.get("confidence_score", item["score"])
            item["ranked_by"]        = r.get("ranked_by", ranking_choice)
        report["items"].sort(key=lambda x: x.get("rank", 999))

        report = _add_llm_explanations(report, message)
        return _format_response(report, model_choice, ranking_choice)

    except RuntimeError as e:
        if "No LLM provider" in str(e) or "No API key" in str(e):
            return (f"⚠️ **LLM provider not found.**\n\nCheck your `.env`:\n"
                    f"```\nOLLAMA_BASE_URL=http://localhost:11434/v1\n"
                    f"LLM_MODEL=llama3.2:1b\n```\n\nDetail: {e}")
        return f"⚠️ **Error:** {e}"
    except json.JSONDecodeError as e:
        return (f"⚠️ **Model returned invalid JSON.**\n\n"
                f"Set `USE_JSON_RESPONSE_FORMAT=false` in `.env` and restart.\n\nDetail: {e}")
    except Exception as e:
        return f"⚠️ **Unexpected error:** `{type(e).__name__}: {e}`"

# ═══════════════════════════════════════════════════════════════════
# SECTION G — NHS INTERFACE (Final Pipe Removal & Style Lockdown)
# ═══════════════════════════════════════════════════════════════════

_C = {
    "blue":      "#005EB8",  # NHS Blue
    "dark_blue": "#003087",  # NHS Dark Blue
    "lt_blue":   "#41B6E6",  # NHS Light Blue
    "white":     "#FFFFFF",  
    "pale":      "#F0F4F5",  
    "border":    "#D8DDE0",  
    "text":      "#212B32",  
}

_CSS = f"""
/* 1. Global Reset & Scaling */
body, .gradio-container {{
    background-color: {_C['pale']} !important;
    font-family: Arial, sans-serif !important;
    margin: 0 !important;
    padding: 0 !important;
}}

.nhs-main-wrapper {{
    max-width: 1280px;
    margin: 20px auto !important;
    background: white !important;
    border: 1px solid {_C['border']} !important;
    border-radius: 4px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
}}

/* 2. Header & Status */
.nhs-header {{ background: {_C['blue']}; padding: 20px 30px; }}
.nhs-header-title {{ font-size: 1.5rem; font-weight: 700; color: white !important; margin-bottom: 2px; }}
.nhs-header-sub {{ font-size: 0.95rem; color: white !important; opacity: 0.9; }}

.nhs-status {{
    background: {_C['dark_blue']};
    padding: 10px 30px;
    font-size: 0.85rem;
    color: white !important;
    display: flex;
    gap: 10px;
}}
.nhs-status span, .nhs-status b {{ color: white !important; }}

/* 3. SELECTORS & CONTENT */
.nhs-content {{ padding: 15px 30px 30px 30px !important; }}

.nhs-selectors {{
    margin-bottom: 10px !important;
    padding-bottom: 5px !important;
    border-bottom: 1px solid {_C['border']} !important;
}}

/* 4. CHAT BUBBLES: Aggressive Artifact Removal */
.chatbot {{
    border: none !important;
    background: #F0F4F5 !important; 
}}

/* THE ATOMIC FIX: Targets Gradio's internal highlight and row backgrounds */
.chatbot .message-row, 
.chatbot .message-row.user,
.chatbot .message-row.bot,
.chatbot .message,
.chatbot .message.user,
.chatbot .message.bot,
.chatbot [data-testid="user-message"] {{
    background-color: transparent !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

.chatbot .message-row:hover,
.chatbot .message-row:focus {{
    background-color: transparent !important;
}}

/* USER BUBBLE: Forced NHS Blue & Logic to prevent word splitting and orange ghosting */
.chatbot .message.user {{
    align-items: flex-end !important; 
    display: flex !important;
    flex-direction: column !important;
}}

.chatbot .message.user > div {{
    background-color: #41B6E6 !important; 
    color: white !important;
    border: none !important;
    border-radius: 12px 12px 2px 12px !important;
    padding: 12px 16px !important;
    font-size: 1rem !important;
    
    /* FIX FOR WORD SPLITTING & WIDTH */
    display: inline-block !important;
    width: auto !important;
    min-width: min-content !important; 
    max-width: 80% !important;
    word-break: normal !important; 
    overflow-wrap: break-word !important;
    white-space: pre-wrap !important; 
    box-shadow: none !important;
}}

.chatbot .message.user > div * {{
    background: transparent !important;
    color: white !important;
}}

.chatbot .message.user > div.selected,
.chatbot .message.user > div:focus {{
    outline: none !important;
    box-shadow: none !important;
    background-color: #41B6E6 !important;
}}

/* LLM BUBBLE */
.chatbot .message.bot > div {{
    background-color: white !important;
    color: #212B32 !important;
    border: 1px solid #D8DDE0 !important;
    border-radius: 12px 12px 12px 2px !important;
    padding: 10px 14px !important;
    font-size: 1rem !important;
}}

/* 5. TEXTBOX: Remove Grey Box and Labels */
.nhs-input {{
    background-color: white !important;
    border: 2px solid black !important;
    border-radius: 0px !important;
}}

/* Targets all potential label/header wrappers in the textbox */
.nhs-input > label, 
.nhs-input .container, 
.nhs-input div[class*="label"],
.nhs-input .form {{
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

.nhs-input textarea {{
    background-color: white !important;
    color: black !important;
    border: none !important;
    box-shadow: none !important;
    font-size: 1rem !important;
}}

/* 6. CLEANUP ARTIFACTS */
.chatbot .message::before, .chatbot .message::after {{
    display: none !important;
}}

.nhs-footer {{
    text-align: center;
    padding: 20px;
    font-size: 0.85rem;
    color: #768692;
    border-top: 1px solid {_C['border']};
}}

.gradio-container .block, .gradio-container .form, .gradio-container .gr-box, .gradio-container .group {{
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}}

/* 7. SCROLLBAR FIX */
.chatbot .message-row {{
    overflow: hidden !important;
}}

.chatbot > div.wrapper {{
    overflow-y: auto !important;
}}

.chatbot {{
    overflow-x: hidden !important;
}}

.chatbot .message {{
    max-width: 100% !important;
    overflow: visible !important;
}}

.nhs-content {{
    overflow: visible !important;
}}

/* 8. DEAD SPACE & FOOTER FIX (Copy buttons, timing, etc) */
.chatbot .message-row .message-footer,
.chatbot .message-row .status,
.chatbot .message-row button,
.chatbot .message-row [class*="p-2"],
.chatbot .message-row [class*="gap-2"] {{
    display: none !important;
    height: 0 !important;
    margin: 0 !important;
    padding: 0 !important;
    visibility: hidden !important;
    opacity: 0 !important;
}}

.chatbot .message.user {{
    margin-bottom: 4px !important;
    padding-bottom: 0 !important;
}}

.chatbot .message-row > div:last-child {{
    margin-bottom: 0 !important;
    padding-bottom: 0 !important;
}}

.chatbot .message-row {{
    min-height: auto !important;
}}
"""

def _logo_html() -> str:
    p = Path(__file__).parent / "nhs-logo-880x4951.jpeg"
    if p.exists():
        enc = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/jpeg;base64,{enc}" style="height:38px; background:white; padding:5px; border-radius:4px;">'
    return '<div style="background:white; color:#005EB8; padding:8px 15px; font-weight:900;">NHS</div>'

def _build_interface() -> gr.Blocks:
    available_models = _get_available_models()
    default_model    = _get_default_model()
    
    with gr.Blocks(title="NICE Clinical Code Assistant", css=_CSS) as demo:
        with gr.Column(elem_classes=["nhs-main-wrapper"]):
            
            # Header Block - Original Style
            gr.HTML(f"""
            <div class="nhs-header">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div>
                        <div class="nhs-header-title">NICE Clinical Code Assistant</div>
                        <div class="nhs-header-sub">Group 5 · NICE Employer Project 2025</div>
                    </div>
                    {_logo_html()}
                </div>
            </div>
            <div class="nhs-status">
                <span><b>Provider:</b> NICE Infrastructure</span> &nbsp;|&nbsp; 
                <span><b>Data:</b> SNOMED CT UK (2025)</span>
            </div>
            """)

            with gr.Column(elem_classes=["nhs-content"]):
                # Compact selectors
                with gr.Row(elem_classes=["nhs-selectors"]):
                    model_dd = gr.Dropdown(choices=available_models, value=default_model, label="AI Model")
                    rank_dd = gr.Dropdown(choices=["Pod 1", "Pod 2"], value="Pod 1", label="Clinical Ranking")
                
                gr.ChatInterface(
                    fn=chat,
                    additional_inputs=[model_dd, rank_dd],
                    examples=[
                        ["obesity with type 2 diabetes", "llama3.2:1b", "Pod 1"],
                        ["hypertension and chronic kidney disease", "llama3.2:1b", "Pod 1"]
                    ],
                    textbox=gr.Textbox(
                        placeholder="Search clinical terms or enter patient symptoms...",
                        container=False, 
                        elem_classes=["nhs-input"]
                    ),
                    chatbot=gr.Chatbot(height=480, elem_classes=["chatbot"])
                )

            gr.HTML("""<div class="nhs-footer">NICE Employer Project 2025 · Verified Data</div>""")

    return demo
    
# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NICE Clinical Code Assistant")
    print("="*60)
    print(f"Model:    {os.getenv('LLM_MODEL','NOT SET')}")
    print(f"Provider: {os.getenv('OLLAMA_BASE_URL') or os.getenv('OPENROUTER_API_KEY') or os.getenv('OPENAI_API_KEY') or 'NONE'}")
    print(f"Dropdown: {_get_available_models()}")
    print("\nhttp://localhost:7860  |  Ctrl+C to stop")
    print("="*60 + "\n")

    demo = _build_interface()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, show_error=True)
