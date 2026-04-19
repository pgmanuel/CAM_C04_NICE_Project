"""
app.py
------
NICE Clinical Code Assistant — Final Production UI.

This is the main file to run. It imports from six separate modules:
  - pipeline.py         retrieval + hybrid ranking engine
  - ragas_eval.py       evaluation metrics (faithfulness, relevancy, recall)
  - reasoning_eval.py   step-by-step reasoning trace
  - feedback_hitl.py    human-in-the-loop thumbs up/down feedback
  - app_audit.py        audit trail, run logging, validation flags
  - cluster_analysis.py t-SNE / KMeans / completeness visualisations  ← NEW

Architecture: this file owns the Gradio UI only. All clinical
logic, evaluation, and data storage live in their own modules.

Run with:   python app.py
Then open:  http://localhost:7860
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

# ─── ENV — BOM-safe load ──────────────────────────────────────────
_env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=_env_path, encoding="utf-8-sig", override=True)

print(f"[env] OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL')}")
print(f"[env] LLM_MODEL:       {os.getenv('LLM_MODEL')}")

import pipeline
import ragas_eval
import reasoning_eval
import feedback_hitl
import app_audit
import cluster_analysis  # ← NEW: t-SNE / KMeans / heatmap engine

# ─── Audit logger singleton ──────────────────────────────────────
_audit = app_audit.AuditLogger(output_dir="outputs/run_logs")


# ═══════════════════════════════════════════════════════════════════
# SECTION A — MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════

_PROVIDER_MODELS: dict[str, list[str]] = {
    "ollama":     ["llama3.2:1b", "llama3.2", "mistral", "phi3:mini", "phi4:mini"],
    "openai":     ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    "openrouter": ["openai/gpt-4o-mini", "openai/gpt-4o", "anthropic/claude-3-haiku"],
}


def _get_ollama_models() -> list[str]:
    base = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
    if not base:
        return []
    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=3) as r:
            data = json.loads(r.read().decode())
        return [m["name"] for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def _get_available_models() -> list[str]:
    m: list[str] = []
    if os.getenv("OLLAMA_BASE_URL"):
        discovered = _get_ollama_models()
        m += discovered if discovered else _PROVIDER_MODELS["ollama"]
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
    if model_choice and model_choice != "no-provider-configured":
        os.environ["LLM_MODEL"] = model_choice


# ═══════════════════════════════════════════════════════════════════
# SECTION B — RESPONSE FORMATTING
# ═══════════════════════════════════════════════════════════════════

def _format_codes_response(report: dict, model_choice: str) -> str:
    """
    Formats the pipeline report into Markdown for the chat bubble.

    FORMATTING APPROACH
    -------------------
    The CSS rule ".chatbot .message.bot > div *" forces ALL child
    elements to font-size:1rem, overriding the browser UA stylesheet.
    **bold** stays at 1rem — distinguished by font-weight:700 only.
    """
    items = report.get("items", [])

    if not items:
        error = report.get("error", "")
        if error:
            return (
                "⚠️ **Retrieval failed.**\n\n"
                f"`{error}`\n\n"
                "Check ChromaDB is built (`python ingest_data.py`) and Ollama is running."
            )
        return (
            "⚠️ **No matching codes found.**\n\n"
            "Try more specific medical terminology or break the query into individual conditions."
        )

    sub_queries   = report.get("sub_queries", [])
    primary       = report.get("primary_condition", "")
    comorbidities = report.get("comorbidities", [])

    lines = [
        f"Model: `{model_choice}` · Pipeline: Hybrid (Semantic 70% + NHS Usage 20% + QOF 10%)",
    ]
    if primary or comorbidities:
        parts = []
        if primary:       parts.append(f"Primary: {primary}")
        if comorbidities: parts.append(f"Comorbidities: {', '.join(comorbidities)}")
        lines.append("Decomposed into: " + " · ".join(parts))
    if sub_queries:
        lines.append("Sub-queries: " + ", ".join(f"`{q}`" for q in sub_queries[:5]))

    lines += ["", f"📋 Code Recommendations — {len(items)} found", "---"]

    for item in items:
        rank          = item.get("rank", "—")
        score         = item.get("confidence_score", 0.0)
        score_display = f"{score:.0%}" if isinstance(score, float) else str(score)
        qof_tag       = " · 🏥 QOF" if item.get("in_qof") else ""
        usage         = item.get("usage_count", 0)
        usage_display = f" · NHS usage: {usage:,}" if usage else ""
        explanation   = item.get("explanation", "")
        term          = item.get("term", "")
        code          = item.get("code", "")

        lines.append(f"**#{rank}  {term}**{qof_tag}")
        lines.append(f"`{code}` · Score: {score_display}{usage_display}")
        if explanation:
            lines.append(f"*{explanation}*")
        lines.append("---")

    return "\n".join(l for l in lines if l is not None)


# ═══════════════════════════════════════════════════════════════════
# SECTION B2 — AUDIT PANEL FORMATTER
# ═══════════════════════════════════════════════════════════════════

def _format_audit_panel(run_id: str) -> str:
    rec = _audit._active.get(run_id)
    if not rec:
        return "*No audit record available for this run.*"

    lines = [
        f"**Run ID:** `{rec.run_id}`",
        f"**Model:** `{rec.model_name}` · **Ranker:** `{rec.ranking_model}`",
        f"**Codes logged:** {len(rec.codes)} · **Completed:** `{rec.completed_at[:19] if rec.completed_at else 'pending'}`",
        "",
    ]

    if rec.validation_flags:
        severity_icon = {"CRITICAL": "🔴", "MEDIUM": "🟡", "LOW": "🔵"}
        lines.append(f"**Validation Flags ({len(rec.validation_flags)}):**\n")
        for flag in rec.validation_flags:
            icon = severity_icon.get(flag.get("severity", "LOW"), "⚪")
            lines.append(
                f"{icon} **[{flag.get('severity', '?')}]** `{flag.get('type', '?')}` "
                f"— {flag.get('message', '')}"
            )
    else:
        lines.append("✅ **No validation issues detected.**")

    lines += [
        "",
        "**Code Provenance (top 5):**\n",
        "| Rank | Code | Term | Score | Source |",
        "|------|------|------|-------|--------|",
    ]
    for cp in rec.codes[:5]:
        snomed = getattr(cp, "snomed_code", "?")
        term   = getattr(cp, "term", "")[:35]
        rank   = getattr(cp, "rank", "?")
        score  = getattr(cp, "confidence_score", 0.0)
        src    = getattr(cp, "source_type", "csv_match")
        lines.append(f"| #{rank} | `{snomed}` | {term} | {score:.0%} | {src} |")

    lines += ["", f"*Full record saved to `outputs/run_logs/{run_id}.json`.*"]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# SECTION B3 — DATA ANALYSIS HANDLER  ← NEW
# ═══════════════════════════════════════════════════════════════════

def run_data_analysis(force: bool = False):
    """
    Triggered when the user clicks "🔬 Run Analysis" in the
    Data Analysis tab.

    Calls cluster_analysis.get_analysis_results() which:
      1. Loads combined_normalized_codes.csv
      2. Embeds all terms with bge-small-en (cached after first run)
      3. Runs KMeans k=8 (cached after first run)
      4. Runs t-SNE (cached after first run)
      5. Builds five Plotly figures

    Returns 7 values that Gradio maps to:
      analysis_status  — Markdown status message
      fig_tsne_cond    — gr.Plot: t-SNE by condition
      fig_tsne_clust   — gr.Plot: t-SNE by cluster
      fig_cluster_bar  — gr.Plot: cluster composition bar
      fig_heatmap      — gr.Plot: completeness heatmap
      fig_obs          — gr.Plot: t-SNE sized by observations
      summary_panel    — gr.Markdown: stats summary
    """
    try:
        results = cluster_analysis.get_analysis_results(force_recompute=force)
        status  = (
            "✅ **Analysis complete.** Embeddings and t-SNE loaded from cache "
            "if previously computed, or freshly computed and saved."
        )
        return (
            status,
            results["fig_tsne_condition"],
            results["fig_tsne_cluster"],
            results["fig_cluster_bar"],
            results["fig_heatmap"],
            results["fig_obs_scatter"],
            results["summary_md"],
        )
    except Exception as e:
        empty = _empty_figure(str(e))
        err   = (
            f"⚠️ **Analysis failed:** {e}\n\n"
            "Make sure `combined_normalized_codes.csv` is in the same folder as `app.py`, "
            "and that `sentence-transformers` and `scikit-learn` are installed:\n\n"
            "`pip install sentence-transformers scikit-learn plotly`"
        )
        return err, empty, empty, empty, empty, empty, ""


def _empty_figure(msg: str = "Run analysis to see results"):
    """Return a placeholder Plotly figure before analysis has been run."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_annotation(
        text=msg,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color="#768692", family="Arial"),
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="#F0F4F5",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=30, b=20),
        height=400,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════
# SECTION C — MAIN CHAT HANDLER
# ═══════════════════════════════════════════════════════════════════

def process_query(
    message:      str,
    history:      list,
    model_choice: str,
) -> tuple:
    if not message.strip():
        return history, "", "", "", ""

    if message.strip().lower() in {"hi", "hello", "hey", "help", "?"}:
        resp = (
            "👋 **Welcome to the NICE Clinical Code Assistant.**\n\n"
            "Enter a clinical query to search the SNOMED CT database. "
            "Results are ranked by semantic relevance, NHS usage, and QOF status.\n\n"
            "**Try:** *Obesity with type 2 diabetes* or *Hypertension and CKD*"
        )
        return (
            history + [{"role": "user", "content": message}, {"role": "assistant", "content": resp}],
            "", "", "", "",
        )

    _apply_model_choice(model_choice)
    run_id = _audit.start_run(message, model_choice, "ChromaDB Hybrid")

    ready, reason = pipeline.is_ready()
    if not ready:
        err = (
            "⚠️ **Engine not ready.**\n\n"
            f"{reason}\n\n"
            "1. Run `python ingest_data.py`\n"
            "2. Run `ollama serve`\n"
            "3. Restart app.py"
        )
        return (
            history + [{"role": "user", "content": message}, {"role": "assistant", "content": err}],
            "", "", "", run_id,
        )

    report = pipeline.retrieve_and_rank(query=message, model_choice=model_choice, top_k=10)
    if report.get("error"):
        err = f"⚠️ **Retrieval error:**\n\n`{report['error']}`"
        return (
            history + [{"role": "user", "content": message}, {"role": "assistant", "content": err}],
            "", "", "", run_id,
        )

    report      = pipeline.add_llm_explanations(report, message)
    response_md = _format_codes_response(report, model_choice)

    new_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": response_md},
    ]

    try:
        _audit.finish_run(run_id, report.get("items", []))
        _audit.save(run_id)
        audit_md = _format_audit_panel(run_id)
    except Exception as e:
        audit_md = f"*Audit trail unavailable: {e}*"

    try:
        metrics       = ragas_eval.evaluate(report, message)
        eval_panel_md = ragas_eval.format_eval_panel(metrics)
    except Exception as e:
        eval_panel_md = f"*Evaluation unavailable: {e}*"

    try:
        reasoning_md = reasoning_eval.generate_reasoning_trace(report, message)
    except Exception as e:
        reasoning_md = f"*Reasoning trace unavailable: {e}*"

    return new_history, eval_panel_md, reasoning_md, audit_md, run_id


def handle_feedback(vote: str, run_id: str, history: list, model_choice: str, note: str = "") -> str:
    if not run_id:
        return "⚠️ No active run to record feedback for."

    last_assistant = ""
    last_query     = ""

    if history and len(history) >= 2:
        msg_assistant = history[-1]
        msg_user      = history[-2]
        last_assistant = (
            getattr(msg_assistant, "content", "")
            if not isinstance(msg_assistant, dict)
            else msg_assistant.get("content", "")
        )
        last_query = (
            getattr(msg_user, "content", "")
            if not isinstance(msg_user, dict)
            else msg_user.get("content", "")
        )

    if isinstance(last_assistant, list):
        last_assistant = " ".join([str(i) for i in last_assistant])
    elif not isinstance(last_assistant, str):
        last_assistant = str(last_assistant)

    found_codes = re.findall(r"`(\d{6,18})`", last_assistant)

    try:
        feedback_hitl.record_feedback(
            run_id=run_id,
            query=last_query,
            vote=vote,
            recommended_codes=found_codes,
            model_name=model_choice,
            note=note,
        )
        icon = "👍" if vote == "thumbs_up" else "👎"
        return f"{icon} Feedback recorded. Thank you."
    except Exception as e:
        return f"⚠️ Could not save feedback: {e}"


# ═══════════════════════════════════════════════════════════════════
# SECTION D — NHS STYLING
# ═══════════════════════════════════════════════════════════════════

_C = {
    "blue":      "#005EB8",
    "dark_blue": "#003087",
    "lt_blue":   "#41B6E6",
    "green":     "#009639",
    "white":     "#FFFFFF",
    "pale":      "#F0F4F5",
    "border":    "#D8DDE0",
    "text":      "#212B32",
}

_CSS = f"""
body, .gradio-container {{
    background-color:{_C['pale']} !important;
    font-family:Arial,sans-serif !important;
    margin:0 !important; padding:0 !important;
}}
.nhs-main-wrapper {{
    max-width:1280px; margin:20px auto !important;
    background:white !important;
    border:1px solid {_C['border']} !important;
    border-radius:4px !important; overflow:hidden !important;
    box-shadow:0 4px 15px rgba(0,0,0,0.05) !important;
}}
.nhs-header {{ background:{_C['blue']}; padding:20px 30px; }}
.nhs-header-title {{ font-size:1.5rem; font-weight:700; color:white !important; margin-bottom:2px; }}
.nhs-header-sub   {{ font-size:0.95rem; color:white !important; opacity:0.9; }}
.nhs-status {{
    background:{_C['dark_blue']}; padding:10px 30px;
    font-size:0.85rem; color:white !important; display:flex; gap:10px;
}}
.nhs-status span, .nhs-status b {{ color:white !important; }}
.nhs-content {{ padding:15px 30px 30px 30px !important; }}
.nhs-selectors {{
    margin-bottom:10px !important; padding-bottom:5px !important;
    border-bottom:1px solid {_C['border']} !important;
}}
.nhs-methodology {{
    background:{_C['pale']}; border-left:4px solid {_C['blue']};
    border-radius:0 4px 4px 0; padding:12px 16px;
    font-size:0.88rem; color:{_C['text']}; line-height:1.6; margin-bottom:12px;
}}
.nhs-methodology strong {{ color:{_C['blue']}; }}

/* ── Analysis tab note box ── */
.analysis-info {{
    background:{_C['pale']}; border-left:4px solid {_C['lt_blue']};
    border-radius:0 4px 4px 0; padding:12px 16px;
    font-size:0.88rem; color:{_C['text']}; line-height:1.6; margin-bottom:16px;
}}
.analysis-info strong {{ color:{_C['dark_blue']}; }}

/* ── Tabs styling ── */
.tab-nav {{ border-bottom:2px solid {_C['blue']} !important; }}
.tab-nav button {{
    font-family:Arial,sans-serif !important; font-size:0.95rem !important;
    color:{_C['text']} !important; padding:10px 20px !important;
    border:none !important; background:transparent !important;
}}
.tab-nav button.selected {{
    color:{_C['blue']} !important; font-weight:700 !important;
    border-bottom:3px solid {_C['blue']} !important;
}}

.btn-thumbs-up {{
    background:{_C['green']} !important; color:white !important;
    border:none !important; border-radius:20px !important;
    padding:6px 16px !important; font-size:0.9rem !important; cursor:pointer !important;
}}
.btn-thumbs-up:hover {{ opacity:0.85 !important; }}
.btn-thumbs-down {{
    background:#DA291C !important; color:white !important;
    border:none !important; border-radius:20px !important;
    padding:6px 16px !important; font-size:0.9rem !important; cursor:pointer !important;
}}
.btn-thumbs-down:hover {{ opacity:0.85 !important; }}

/* ── Run Analysis button ── */
.btn-run-analysis {{
    background:{_C['dark_blue']} !important; color:white !important;
    border:none !important; border-radius:4px !important;
    padding:10px 24px !important; font-size:0.95rem !important;
    font-weight:700 !important; cursor:pointer !important;
}}
.btn-run-analysis:hover {{ background:{_C['blue']} !important; }}
.btn-recompute {{
    background:{_C['text']} !important; color:white !important;
    border:none !important; border-radius:4px !important;
    padding:8px 18px !important; font-size:0.85rem !important; cursor:pointer !important;
}}
.btn-recompute:hover {{ opacity:0.85 !important; }}

/* ── Chatbot ── */
.chatbot {{ border:none !important; background:{_C['pale']} !important; overflow:hidden !important; }}
.chatbot > div.wrapper, .chatbot > div[class*="wrap"] {{
    overflow-y:scroll !important; overflow-x:hidden !important;
}}
.chatbot .message-row,
.chatbot .message-row.user, .chatbot .message-row.bot,
.chatbot .message, .chatbot .message.user, .chatbot .message.bot,
.chatbot [data-testid="user-message"] {{
    background-color:transparent !important; background:transparent !important;
    border:none !important; box-shadow:none !important;
}}
.chatbot .message-row {{ min-height:auto !important; overflow:hidden !important; }}
.chatbot .message.user {{
    align-items:flex-end !important; display:flex !important;
    margin-bottom:4px !important; padding-bottom:0 !important;
}}
.chatbot .message.user > div {{
    background-color:{_C['lt_blue']} !important; color:white !important;
    border:none !important; border-radius:12px 12px 2px 12px !important;
    padding:12px 16px !important; font-size:1rem !important;
    display:inline-block !important; width:auto !important; max-width:80% !important;
    word-break:normal !important; overflow-wrap:break-word !important; box-shadow:none !important;
}}
.chatbot .message.user > div * {{ background:transparent !important; color:white !important; }}
.chatbot .message.bot > div {{
    background-color:white !important; color:{_C['text']} !important;
    border:1px solid {_C['border']} !important;
    border-radius:12px 12px 12px 2px !important;
    padding:12px 16px !important; font-size:1rem !important;
    font-family:Arial,sans-serif !important; line-height:1.6 !important;
}}
/* KEY FIX: lock all child elements to 1rem — prevents UA stylesheet
   from making <strong> and headings render larger than body text */
.chatbot .message.bot > div *,
.chatbot .message.bot > div p,
.chatbot .message.bot > div strong,
.chatbot .message.bot > div b,
.chatbot .message.bot > div em,
.chatbot .message.bot > div li,
.chatbot .message.bot > div h1,
.chatbot .message.bot > div h2,
.chatbot .message.bot > div h3,
.chatbot .message.bot > div h4 {{
    font-size:1rem !important; font-family:Arial,sans-serif !important;
    margin-top:0 !important; margin-bottom:0 !important;
}}
.chatbot .message.bot > div strong,
.chatbot .message.bot > div b {{
    font-weight:700 !important; color:{_C['text']} !important;
}}
.chatbot .message.bot > div em {{
    font-style:italic !important; color:#4a5568 !important;
}}
.chatbot .message.bot > div code {{
    background:{_C['pale']} !important; color:{_C['text']} !important;
    border:1px solid {_C['border']} !important; border-radius:3px !important;
    padding:1px 5px !important; font-family:'Courier New',monospace !important;
    font-size:0.95rem !important;
}}
.chatbot .message.bot > div hr {{
    border:none !important; border-top:1px solid {_C['border']} !important;
    margin:8px 0 !important;
}}
.chatbot .message.bot > div p {{ margin:3px 0 !important; }}
.chatbot .message-row .message-footer,
.chatbot .message-row .status, .chatbot .message-row button,
.chatbot .message-row [class*="p-2"], .chatbot .message-row [class*="gap-2"] {{
    display:none !important; height:0 !important; max-height:0 !important;
    margin:0 !important; padding:0 !important; flex:0 0 0px !important;
    overflow:hidden !important; visibility:hidden !important; opacity:0 !important;
}}
.chatbot .message-row > div:last-child {{ margin-bottom:0 !important; padding-bottom:0 !important; }}

.nhs-input {{
    background-color:white !important; border:2px solid black !important; border-radius:0px !important;
}}
.nhs-input > label, .nhs-input .container,
.nhs-input div[class*="label"], .nhs-input .form {{
    background-color:transparent !important; border:none !important; box-shadow:none !important;
}}
.nhs-input textarea {{
    background-color:white !important; color:black !important;
    border:none !important; box-shadow:none !important;
    font-size:1rem !important; font-family:Arial,sans-serif !important;
}}
.gradio-container .block, .gradio-container .form,
.gradio-container .gr-box, .gradio-container .group {{
    background:transparent !important; border:none !important; box-shadow:none !important;
}}
.eval-panel {{
    background:{_C['pale']}; border:1px solid {_C['border']};
    border-radius:4px; padding:16px; margin-top:8px;
    font-size:0.9rem; font-family:Arial,sans-serif;
}}
.analysis-summary {{
    background:white; border:1px solid {_C['border']};
    border-radius:4px; padding:16px; margin-bottom:16px;
    font-size:0.88rem; font-family:Arial,sans-serif;
}}
.nhs-footer {{
    text-align:center; padding:20px; font-size:0.85rem;
    color:#768692; border-top:1px solid {_C['border']};
}}
"""


def _logo_html() -> str:
    p = Path(__file__).parent / "nhs-logo-880x4951.jpeg"
    if p.exists():
        enc = base64.b64encode(p.read_bytes()).decode()
        return f'<img src="data:image/jpeg;base64,{enc}" style="height:38px;background:white;padding:5px;border-radius:4px;">'
    return '<div style="background:white;color:#005EB8;padding:8px 15px;font-weight:900;">NHS</div>'


def _engine_status_html() -> str:
    count = pipeline.db_code_count()
    ready, _ = pipeline.is_ready()
    if not ready and count == 0:
        return '<span style="background:#DA291C;color:white;padding:2px 8px;border-radius:10px;font-size:0.8rem;">⚠ Engine offline</span>'
    if count == 0:
        return '<span style="background:#FFB81C;color:#212B32;padding:2px 8px;border-radius:10px;font-size:0.8rem;">⚠ DB empty</span>'
    return f'<span style="background:#009639;color:white;padding:2px 8px;border-radius:10px;font-size:0.8rem;">✓ {count:,} codes</span>'


# ═══════════════════════════════════════════════════════════════════
# SECTION E — GRADIO BLOCKS INTERFACE
# ═══════════════════════════════════════════════════════════════════

def _build_interface() -> gr.Blocks:
    available_models = _get_available_models()
    default_model    = _get_default_model()

    with gr.Blocks(title="NICE Clinical Code Assistant", css=_CSS) as demo:

        # ── Shared header rendered above all tabs ─────────────────
        with gr.Column(elem_classes=["nhs-main-wrapper"]):

            gr.HTML(f"""
            <div class="nhs-header">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div class="nhs-header-title">NICE Clinical Code Assistant</div>
                        <div class="nhs-header-sub">Group 5 · NICE Employer Project 2025</div>
                    </div>
                    {_logo_html()}
                </div>
            </div>
            <div class="nhs-status">
                <span><b>Data:</b> SNOMED CT UK (2025)</span> &nbsp;|&nbsp;
                <span><b>Pipeline:</b> ChromaDB + Hybrid Ranking</span> &nbsp;|&nbsp;
                <span>{_engine_status_html()}</span>
            </div>
            """)

            # ─────────────────────────────────────────────────────
            # TOP-LEVEL TABS: Clinical Search  |  Data Analysis
            # ─────────────────────────────────────────────────────
            with gr.Tabs(elem_classes=["tab-nav"]):

                # ══════════════════════════════════════════════════
                # TAB 1 — CLINICAL SEARCH (original functionality)
                # ══════════════════════════════════════════════════
                with gr.Tab("🔍 Clinical Search"):
                    with gr.Column(elem_classes=["nhs-content"]):

                        with gr.Row(elem_classes=["nhs-selectors"]):
                            with gr.Column(scale=1, min_width=250):
                                model_dd = gr.Dropdown(
                                    choices=available_models,
                                    value=default_model,
                                    label="AI Model (explanations)",
                                    interactive=True,
                                )
                            with gr.Column(scale=3):
                                gr.HTML("""<div class="nhs-methodology">
                                    <strong>Hybrid Ranking</strong> &nbsp;·&nbsp;
                                    <strong>70%</strong> Semantic similarity &nbsp;·&nbsp;
                                    <strong>20%</strong> NHS usage frequency &nbsp;·&nbsp;
                                    <strong>10%</strong> QOF priority bonus.
                                    Codes marked 🏥 <strong>QOF</strong> are nationally mandated.
                                </div>""")

                        run_id_state = gr.State(value="")
                        chatbot      = gr.Chatbot(height=420, elem_classes=["chatbot"])

                        with gr.Row():
                            btn_up   = gr.Button("👍  Helpful",           elem_classes=["btn-thumbs-up"],   scale=1)
                            btn_down = gr.Button("👎  Needs improvement",  elem_classes=["btn-thumbs-down"], scale=1)
                            fb_note  = gr.Textbox(
                                placeholder="Optional note (e.g. 'Missing code X')",
                                container=False, scale=3, max_lines=1,
                            )
                            fb_status = gr.Markdown(value="")

                        with gr.Row():
                            query_input = gr.Textbox(
                                placeholder="Enter a clinical query...",
                                container=False,
                                elem_classes=["nhs-input"],
                                scale=5, show_label=False,
                            )
                            submit_btn = gr.Button("Search →", variant="primary", scale=1)

                        with gr.Accordion("📊 Evaluation Metrics (RAGAS-style)", open=False):
                            eval_panel = gr.Markdown(
                                value="*Run a query to see evaluation metrics.*",
                                elem_classes=["eval-panel"],
                            )

                        with gr.Accordion("🔍 Reasoning Trace", open=False):
                            reasoning_panel = gr.Markdown(
                                value="*Run a query to see the step-by-step reasoning trace.*",
                                elem_classes=["eval-panel"],
                            )

                        with gr.Accordion("🗂️ Audit Trail & Validation", open=False):
                            audit_panel = gr.Markdown(
                                value="*Run a query to see the audit trail.*",
                                elem_classes=["eval-panel"],
                            )

                        with gr.Accordion("👍👎 Analyst Feedback Summary", open=False):
                            fb_summary = gr.Markdown(
                                value=feedback_hitl.format_feedback_summary_panel(),
                                elem_classes=["eval-panel"],
                            )

                # ══════════════════════════════════════════════════
                # TAB 2 — DATA ANALYSIS (t-SNE / KMeans)  ← NEW
                # ══════════════════════════════════════════════════
                with gr.Tab("🔬 Data Analysis"):
                    with gr.Column(elem_classes=["nhs-content"]):

                        gr.HTML("""
                        <div class="analysis-info">
                            <strong>Unsupervised Cluster Analysis</strong> — NICE DAAR 2025 Code Sets<br>
                            Embeds all 5,700+ codes using <code>BAAI/bge-small-en</code>,
                            clusters with <strong>KMeans (k=8)</strong>, and projects to 2D
                            with <strong>t-SNE</strong>. Embeddings and coordinates are cached
                            to disk after the first run (~60–120 s). Subsequent loads take
                            under 2 seconds. Hover any point to see the code, term, and condition.
                            Based on the <em>BT-03 multimorbidity completeness</em> framework.
                        </div>
                        """)

                        with gr.Row():
                            btn_run_analysis = gr.Button(
                                "🔬 Run Analysis",
                                elem_classes=["btn-run-analysis"],
                                scale=2,
                            )
                            btn_recompute = gr.Button(
                                "♻️ Clear Cache & Recompute",
                                elem_classes=["btn-recompute"],
                                scale=1,
                            )

                        analysis_status = gr.Markdown(
                            value="*Click **Run Analysis** to load the visualisations.*",
                            elem_classes=["eval-panel"],
                        )

                        # Summary stats panel
                        analysis_summary = gr.Markdown(
                            value="",
                            elem_classes=["analysis-summary"],
                            visible=False,
                        )

                        # ── Row 1: t-SNE by condition + t-SNE by cluster ──
                        gr.Markdown("### t-SNE Scatter Plots")
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown(
                                    "**By Condition** — each colour is one NICE code set. "
                                    "Tight clusters = semantically similar codes within a condition. "
                                    "Overlapping colours = shared clinical territory."
                                )
                                fig_tsne_cond = gr.Plot(
                                    value=_empty_figure("Click Run Analysis"),
                                    label="t-SNE by Condition",
                                )
                            with gr.Column(scale=1):
                                gr.Markdown(
                                    "**By KMeans Cluster** — 8 clusters found in embedding space. "
                                    "A cluster spanning multiple conditions = multimorbidity "
                                    "bridging cluster (BT-03)."
                                )
                                fig_tsne_clust = gr.Plot(
                                    value=_empty_figure("Click Run Analysis"),
                                    label="t-SNE by KMeans Cluster",
                                )

                        # ── Row 2: cluster composition + observations scatter ──
                        gr.Markdown("### Cluster Composition & Usage Frequency")
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.Markdown(
                                    "**Cluster Composition** — stacked bars show which conditions "
                                    "each cluster draws from. A mixed bar = bridging cluster."
                                )
                                fig_cluster_bar = gr.Plot(
                                    value=_empty_figure("Click Run Analysis"),
                                    label="Cluster Composition",
                                )
                            with gr.Column(scale=1):
                                gr.Markdown(
                                    "**Usage Frequency** — bubble size = NHS Observation count "
                                    "(log scale). Large bubbles = high-frequency NHS codes. "
                                    "Only codes with recorded observations shown."
                                )
                                fig_obs = gr.Plot(
                                    value=_empty_figure("Click Run Analysis"),
                                    label="t-SNE — Observation Frequency",
                                )

                        # ── Row 3: completeness heatmap (full width) ──
                        gr.Markdown(
                            "### Completeness Heatmap — BT-03 Cluster Fingerprints\n"
                            "Each cell shows what proportion of a condition's codes fall in "
                            "each cluster. Rows that are bright across multiple conditions are "
                            "the **multimorbidity bridging clusters** the agent must capture "
                            "for comorbidity queries to be complete."
                        )
                        fig_heatmap = gr.Plot(
                            value=_empty_figure("Click Run Analysis"),
                            label="Completeness Heatmap",
                        )

            # ── Footer ────────────────────────────────────────────
            gr.HTML(
                '<div class="nhs-footer">NICE Employer Project 2025 · '
                'For clinical decision support only · '
                'All recommendations require expert clinical review</div>'
            )

        # ═══════════════════════════════════════════════════════════
        # EVENT WIRING — Tab 1 (Clinical Search)
        # ═══════════════════════════════════════════════════════════
        _inputs  = [query_input, chatbot, model_dd]
        _outputs = [chatbot, eval_panel, reasoning_panel, audit_panel, run_id_state]

        submit_btn.click(
            fn=process_query, inputs=_inputs, outputs=_outputs,
        ).then(fn=lambda: "", outputs=[query_input])

        query_input.submit(
            fn=process_query, inputs=_inputs, outputs=_outputs,
        ).then(fn=lambda: "", outputs=[query_input])

        btn_up.click(
            fn=lambda rid, hist, mc, n: handle_feedback("thumbs_up",   rid, hist, mc, n),
            inputs=[run_id_state, chatbot, model_dd, fb_note],
            outputs=[fb_status],
        ).then(fn=feedback_hitl.format_feedback_summary_panel, outputs=[fb_summary])

        btn_down.click(
            fn=lambda rid, hist, mc, n: handle_feedback("thumbs_down", rid, hist, mc, n),
            inputs=[run_id_state, chatbot, model_dd, fb_note],
            outputs=[fb_status],
        ).then(fn=feedback_hitl.format_feedback_summary_panel, outputs=[fb_summary])

        # ═══════════════════════════════════════════════════════════
        # EVENT WIRING — Tab 2 (Data Analysis)
        # ═══════════════════════════════════════════════════════════

        # The 7 outputs from run_data_analysis() mapped in order:
        #   analysis_status, fig_tsne_cond, fig_tsne_clust,
        #   fig_cluster_bar, fig_heatmap, fig_obs, analysis_summary
        _analysis_outputs = [
            analysis_status,
            fig_tsne_cond,
            fig_tsne_clust,
            fig_cluster_bar,
            fig_heatmap,
            fig_obs,
            analysis_summary,
        ]

        btn_run_analysis.click(
            fn=lambda: run_data_analysis(force=False),
            inputs=[],
            outputs=_analysis_outputs,
        ).then(
            # Make the summary panel visible once analysis is done
            fn=lambda: gr.update(visible=True),
            outputs=[analysis_summary],
        )

        btn_recompute.click(
            fn=lambda: run_data_analysis(force=True),
            inputs=[],
            outputs=_analysis_outputs,
        ).then(
            fn=lambda: gr.update(visible=True),
            outputs=[analysis_summary],
        )

    return demo


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NICE Clinical Code Assistant — Final (app.py)")
    print("=" * 60)
    ready, reason = pipeline.is_ready()
    print(f"Engine:  {'✓ Ready' if ready else '✗ ' + reason[:60]}")
    print(f"DB:      {pipeline.db_code_count():,} codes")
    print(f"Model:   {os.getenv('LLM_MODEL', 'NOT SET')}")
    print(f"CSV:     {'✓ Found' if (Path(__file__).parent / 'combined_normalized_codes.csv').exists() else '✗ Not found — data analysis will fail'}")
    print("\nhttp://localhost:7860  |  Ctrl+C to stop\n")
    _build_interface().launch(server_name="127.0.0.1", server_port=7860)
