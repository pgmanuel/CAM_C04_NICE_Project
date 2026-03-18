# 09 — Multi-Agent Patterns: LangChain, LangGraph, Memory, and Prompt Engineering

> **The orchestration layer is where individual tools become an intelligent system.** You could have the best vector store, the best embedding model, and the best classifier in the world — but without a well-designed orchestration layer, the system cannot decide *how* to use them together for a novel research question. This document explains the architecture of multi-agent systems, the difference between LangChain and LangGraph, how memory works in agentic contexts, and how to write prompts that produce defensible reasoning in a healthcare setting.

---

## Part 1 — What "Agentic" Actually Means (And Why It's Different)

Before distinguishing between LangChain and LangGraph, it helps to be precise about what makes a system "agentic" versus a conventional pipeline. A conventional pipeline is deterministic: you define Step 1, Step 2, Step 3, and the system always executes them in that order. A pipeline is excellent when you know in advance exactly what needs to happen. But for the NICE use case, you don't always know in advance how many retrieval steps are needed, which sources are relevant, or whether the initial results are good enough or need refinement.

An **agent** solves this by giving the decision-making to the language model itself. The LLM reads the task, looks at the tools available to it, decides which tool to call first, reads the result, decides whether to call another tool or whether it has enough information to answer, and continues this loop until it reaches a satisfactory conclusion. This is called the **ReAct loop** (Reason, Act, Observe), and it is the fundamental pattern underlying all the agent architectures discussed in this document.

The power of the agentic approach is that it generalises. You don't need to write a separate pipeline for "obesity alone" vs "obesity with hypertension" vs "obesity with hypertension and dyslipidaemia." The same agent, given the same tools, can handle all three by reasoning about what each case requires and calling tools accordingly. The cost is that the system is less predictable than a fixed pipeline and requires careful prompt engineering and observability to keep it on track.

---

## Part 2 — LangChain: The Tool-Calling Foundation

**LangChain** is best understood as a framework for giving language models access to functions and chaining those function calls together. It provides three core abstractions that are essential for this project.

The first is the **Tool** — a Python function wrapped with a name and a description that the LLM can read. The description is not cosmetic. The LLM reads the description to decide *whether and when* to call the tool. If your description is vague or inaccurate, the agent will call the wrong tools at the wrong times. Writing good tool descriptions is a form of prompt engineering, and it is one of the most impactful things you can do to improve agent performance.

The second abstraction is the **Agent** — the LLM with tools attached, operating in a ReAct loop. When you call `agent.run("Identify codes for obesity with T2DM")`, the agent doesn't immediately produce an answer. It reasons about what to do first (the "Reason" step), calls a tool (the "Act" step), receives the result (the "Observe" step), and repeats. You can observe this loop by setting `verbose=True`, which prints every reasoning step to the console. Always use verbose mode during development — it is your window into the agent's decision-making.

The third is the **AgentExecutor** — the runtime that manages the loop, handles errors, enforces iteration limits, and optionally captures the full intermediate step trace for the audit log. The `return_intermediate_steps=True` parameter is crucial for auditability because it gives you programmatic access to every tool call the agent made, which you feed into the `RunLogger` described in `05_auditability_and_monitoring.md`.

Here is a complete implementation of the NICE clinical code agent using the modern LangChain tool-calling API:

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
import json
from datetime import datetime

# -----------------------------------------------------------------------
# DEFINING TOOLS — Write descriptions as if explaining to a thoughtful
# clinical colleague, not as code documentation. The LLM reads these
# descriptions to decide when to use each tool. Precision matters.
# -----------------------------------------------------------------------

@tool
def qof_lookup(condition_keyword: str) -> str:
    """
    Retrieve clinical codes mandated by the NHS Quality and Outcomes Framework 
    (QOF) Business Rules (version 49, 2024-25) for a given clinical condition.
    
    Use this tool FIRST when building any code list, as QOF codes represent
    the highest-authority, nationally mandated codes that GPs are required to
    use. Codes returned by this tool should always be included at HIGH confidence.
    
    Input: A condition name or keyword (e.g., 'obesity', 'type 2 diabetes',
           'hypertension', 'dyslipidaemia')
    Output: JSON list of codes with indicator ID, description, and refset reference.
    
    Examples of when to call this tool:
    - Before any semantic search — establish the QOF anchor codes first
    - When you need to verify whether a candidate code is policy-mandated
    - When you need the exact indicator ID for a condition (for the audit trail)
    """
    # In production: query the parsed QOF DataFrame loaded at startup
    # Stub return for illustration:
    return json.dumps([
        {"indicator_id": "OB001", "snomed_code": "414916001",
         "description": "Obesity (BMI 30+)", "authority": "QOF_MANDATED"}
    ])


@tool
def semantic_code_search(query: str, top_k: int = 20) -> str:
    """
    Search the SNOMED CT vector store for clinical codes semantically similar 
    to the given query. Uses BioBERT embeddings and cosine similarity to find 
    codes that match the MEANING of the query, even when no keywords are shared.
    
    Use this tool to discover codes beyond the QOF core — particularly for
    capturing clinical variants, legacy codes, and comorbidity-specific codes.
    Results are sorted by semantic similarity score (highest first).
    
    Input: A plain-English description of a clinical concept (1-3 sentences).
           Be specific — 'type 2 diabetes with peripheral neuropathy' will 
           return more precise results than 'diabetes'.
    Output: JSON list of up to top_k codes with similarity scores and metadata.
    
    Note: Always check the 'deprecated_flag' field in results. Codes flagged
    as deprecated should be excluded or moved to REVIEW tier.
    """
    # In production: call semantic_search() from 08_embeddings_and_semantic_search.py
    return json.dumps([
        {"snomed_code": "44054006", "description": "Type 2 diabetes mellitus",
         "similarity": 0.94, "in_qof": True, "deprecated_flag": False}
    ])


@tool  
def hierarchy_explorer(snomed_code: str, direction: str = "both") -> str:
    """
    Navigate the SNOMED CT polyhierarchical concept graph to find parent 
    (broader) and child (more specific) concepts related to a given code.
    
    Use this tool to ensure COMPLETENESS — after identifying a key concept,
    explore its children to capture all relevant sub-types, and its parents
    to understand which broader categories it belongs to.
    
    This is especially important for the NICE comorbidity use case: a code
    like 'obesity hypertension' may be a child of BOTH 'obesity' AND 
    'hypertension' — the polyhierarchy means it appears in two branches
    simultaneously and may be missed if you only traverse one branch.
    
    Input: 
        snomed_code: The SNOMED code to explore (e.g., '414916001')
        direction: 'parents' for broader concepts, 'children' for more specific,
                   'both' for complete neighbourhood (default)
    Output: JSON dict with 'parents' list and 'children' list, each with
            code, description, and relationship type.
    """
    return json.dumps({
        "parents": [{"code": "238136002", "description": "Morbid obesity"}],
        "children": [{"code": "57653000", "description": "Obesity hypertension"}]
    })


@tool
def usage_frequency_lookup(snomed_code: str) -> str:
    """
    Retrieve the national primary care usage frequency for a specific SNOMED 
    code from the NHS England SNOMED Code Usage in Primary Care dataset 
    (OpenCodeCounts, Bennett Institute for Applied Data Science).
    
    Use this tool to validate candidate codes identified by semantic search.
    A code with very low national usage (<1,000 annual occurrences) may be:
    - A recently deprecated code
    - An overly specific sub-type used only in specialist settings
    - A legitimate rare-condition code (in which case low frequency is expected)
    
    Input: SNOMED code string (e.g., '44054006')
    Output: JSON dict with annual_count, percentile_rank, and usage_trend
            ('growing', 'stable', or 'declining').
    """
    return json.dumps({
        "snomed_code": "44054006",
        "annual_count": 4823109,
        "percentile_rank": 99.2,
        "usage_trend": "stable",
        "deprecated_flag": False
    })


@tool
def score_and_rank_candidates(codes_json: str) -> str:
    """
    Apply the composite scoring function to a list of candidate SNOMED codes
    and return them ranked by inclusion probability, with SHAP-derived 
    explanations for each score component.
    
    The composite score combines: QOF membership (highest weight), national 
    usage frequency percentile, number of authoritative sources, semantic 
    similarity to the target query, and supervised classifier probability.
    Weights are stored in config/scoring_weights.yaml and are tunable.
    
    Call this tool AFTER you have gathered candidates from qof_lookup,
    semantic_code_search, and hierarchy_explorer — it is the final ranking
    step before composing the output.
    
    Input: JSON string containing a list of SNOMED code strings to score
           (e.g., '["44054006", "73211009", "414916001"]')
    Output: JSON list sorted by composite_score (descending), each entry
            containing the score, confidence_tier, and shap_contributions dict.
    """
    codes = json.loads(codes_json)
    # In production: call the trained Random Forest classifier + SHAP
    return json.dumps([
        {"snomed_code": c, "composite_score": 0.85,
         "confidence_tier": "HIGH",
         "shap_contributions": {
             "in_qof": 0.45, "log_frequency": 0.18, "cluster_match": 0.11
         }}
        for c in codes
    ])
```

---

## Part 3 — The System Prompt: Where Clinical Reasoning Is Encoded

The system prompt is the most important single piece of text in the entire agentic system. It is the set of standing instructions the LLM receives before every conversation. For a general chatbot, the system prompt might simply say "You are a helpful assistant." For a NICE clinical code recommendation system, the system prompt needs to encode the entire reasoning protocol that a NICE analyst would apply — the hierarchy of evidence sources, the criteria for confidence tiers, the handling of uncertain cases, and the required output format.

A poorly written system prompt produces a system that is fast but untrustworthy. A well-written one produces a system that reasons the way a careful clinical informatician would. The following prompt is designed to encode the NICE-specific reasoning rules explicitly, rather than relying on the model's general knowledge:

```python
NICE_SYSTEM_PROMPT = """You are a specialist clinical informatics assistant working 
within NICE (National Institute for Health and Care Excellence). Your role is to 
generate comprehensive, defensible, and auditable clinical code lists for NICE 
research questions, following a rigorous evidence-based methodology.

## YOUR MANDATORY REASONING PROTOCOL

You MUST follow this exact sequence for every code list request. Do not skip steps.

STEP 1 — ANCHOR WITH QOF: Always begin by calling qof_lookup() for each condition 
mentioned in the research question. QOF codes are your highest-confidence anchors. 
Log the QOF version and indicator IDs for every code found at this step.

STEP 2 — BROADEN WITH SEMANTIC SEARCH: Call semantic_code_search() with 2-3 
different phrasings of each condition to capture codes beyond the QOF core. 
Vary the specificity of your queries — one broad query and one specific query 
per condition typically produces the best coverage.

STEP 3 — ENSURE COMPLETENESS WITH HIERARCHY: For each key concept identified, 
call hierarchy_explorer() to check for relevant child concepts (more specific 
sub-types) you may have missed. Pay particular attention to bridging concepts 
that are children of MULTIPLE parent conditions — these are critical for 
comorbidity cohorts.

STEP 4 — VALIDATE WITH USAGE DATA: For any candidate code identified ONLY 
through semantic search (not in QOF), call usage_frequency_lookup() to check 
its national prevalence. Codes with less than 1,000 annual occurrences AND 
no QOF mandate should be moved to REVIEW tier with an explicit note.

STEP 5 — SCORE AND RANK: Once you have gathered all candidates, call 
score_and_rank_candidates() with the full candidate list. Use the returned 
scores and SHAP contributions to assign final confidence tiers.

## CONFIDENCE TIER RULES (apply exactly as stated)

HIGH confidence requires: present in QOF Business Rules v49 2024-25 OR 
present in NHS England Reference Set AND national usage >100,000/year.

MEDIUM confidence requires: not in QOF but present in NHS Reference Set 
OR semantic similarity >0.80 AND usage >10,000/year.

REVIEW required for: semantic similarity >0.65 but not in any official source, 
OR deprecated flag raised, OR usage <1,000/year despite official source presence.

## OUTPUT FORMAT (you must follow this exactly)

For each code in the final list, produce a structured entry containing:
- snomed_code: the SNOMED CT concept identifier
- description: the official SNOMED preferred term
- confidence_tier: HIGH, MEDIUM, or REVIEW
- sources: list of authoritative sources justifying inclusion
- rationale: 1-2 sentences explaining why this code is included, written in 
  plain English suitable for a NICE clinical reviewer (not technical jargon)
- review_note: (REVIEW tier only) specific question or concern for the 
  clinical reviewer to address

## WHAT YOU MUST NEVER DO

Never recommend a code without citing at least one authoritative source. 
Never suppress or hide uncertainty — if you are unsure, assign REVIEW tier. 
Never omit the hierarchy exploration step for comorbidity queries — 
bridging concepts in the SNOMED polyhierarchy are frequently missed and 
are often the most clinically important codes in comorbidity cohorts.
Never fabricate SNOMED codes or QOF indicator IDs — only report codes 
that are returned by the tools you have been given."""
```

Several design decisions in this prompt are worth calling out explicitly. The numbered protocol with the explicit "do not skip steps" instruction is important because LLMs have a tendency to take shortcuts when they feel confident — they may retrieve QOF codes and jump straight to output without doing the hierarchy exploration. Making the protocol explicit and numbered reduces this tendency significantly. The confidence tier rules are written as precise, binary conditions rather than vague guidelines, because LLMs produce more consistent outputs when the criteria are unambiguous. The output format specification at the end produces structured outputs that can be parsed programmatically into the provenance record system.

---

## Part 4 — LangGraph: When You Need State and Conditional Branching

LangChain's `AgentExecutor` works well for the ReAct loop pattern — give the LLM tools and let it decide when it's done. But it has limitations for more complex multi-agent workflows where you need explicit control flow, conditional branching, or specialised sub-agents for different tasks. This is where **LangGraph** comes in.

LangGraph models your workflow as a directed graph where **nodes are functions or LLM calls** and **edges are the transitions between them**. Unlike AgentExecutor's unconstrained loop, LangGraph lets you define exactly which states the system can be in and which transitions are allowed. This is particularly valuable for the NICE use case because you can implement the multi-agent architecture described in the research document (Extraction Agent → Terminology Agent → Hierarchy Explorer → Policy Compliance Agent → Audit Agent) as an explicit graph with guaranteed execution order.

The key distinction between the two approaches is this: use `AgentExecutor` (LangChain) when you want the LLM to decide the order of operations freely within a set of available tools. Use **LangGraph** when you have a defined workflow with stages, where certain steps must happen before others, and where different branches of logic are needed based on intermediate results.

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
from typing import TypedDict, Annotated
import operator

# -----------------------------------------------------------------------
# STATE DEFINITION
# LangGraph passes a shared state object between all nodes in the graph.
# Every node can read from and write to this state.
# This is how information flows between the specialised agents.
# -----------------------------------------------------------------------

class CodeListState(TypedDict):
    """The shared state object passed between all nodes in the workflow."""
    research_question: str          # The original input from the NICE analyst
    conditions_identified: list     # Output of the Extraction Agent
    qof_anchor_codes: list          # Output of the QOF lookup step
    semantic_candidates: list       # Output of semantic search steps
    hierarchy_candidates: list      # Output of hierarchy exploration
    all_candidates: list            # Merged, deduplicated candidate list
    scored_candidates: list         # Output of scoring function
    final_code_list: list           # The approved, tiered final output
    audit_trace: Annotated[list, operator.add]  # Accumulated audit log entries
    iteration_count: int            # Safety guard against infinite loops
    needs_review: bool              # Flag for human-in-the-loop trigger


# -----------------------------------------------------------------------
# NODE FUNCTIONS — Each node is a function that takes the current state,
# does some work, and returns a dict of state updates.
# -----------------------------------------------------------------------

def extraction_agent(state: CodeListState) -> dict:
    """
    Node 1: Extraction Agent
    Parses the research question to identify distinct clinical conditions
    and comorbidity relationships. Produces a structured list of conditions
    that subsequent agents will process.
    
    For 'obesity with type 2 diabetes and hypertension', this should
    identify three conditions: obesity (primary), T2DM (comorbidity 1),
    hypertension (comorbidity 2).
    """
    llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0)
    
    response = llm.invoke([
        ("system", """Extract all clinical conditions from the research question.
         Return a JSON list where each item has: condition_name (string), 
         is_primary (boolean), and search_keywords (list of synonyms/variants).
         Return ONLY valid JSON, no other text."""),
        ("human", state["research_question"])
    ])
    
    import json
    conditions = json.loads(response.content)
    
    return {
        "conditions_identified": conditions,
        "audit_trace": [{
            "agent": "extraction_agent",
            "action": "parse_research_question",
            "output": conditions,
            "timestamp": datetime.now().isoformat()
        }]
    }


def qof_anchor_node(state: CodeListState) -> dict:
    """
    Node 2: QOF Policy Compliance Agent
    Retrieves mandated codes for all identified conditions from QOF v49.
    These become the HIGH-confidence anchor codes.
    """
    anchor_codes = []
    audit_entries = []
    
    for condition in state["conditions_identified"]:
        result = qof_lookup.invoke(condition["condition_name"])
        codes = json.loads(result)
        anchor_codes.extend(codes)
        audit_entries.append({
            "agent": "qof_anchor_node",
            "action": "qof_lookup",
            "input": condition["condition_name"],
            "codes_found": len(codes),
            "timestamp": datetime.now().isoformat()
        })
    
    return {"qof_anchor_codes": anchor_codes, "audit_trace": audit_entries}


def should_continue(state: CodeListState) -> str:
    """
    Conditional edge function: decides whether to continue the workflow
    or hand off to human review.
    
    LangGraph uses these router functions to implement conditional branching —
    the return value is the name of the next node to execute.
    """
    # Safety guard: prevent infinite loops
    if state.get("iteration_count", 0) > 10:
        return "generate_output"
    
    # If too many codes flagged for review, escalate to human
    review_count = sum(
        1 for c in state.get("scored_candidates", [])
        if c.get("confidence_tier") == "REVIEW"
    )
    if review_count > 20:
        return "human_review_required"
    
    return "generate_output"


# -----------------------------------------------------------------------
# GRAPH CONSTRUCTION
# Connect the nodes with directed edges to form the workflow graph.
# -----------------------------------------------------------------------

def build_nice_code_list_graph() -> StateGraph:
    """
    Construct the complete multi-agent workflow as a LangGraph StateGraph.
    
    The graph enforces execution order: extraction → QOF anchoring → 
    semantic search → hierarchy exploration → scoring → output.
    Conditional branching handles the human-review escalation path.
    """
    workflow = StateGraph(CodeListState)
    
    # Add all nodes
    workflow.add_node("extraction_agent", extraction_agent)
    workflow.add_node("qof_anchor", qof_anchor_node)
    # (semantic_search_node, hierarchy_node, scoring_node would be added similarly)
    
    # Define the entry point
    workflow.set_entry_point("extraction_agent")
    
    # Add sequential edges
    workflow.add_edge("extraction_agent", "qof_anchor")
    # workflow.add_edge("qof_anchor", "semantic_search")
    # workflow.add_edge("semantic_search", "hierarchy_explorer")
    # workflow.add_edge("hierarchy_explorer", "scoring")
    
    # Add conditional branching at the scoring step
    workflow.add_conditional_edges(
        "qof_anchor",  # After scoring, decide next step
        should_continue,
        {
            "generate_output": END,
            "human_review_required": END  # In production: route to review queue
        }
    )
    
    return workflow.compile()


# Example usage:
# graph = build_nice_code_list_graph()
# result = graph.invoke({
#     "research_question": "obesity with type 2 diabetes and hypertension",
#     "audit_trace": [],
#     "iteration_count": 0,
#     "needs_review": False
# })
```

---

## Part 5 — Memory: Giving the Agent Context Across Turns

One significant limitation of LLMs is that they have no inherent memory between calls — every call starts from a blank slate. For a clinical code recommendation agent, this means that if an analyst has a back-and-forth conversation ("Now also add codes for dyslipidaemia", "Remove codes related to secondary hypertension"), the agent loses context about what was already decided unless memory is explicitly managed.

LangChain provides several memory strategies, and choosing the right one for this use case requires thinking about what information the agent actually needs to retain.

**ConversationBufferMemory** simply stores every message in the conversation history and passes the entire history to the LLM with each call. This is the simplest approach but becomes expensive as conversations grow long — once the history exceeds the LLM's context window, it must be truncated.

**ConversationSummaryMemory** periodically compresses older parts of the conversation history into a summary, keeping the most recent exchanges verbatim and summarising older ones. This is a good choice for longer clinical code building sessions where the analyst goes through multiple refinement rounds.

**VectorStoreRetrieverMemory** stores conversation history as embeddings in a vector store and retrieves only the most relevant past exchanges based on the current query. This is the most sophisticated option and the right choice for multi-session work — where an analyst returns days later to refine a code list they built previously.

For this project, a pragmatic starting approach is ConversationBufferMemory with a rolling window of the last five exchanges, which covers the typical back-and-forth within a single code-building session without the complexity of summary compression.

---

## Part 6 — Prompt Engineering for Healthcare: Key Principles

Writing effective prompts for a healthcare AI system requires more care than for general applications, because the consequences of errors are higher and the standard of reasoning required is more rigorous. The following principles should guide every prompt you write for this project.

**Be explicit about the chain of evidence, not just the conclusion.** A prompt that says "recommend clinical codes" will produce codes. A prompt that says "recommend clinical codes, and for each code cite the specific source (QOF indicator ID, Reference Set name, or usage frequency statistic) that justifies its inclusion" will produce auditable codes. Explicitness in the prompt produces explicitness in the output.

**State negative constraints as explicitly as positive ones.** The system prompt above includes a "WHAT YOU MUST NEVER DO" section. This is not redundant with the positive protocol — it catches edge cases that the positive instructions don't cover, such as the temptation to skip hierarchy exploration when initial results seem comprehensive.

**Use few-shot examples for the output format.** If you need the output in a specific JSON structure (which you do, for the provenance record system), include one or two example outputs in the prompt. LLMs are much more consistent at producing a specific format when they have seen examples of it, rather than just a description of it.

**Temperature should be zero for this application.** The `temperature` parameter controls how much randomness the LLM introduces in its outputs. A temperature of zero produces the most deterministic, consistent outputs — the same input will always produce the same output. For a clinical code recommendation system where reproducibility and consistency are core requirements, temperature zero is the correct setting. Higher temperatures are appropriate for creative tasks but introduce unacceptable variability in a governance-sensitive healthcare context.

---

*Next: See `10_annotated_code_skeletons.md` for a complete, executable set of annotated code examples covering every technique in the project.*
