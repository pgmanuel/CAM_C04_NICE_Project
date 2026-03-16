# 02 — Agentic Workflow Design: Building the Clinical Code Recommendation Engine

> **Purpose:** This document explains the architecture of the AI agent that forms the core deliverable of the NICE project. It covers what an agentic workflow *is*, why it is the right approach for this problem, how each component works, and how the system satisfies NICE's specific requirements for defensibility, explainability, and auditability.

---

## What Is an Agentic Workflow?

A traditional AI system takes an input, runs it through a fixed pipeline, and returns an output. There is no decision-making — every input follows the same path. An **agentic workflow** is fundamentally different: the AI model itself decides what steps to take, in what order, and whether it has enough information to respond or needs to gather more. It has access to a set of **tools** — functions it can call — and it autonomously chooses which tools to use based on the task at hand.

Think of it like the difference between a vending machine and a research assistant. A vending machine executes a fixed sequence when you press a button. A research assistant reads your question, decides which books to pull from the shelf, cross-references them, notices a contradiction, goes back for more evidence, and then writes you a summary with citations. Our agent needs to behave like the research assistant.

For the NICE use case, this matters because building a clinical code list is not a single lookup — it requires multiple steps: interpreting the research question, searching multiple data sources, resolving conflicts between sources, flagging uncertain cases, and composing a structured output with justification. An agentic workflow can handle this dynamic, multi-step reasoning in a way a fixed pipeline cannot.

---

## System Architecture Overview

The system has four major layers that work together.

**Layer 1: The Knowledge Base (Vector Store + Structured Tables)**  
All clinical codes, their text descriptions, their embeddings, their usage frequencies, and their source provenance are stored in two complementary data structures. The vector store (ChromaDB) enables semantic similarity search — "find me codes whose descriptions are semantically similar to this concept". The structured tables (loaded into pandas or SQLite) enable exact lookups — "give me all codes in the QOF 2024-25 hypertension indicator" or "what is the usage count for code 44054006?". Having both is important: vector search is great for discovery, while exact lookup is essential for verification.

**Layer 2: The Tool Library**  
These are the functions the agent is given permission to call. Each tool is defined with a name, a description (which the LLM reads to decide whether to call it), and an input/output schema. The key tools are described in detail below.

**Layer 3: The LLM Reasoning Engine (Claude via Anthropic API)**  
This is the brain of the system — a large language model that receives the user's research question, the list of available tools, and any results from previous tool calls. It decides what to do next at each step, explains its reasoning, and ultimately composes the final structured output.

**Layer 4: The Output and Audit Layer**  
Every agent "run" produces not just a code list, but a full trace of every tool call made, every piece of evidence retrieved, and every reasoning step taken. This trace is the audit trail that NICE stakeholders can review.

---

## The Tool Library in Detail

### Tool 1: `semantic_code_search(query: str, top_k: int = 20) -> list[dict]`

This is the agent's primary discovery mechanism. It takes a plain-English query (e.g. "type 2 diabetes blood glucose monitoring") and returns the top-K most semantically similar codes from the vector store, each with its description, similarity score, and source metadata.

Internally, this tool embeds the query using the same sentence-transformer model used to embed the knowledge base (ensuring the vector spaces align), then performs approximate nearest-neighbour search in ChromaDB. The returned results are not filtered by score — the agent decides for itself whether a result with similarity 0.72 is good enough to include given the overall context.

### Tool 2: `qof_lookup(condition_keyword: str) -> list[dict]`

This tool performs an exact search of the QOF Business Rules data. Given a condition name or keyword, it returns all codes in the matching QOF indicator, along with the indicator ID, the QOF domain (Clinical, Public Health, etc.), and the indicator's description. These results carry the highest authority weight — if a code is mandated by QOF, it almost certainly belongs in the list.

### Tool 3: `usage_frequency_lookup(snomed_code: str) -> dict`

Given a specific SNOMED code, this tool returns its national usage count from the OpenCodeCounts data, along with the percentile rank (i.e. this code is used more frequently than X% of all SNOMED codes in primary care). The agent uses this to distinguish between mainstream codes and rare or obsolete ones.

### Tool 4: `hierarchy_explorer(snomed_code: str, direction: str = "both") -> dict`

This tool navigates the SNOMED concept hierarchy. Given a code, it returns its parent concepts (broader terms) and child concepts (more specific terms). The agent uses this to ensure completeness — if it has found the parent concept "Obesity", it can use this tool to retrieve all child concepts and evaluate whether any should also be included.

### Tool 5: `check_existing_nice_lists(condition: str) -> list[dict]`

This tool searches the NICE example code lists (the DAAR_2025_004 series) for any condition that matches the query. If a code list already exists for a related condition, the agent can inspect it, use it as a starting point, or compare its own recommendations against it. This is how the system learns from NICE's own historical work.

### Tool 6: `score_and_rank_candidates(codes: list[str]) -> list[dict]`

This is the integration tool that combines all signals. It takes a list of candidate code strings and returns them ranked by a composite score that combines: semantic similarity to the research question, usage frequency percentile, QOF membership (binary, with a high weight), number of sources the code appears in, and the probability output from the supervised learning classifier trained in Phase 4. The final score is a weighted sum, with weights tuned based on cross-validation against the NICE gold-standard lists.

---

## The Agent's Reasoning Loop

The agent follows a **ReAct** pattern (Reason → Act → Observe → Repeat). Here is what a typical run looks like for the query: *"Code list for patients with obesity who have type 2 diabetes as a comorbidity."*

**Step 1 — Initial planning:** The agent reads the query and reasons: "This involves two conditions — obesity and type 2 diabetes — with a comorbidity relationship. I should search for each separately, then check for shared or bridging codes. I'll start with the QOF lookup since that will give me authoritative anchors."

**Step 2 — QOF lookup for obesity:** Calls `qof_lookup("obesity")`. Receives back the QOF obesity indicator codes. Notes that BMI measurement codes are included. Flags this as high-confidence.

**Step 3 — QOF lookup for T2DM:** Calls `qof_lookup("type 2 diabetes")`. Receives back the QOF diabetes indicator codes — a substantially larger set covering diagnosis, monitoring, and complication codes.

**Step 4 — Semantic broadening:** Calls `semantic_code_search("obesity BMI overweight body weight", top_k=30)` to find additional codes beyond the QOF core. Reviews results, noting some are deprecated variants; flags these with lower confidence.

**Step 5 — Hierarchy exploration:** Calls `hierarchy_explorer("obesity", direction="children")` to ensure no specific sub-types of obesity are missing (e.g. morbid obesity, obesity due to hypothyroidism).

**Step 6 — Scoring:** Calls `score_and_rank_candidates([...all collected codes...])`. Receives ranked output. Notes that some codes from semantic search have low frequency scores — the agent reasons: "These low-frequency codes may be overly specific clinical sub-types; I will include them but flag for clinical review."

**Step 7 — Check existing NICE lists:** Calls `check_existing_nice_lists("obesity type 2 diabetes")`. Finds the DAAR_2025_004 BMI and T2DM lists. Compares its own recommendations against them and notes five codes that appear in the NICE list but were not captured by its search. Goes back and adds them with a note: "Present in NICE reference list; justification from existing NICE methodology."

**Step 8 — Composition:** Synthesises the final structured code list, grouping codes by condition, with confidence tiers (HIGH / MEDIUM / REVIEW), per-code source citations, and a brief rationale for each.

---

## Satisfying NICE's Core Requirements

**Defensibility** is achieved because every code recommendation traces back to at least one authoritative source (QOF, NHS reference set, OpenCodeCounts) cited by the agent's tool calls. A code with no source citation cannot appear in the output.

**Explainability** is achieved through the SHAP values from the supervised classifier and the agent's chain-of-thought reasoning trace. For each code, the system can explain the contribution of each signal to its inclusion decision.

**Auditability** is achieved by logging every tool call with its inputs, outputs, and the agent's reasoning for making the call. This log can be reviewed by a NICE analyst or archived alongside the final code list as evidence of due diligence.

**Uncertainty surfacing** is achieved through the confidence tier system (HIGH / MEDIUM / REVIEW) and explicit flags for codes at the boundary (e.g. "commonly used interchangeably with a confirmed code — recommend clinical validation").

**Scalability** is achieved through LangChain's orchestration framework, which allows the same agent logic to be applied to any condition by simply changing the input research question. Adding a new data source (a new NHS reference set, for example) requires only adding a new tool — the agent will automatically learn to use it.

---

## How to Extend the System

The architecture is deliberately modular, which means extending it in any of the following directions is straightforward.

To add a new data source, you embed the new codes with the same sentence-transformer model, add them to ChromaDB with appropriate metadata, and create a new tool function that queries them. The agent will then incorporate this source in its reasoning automatically.

To improve recommendation quality, you retrain the supervised classifier from Phase 4 on a larger set of NICE gold-standard lists. As NICE completes more analyses and contributes more labelled code lists, the model continuously improves.

To add multi-condition comorbidity complexity (the key NICE use case around obesity + 1 comorbidity, obesity + 2 comorbidities, etc.), you pass the agent a structured research question with explicit comorbidity slots, and the agent runs its tool loop for each condition in the hierarchy, then performs a final merge and deduplication step to produce a unified multi-condition code list.

---

## Running the Agent: A Code Sketch

The following is a conceptual outline of the agent setup using LangChain with the Anthropic Claude API. This is not production-ready code, but illustrates the structure.

```python
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
import chromadb
import pandas as pd

# --- Initialise the knowledge base ---
# (assumes you have already embedded all codes and stored them)
chroma_client = chromadb.PersistentClient(path="./clinical_codes_db")
collection = chroma_client.get_collection("snomed_codes")

# Load structured sources for exact lookup
qof_df = pd.read_csv("data/qof_business_rules_2024_25.csv")
usage_df = pd.read_csv("data/snomed_usage_counts.csv")

# --- Define the tools the agent can call ---

@tool
def semantic_code_search(query: str, top_k: int = 20) -> list:
    """
    Search for SNOMED codes semantically similar to the given clinical concept.
    Use this when you need to discover codes beyond the QOF core or explore
    a clinical domain broadly.
    """
    # Embed the query using the same model used for the knowledge base
    results = collection.query(query_texts=[query], n_results=top_k)
    return [
        {
            "code": meta["snomed_code"],
            "description": meta["description"],
            "similarity": 1 - dist,  # ChromaDB returns L2 distance
            "sources": meta["sources"]
        }
        for meta, dist in zip(
            results["metadatas"][0], results["distances"][0]
        )
    ]

@tool
def qof_lookup(condition_keyword: str) -> list:
    """
    Look up clinical codes required by the QOF Business Rules (2024-25)
    for a given condition. QOF codes carry the highest authority weight.
    """
    mask = qof_df["condition"].str.contains(condition_keyword, case=False, na=False)
    return qof_df[mask].to_dict(orient="records")

@tool
def usage_frequency_lookup(snomed_code: str) -> dict:
    """
    Retrieve the national usage frequency for a specific SNOMED code
    from NHS England primary care data. Higher frequency = more mainstream code.
    """
    row = usage_df[usage_df["snomed_code"] == snomed_code]
    if row.empty:
        return {"snomed_code": snomed_code, "count": 0, "percentile": 0}
    return row.iloc[0].to_dict()

# --- Define the system prompt ---

system_prompt = """You are an expert clinical informatics assistant working for NICE 
(National Institute for Health and Care Excellence). Your task is to generate comprehensive, 
defensible clinical code lists for research questions.

For every code you recommend, you must cite at least one authoritative source 
(QOF Business Rules, NHS Reference Set, or high-frequency usage data).
Flag codes that are uncertain or require clinical review.
Think step by step. Use the tools available to you systematically.

Output format: structured JSON with fields: snomed_code, description, 
confidence (HIGH/MEDIUM/REVIEW), sources (list), rationale (string)."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# --- Assemble and run the agent ---

llm = ChatAnthropic(model="claude-opus-4-5", temperature=0)
tools = [semantic_code_search, qof_lookup, usage_frequency_lookup]

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({
    "input": "Build a code list for patients with obesity who also have type 2 diabetes."
})
```

---

*This completes the three-document project planning package. The recommended reading order is `00_project_brief_plain_english.md` → `01_data_science_roadmap.md` → `02_agentic_workflow_design.md`.*
