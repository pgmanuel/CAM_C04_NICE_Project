# 05 — Auditability, Tracing, and Monitoring: Making Every Decision Visible

> **The hardest requirement in this project.** It is relatively easy to build a system that produces good code lists. It is significantly harder to build one where every decision is traceable, every source is cited, every step is logged, and the whole thing can be audited by a clinical governance team six months after the run. This document explains the full strategy for achieving this.

---

## Why Auditability Is Harder Than It Looks

When we say "log every tool call", it sounds simple — just write some print statements and save them to a file. But for NICE, auditability means something much more demanding. A NICE analyst who receives a system-generated code list should be able to answer *any* of the following questions about it:

- Why was code 44054006 included but not code 73211009?
- Which version of the QOF Business Rules was consulted when this list was generated?
- If the same query were run tomorrow after a QOF update, which codes would change and why?
- Who or what approved this list, at what stage, and on what date?
- If a patient cohort turns out to be miscounted, can we trace the error back to a specific code decision?

A log file with timestamps and function names cannot answer these questions. You need a principled, structured approach to provenance — the formal discipline of recording the lineage of a digital artefact. Fortunately, there is a well-established W3C standard for this called **PROV-O**, and your research document references it directly as a recommendation for this project.

---

## Layer 1 — Structured Run Logs (The Foundation)

The most basic layer of auditability is a structured run log. Every time the agent runs, it should produce a machine-readable JSON file (not a plain text log) that captures the complete execution trace. Unlike a flat log file, a structured JSON log can be queried programmatically — you can ask "which runs used QOF v49?" or "which codes were flagged for review in all runs for obesity?" and get an answer instantly.

Here is the structure each run log should follow:

```json
{
  "run_id": "run_2026_03_14_obesity_t2dm_001",
  "timestamp": "2026-03-14T10:23:45Z",
  "query": "obesity with type 2 diabetes comorbidity",
  "model_version": "claude-opus-4-5",
  "data_source_versions": {
    "qof_business_rules": "v49_2024-25",
    "opencodecounts": "2025-Q3",
    "snomed_version": "UK_20241001",
    "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
    "classifier_version": "rf_v2_trained_2026-01-15"
  },
  "tool_calls": [
    {
      "step": 1,
      "tool_name": "qof_lookup",
      "input": {"condition_keyword": "obesity"},
      "output_summary": "Returned 18 codes from OB001 indicator",
      "output_codes": ["414916001", "6077007", "..."],
      "agent_reasoning": "Starting with QOF anchor codes for highest confidence baseline",
      "timestamp": "2026-03-14T10:23:46Z"
    },
    {
      "step": 2,
      "tool_name": "semantic_code_search",
      "input": {"query": "obesity BMI overweight body weight", "top_k": 50},
      "similarity_scores": {"414916001": 0.94, "8943001": 0.61, "..."},
      "agent_reasoning": "Broadening search beyond QOF to capture non-mandated but clinically relevant codes",
      "timestamp": "2026-03-14T10:23:49Z"
    }
  ],
  "final_output": {
    "total_codes_recommended": 87,
    "high_confidence": 34,
    "medium_confidence": 31,
    "review_required": 22,
    "deprecated_flagged": 4
  }
}
```

Notice that this log captures not just *what* happened but *why* the agent chose to do it (the `agent_reasoning` field at each step). The LangChain `AgentExecutor` with `verbose=True` and `return_intermediate_steps=True` gives you the raw ingredients for this. You then need a post-processing step to structure the intermediate steps into this clean JSON format.

To implement this with minimal overhead, use `loguru` for the underlying logging backend (it handles async logging and structured output cleanly) and create a `RunLogger` class that wraps your `AgentExecutor` and captures every tool call automatically.

---

## Layer 2 — Per-Code Provenance Records (The Heart of Auditability)

The run log tells you what the agent did. The provenance record tells you *why each individual code is in the final list*. This is a separate, code-level document that sits alongside the code list itself as its permanent companion.

For every code in the output, the provenance record contains a formal description of the evidence chain that justified its inclusion. Using the W3C PROV-O framework (which your research document recommends), this can be represented in JSON-LD format — a structured, interoperable format that links to external ontologies and can be stored in a knowledge graph if needed. However, for practical purposes during the project, a clean Python dictionary that captures the same information is sufficient:

```python
provenance_record = {
    "snomed_code": "44054006",
    "description": "Type 2 diabetes mellitus",
    "confidence_tier": "HIGH",
    "inclusion_sources": [
        {
            "source_type": "QOF_BUSINESS_RULE",
            "source_version": "v49_2024-25",
            "indicator_id": "DM012",
            "indicator_description": "Patients with diabetes with HbA1c measurement",
            "authority_weight": 1.0
        },
        {
            "source_type": "OPENCODECOUNTS",
            "annual_usage_count": 4_823_109,
            "percentile_rank": 99.2,
            "trend": "STABLE",
            "authority_weight": 0.7
        },
        {
            "source_type": "SEMANTIC_SIMILARITY",
            "query": "type 2 diabetes mellitus",
            "cosine_similarity": 0.97,
            "embedding_model": "pritamdeka/S-PubMedBert-MS-MARCO",
            "authority_weight": 0.4
        }
    ],
    "classifier_score": 0.94,
    "shap_contributions": {
        "in_qof": 0.45,
        "log_frequency": 0.18,
        "cluster_id_match": 0.11,
        "source_count": 0.08
    },
    "exclusion_flags": [],
    "human_review_required": False,
    "rationale": (
        "Included at HIGH confidence. Primary justification: mandated by QOF Business "
        "Rules v49 2024-25 (Indicator DM012). Supporting evidence: recorded in 4.8M "
        "national GP records (99th percentile frequency). Classifier probability: 0.94. "
        "No deprecation or conflict flags detected."
    )
}
```

This provenance record is what NICE analysts can inspect, challenge, and archive. If they override a decision — including a code the agent flagged REVIEW, or excluding a code the agent gave HIGH confidence — that override is also recorded in the provenance record with the reviewer's initials and the date. This creates the full human-in-the-loop audit trail.

---

## Layer 3 — The Human-in-the-Loop Review Interface

Having structured provenance records is necessary but not sufficient. You also need a workflow that brings them in front of the right person in a way that supports good decision-making. Your research document references a specific design pattern called the **Governor (Verification) Pattern** — AI-generated suggestions are presented in a "provisional" state requiring explicit human approval before they become final.

In practice, this means the agent's output is never directly used as a final code list. Instead, it is automatically placed in a review queue with three sections: HIGH confidence codes ready for quick bulk approval, MEDIUM confidence codes that deserve a thirty-second scan before individual approval, and REVIEW codes that each require explicit clinical judgement with notes. The analyst must actively approve or reject every code, and their actions are logged against their user identity. No code moves from provisional to final without a human touch.

This is not just good governance — it is also how you generate more training data for the supervised classifier. Every approval and rejection from a NICE analyst is a label: codes they approve are new positive examples; codes they reject despite the agent's HIGH confidence recommendation are important failure cases that should trigger model retraining.

---

## Layer 4 — Experiment Tracking with MLflow

As the project evolves, you will iterate on many components: different embedding models, different classifier architectures, different scoring weights for the composite ranking function, different prompt templates for the LLM. Without systematic experiment tracking, it becomes impossible to know whether a change you made two weeks ago improved or degraded performance.

**MLflow** is the standard tool for this. It provides a local (or hosted) experiment tracking server where you log, for each run: the model parameters used, the evaluation metrics against the NICE gold-standard lists, the artefacts produced (the trained classifier, the embedding model checkpoint, the vectore store snapshot), and the run metadata (which data source versions were used, what time, by whom).

Set up MLflow tracking early, even before you have a working classifier. The discipline of logging everything from the start means you will have a clean experiment history when it matters most — when someone from NICE asks "is this system better or worse than it was three months ago, and what changed?"

```python
import mlflow

with mlflow.start_run(run_name="obesity_t2dm_pipeline_v3"):
    # Log which data source versions were used
    mlflow.log_params({
        "qof_version": "v49_2024-25",
        "embedding_model": "PubMedBert",
        "classifier": "RandomForest",
        "n_estimators": 200
    })
    
    # Log performance against the NICE gold standard
    mlflow.log_metrics({
        "recall_at_k50": 0.87,
        "precision_at_k50": 0.61,
        "f1_score": 0.72,
        "cohen_kappa": 0.68
    })
    
    # Save the trained classifier and the run's provenance logs as artefacts
    mlflow.sklearn.log_model(classifier, "code_inclusion_classifier")
    mlflow.log_artifact("outputs/run_2026_03_14_obesity_provenance.json")
```

---

## Layer 5 — Monitoring for Code List Drift

One of NICE's explicit business questions is: *"How should code lists be monitored for drift when SNOMED or QOF updates are released?"* This is a production monitoring problem, and it requires a specific approach.

Set up a scheduled monitoring pipeline that runs automatically when new data versions are released — at minimum, after each annual QOF update (typically in April) and each NHS SNOMED usage data release. For each existing code list in the archive, this pipeline checks three things.

First, it checks for **deprecated codes**: SNOMED codes that existed in the previous version but have been removed or inactivated in the new version. These are critical because a deprecated code in a production code list will silently fail to match any patient records — the query will return zero results without any error, which is far more dangerous than a visible failure.

Second, it checks for **QOF rule changes**: codes that were mandated by QOF in the previous version but are no longer required, or new codes that have been added to the framework. Any existing code list that references the affected indicators should be flagged for review.

Third, it checks for **usage trend changes**: codes whose usage frequency has changed significantly since the list was created. A code that has dropped from high-frequency to low-frequency in the intervening period may have been superseded by a new coding preference, even if it has not been formally deprecated.

The output of this monitoring pipeline is a **drift report** for each affected code list — a structured document summarising what has changed, which codes are at risk, and what actions are recommended. These reports should be sent automatically to the responsible analyst and logged in the same provenance system as the original code list.

---

## Layer 6 — Evaluation Metrics That Reflect Clinical Reality

Your research document makes an important point about evaluation: *"Traditional precision and recall metrics are often misleading in clinical coding because of the inherent variability in expert judgment — two clinicians may disagree on whether a specific 'probable' code should be included."*

This means that measuring your system purely by how closely it reproduces the NICE example code lists is insufficient. A code that your agent recommends but that isn't in the NICE list might genuinely be a good code that NICE missed — or it might be irrelevant noise. You cannot tell without clinical review.

The recommended approach is to use **Cohen's Kappa** as your primary agreement metric rather than raw precision and recall. Cohen's Kappa measures the agreement between your system's recommendations and the NICE analyst's final approved list, while controlling for the level of agreement that would be expected by chance. A Kappa above 0.8 represents strong agreement; a Kappa above 0.6 represents moderate to good agreement and is a realistic target for this type of system.

Additionally, the research document references **RPAD/RRAD** (Relative Precision and Recall of Algorithmic Diagnostics) — a newer metric specifically designed for clinical coding evaluation that accounts for the clinical consequences of different error types. Missing a code that would capture 50,000 patients (false negative) is a much more serious error than recommending an extra code that adds 100 spurious patients to a cohort (false positive). RPAD/RRAD weights errors by their patient-level impact, which is exactly the right framing for NICE.

---

## Practical Implementation Checklist

To summarise the auditability strategy into concrete actions, here is what the project needs to build and configure.

A `RunLogger` class that wraps `AgentExecutor` and produces structured JSON run logs with versioned data source references and per-step reasoning capture. A `ProvenanceRecord` dataclass that stores per-code evidence chains in the structure described in Layer 2, serialisable to JSON and linkable to the run log. A review queue interface (even a simple Jupyter widget to start with) that presents provisional codes in the three-tier structure and captures analyst approvals with timestamps and user identity. An MLflow experiment tracking setup configured from day one, logging every model training run and pipeline evaluation against the NICE gold-standard lists. A drift monitoring pipeline that runs on a scheduled basis after each major data source update, producing structured drift reports for every archived code list. And finally, an evaluation harness that measures Cohen's Kappa and where possible approximates RPAD/RRAD weighting for the obesity and comorbidity test cases.

None of these are optional. Each one addresses a different dimension of the defensibility requirement. Together, they ensure that the NICE clinical code recommendation system is not just an accurate tool, but a trustworthy one.

---

*Next: See `06_project_plan_and_features.md` for the full project plan with team task breakdown.*
