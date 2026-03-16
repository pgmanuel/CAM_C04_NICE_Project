# BT-06 — Evaluation Framework, Explainability & The NICE Grading Rubric

> **The capstone document in the backtesting suite.** Everything in BT-01 through BT-05 produces evidence — frequency profiles, cluster completeness scores, temporal validity classifications, pipeline attribution maps. This document explains how to translate all of that evidence into a single, structured evaluation report that a NICE clinical governance team can read, act on, and archive. It also defines the grading rubric: the precise criteria that determine whether the agent has passed, partially passed, or failed for each condition it was tested on.

---

## Why "Performance" Needs to Be Defined in Layers

A common mistake in AI evaluation is to use a single metric as a proxy for overall quality. Accuracy, F1-score, AUROC — these aggregate metrics are useful engineering tools during model development, but they are insufficient for presenting an AI system to clinical and governance stakeholders for several reasons. They don't distinguish between different types of errors, they hide the degree of confidence the system has in its outputs, they don't capture whether the system's reasoning is sound even when its conclusion is wrong, and they don't address the question of whether the system can be trusted to behave consistently when the inputs change slightly.

For NICE specifically, "good" is a multi-dimensional concept. The research document enumerates four dimensions that need to be separately measured and reported: semantic accuracy (are the recommended codes the right ones?), operational efficiency (does the system reduce the analyst's workload meaningfully?), factual grounding (are the recommendations traceable to authoritative sources rather than hallucinated?), and population coverage (does the resulting patient cohort capture the right number and type of patients?). Only a system that scores well across all four dimensions is fit for purpose in NICE's governance environment.

The evaluation framework in this document operationalises these four dimensions into specific, measurable metrics and defines the pass/fail thresholds that constitute a defensible claim that the system is ready for supervised analyst use.

---

## Dimension 1 — Semantic Accuracy

Semantic accuracy measures how well the agent's code recommendations match what expert clinicians would select. The primary metric is **Cohen's Kappa**, which measures agreement between the agent's binary recommendations (include/exclude) and the NICE analyst's binary decisions, while controlling for the level of agreement that would be expected by chance alone.

The formula for Cohen's Kappa is:

$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

Where $P_o$ is the observed agreement (the fraction of codes on which agent and analyst agree) and $P_e$ is the expected agreement by chance (computed from the marginal distributions of each rater's decisions). A Kappa of 1.0 represents perfect agreement; 0.0 represents agreement no better than chance; negative values represent worse-than-chance agreement (systematic disagreement). In clinical coding research, Kappa above 0.80 is considered strong agreement; above 0.60 is considered moderate to good.

The secondary metrics are condition-level recall and precision, but weighted by clinical consequence rather than treated symmetrically. The research document introduces **RPAD** (Relative Precision of Algorithmic Diagnostics) and **RRAD** (Relative Recall of Algorithmic Diagnostics) as clinical-consequence-weighted variants of these standard metrics. The weighting reflects the asymmetric error cost described in BT-00: missing a high-frequency code that captures tens of thousands of patients is penalised far more heavily than adding a low-frequency code that captures a handful of spurious patients. Computing RPAD and RRAD requires knowing the patient-population impact of each code — the usage frequency data from OpenCodeCounts provides an approximation of this, with national usage count serving as a proxy for the number of patients the code would affect.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, classification_report


def compute_nice_evaluation_metrics(
        backtest_runs: list,
        usage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full NICE evaluation metric suite for each backtested condition.
    
    Returns a DataFrame with one row per condition and columns for all four
    evaluation dimensions: semantic accuracy, operational efficiency,
    factual grounding, and a composite score.
    
    The composite score is a weighted average that reflects NICE's priorities:
    factual grounding carries the highest weight because an ungrounded
    recommendation is a governance risk regardless of its accuracy.
    """
    results = []
    
    for run in backtest_runs:
        # ---------------------------------------------------------------
        # Semantic accuracy metrics
        # ---------------------------------------------------------------
        all_relevant_codes = run.nice_gold_codes | run.final_codes
        
        y_true = [1 if c in run.nice_gold_codes else 0
                  for c in all_relevant_codes]
        y_pred = [1 if c in run.final_codes else 0
                  for c in all_relevant_codes]
        
        kappa = cohen_kappa_score(y_true, y_pred)
        recall = len(run.true_positives) / max(len(run.nice_gold_codes), 1)
        precision = len(run.true_positives) / max(len(run.final_codes), 1)
        f1 = 2 * (precision * recall) / max(precision + recall, 1e-9)
        
        # Patient-weighted recall: weight each missed code by its national
        # usage frequency as a proxy for patient population impact.
        # A missed code with 5 million annual usages is far more consequential
        # than a missed code with 50 usages.
        fn_codes = run.false_negatives
        fn_usage = usage_df[
            usage_df['snomed_code'].isin(fn_codes)
        ]['usage_count'].sum()
        total_nice_usage = usage_df[
            usage_df['snomed_code'].isin(run.nice_gold_codes)
        ]['usage_count'].sum()
        rrad = 1.0 - (fn_usage / max(total_nice_usage, 1))  # RRAD: patient-weighted recall
        
        fp_codes = run.false_positives
        fp_usage = usage_df[
            usage_df['snomed_code'].isin(fp_codes)
        ]['usage_count'].sum()
        agent_usage = usage_df[
            usage_df['snomed_code'].isin(run.final_codes)
        ]['usage_count'].sum()
        rpad = len(run.true_positives) / max(len(run.final_codes), 1)  # Standard precision approximation
        
        # ---------------------------------------------------------------
        # Factual grounding metrics — what fraction of recommendations
        # are traceable to an authoritative source?
        # This is computed from the run's provenance records.
        # ---------------------------------------------------------------
        grounded_count = sum(
            1 for step in run.agent_steps
            if step.tool_name in ('qof_lookup', 'usage_frequency_lookup')
            and step.success
        )
        # Proxy: proportion of final codes that appeared in a QOF or
        # usage tool call (rather than only in semantic search)
        # In production: compute from provenance records per code
        grounding_score = min(1.0, grounded_count / max(len(run.agent_steps), 1))
        
        # ---------------------------------------------------------------
        # Composite grade — weighted average of key metrics
        # Weights reflect NICE's stated priorities:
        # Factual grounding (0.35) > RRAD / patient coverage (0.30) >
        # Cohen's Kappa (0.25) > Precision (0.10)
        # ---------------------------------------------------------------
        composite = (0.25 * kappa + 0.30 * rrad + 0.35 * grounding_score + 0.10 * precision)
        
        if composite >= 0.85:
            grade = 'PASS'
        elif composite >= 0.70:
            grade = 'CONDITIONAL_PASS'
        else:
            grade = 'FAIL'
        
        results.append({
            'condition': run.condition,
            'cohen_kappa': round(kappa, 3),
            'recall': round(recall, 3),
            'precision': round(precision, 3),
            'f1': round(f1, 3),
            'rrad_patient_weighted': round(rrad, 3),
            'grounding_score': round(grounding_score, 3),
            'composite_score': round(composite, 3),
            'grade': grade,
            'tp': len(run.true_positives),
            'fn': len(run.false_negatives),
            'fp': len(run.false_positives)
        })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("NICE EVALUATION GRADING RUBRIC — FULL RESULTS")
    print("="*80)
    print(df.to_string(index=False))
    print("\nGrade thresholds: PASS ≥ 0.85 | CONDITIONAL_PASS ≥ 0.70 | FAIL < 0.70")
    
    return df
```

---

## Dimension 2 — Factual Grounding and Hallucination Rate

In the context of NICE, hallucination does not mean the agent inventing fictional patients or describing conditions that don't exist. It means recommending a clinical code that cannot be traced back to any authoritative source — a code that appears in the agent's output because the LLM's parametric knowledge suggested it was plausible, not because any of the retrieval tools actually returned it. This is a critical distinction because the governance requirement is explicit: every recommended code must have a citation.

The hallucination rate is computed by examining the provenance records for each run and counting the proportion of final codes whose evidence chain contains at least one tool-returned source (QOF, Reference Set, OpenCodeCounts) versus codes whose evidence chain only contains the LLM's own reasoning without any grounded source. A well-functioning RAG system should have a near-zero hallucination rate because the agent is constrained by its system prompt to cite sources for everything. If the hallucination rate rises above 5%, it is a signal that the LLM is overriding the grounding constraint — likely because the system prompt is not strong enough, or the retrieval tools are returning too few results and the LLM is filling the gap with parametric knowledge.

The monitoring approach for hallucination is to add a post-processing validation step to every agent run: for each code in the final output, check whether that code's SNOMED identifier appears in at least one tool call output within the audit trace. If it does not, flag it as ungrounded and move it automatically to REVIEW tier, regardless of the confidence tier the agent assigned. This validation can be implemented as a simple set membership check on the structured run log.

---

## Dimension 3 — The PROV-O Audit Report: Making the XAI Output Readable

The PROV-O ontology (W3C standard for provenance tracking) gives us the formal vocabulary to describe the "life story" of each code recommendation in a machine-readable way. But for a NICE analyst reviewing the system's output, what matters is not the formal ontology — it is a human-readable document that answers the question "why did the system recommend this code?" in plain language, with citations.

The XAI report is the human-readable translation of the PROV-O provenance record. It should be structured so that a clinical reviewer can scan it quickly and understand the basis of each recommendation without needing to understand the underlying technical system. The following is the required format, derived from the NICE governance requirements described in `05_auditability_and_monitoring.md`.

```python
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, XSD
from datetime import datetime
import json


PROV = Namespace("http://www.w3.org/ns/prov#")
NICE = Namespace("https://www.nice.org.uk/ontology/codelist#")


def generate_prov_o_record(
        snomed_code: str,
        description: str,
        confidence_tier: str,
        sources: list[dict],
        shap_contributions: dict,
        rationale: str,
        run_id: str,
        query: str) -> tuple[Graph, str]:
    """
    Generate a W3C PROV-O compliant provenance record for a single
    code recommendation, serialised as both an RDF graph (for archiving
    in a knowledge store) and a plain-English XAI report (for analyst review).
    
    The PROV-O record formally captures:
    - Entity: the recommended SNOMED code
    - Activity: the agent run that produced the recommendation
    - Agent: the AI system (model + tools + versions)
    - WasDerivedFrom: the source documents/APIs that grounded the recommendation
    - WasGeneratedBy: the specific tool calls that retrieved the evidence
    
    The plain-English report translates this formal record into a format
    suitable for a NICE clinical governance review.
    """
    # Build the RDF provenance graph
    g = Graph()
    g.bind("prov", PROV)
    g.bind("nice", NICE)
    
    # Define the core entities and activities
    code_entity = NICE[f"code/{snomed_code}"]
    run_activity = NICE[f"run/{run_id}"]
    ai_agent = NICE["agent/clinical_code_recommender_v1"]
    
    # Entity: the SNOMED code being recommended
    g.add((code_entity, RDF.type, PROV.Entity))
    g.add((code_entity, RDFS.label, Literal(description)))
    g.add((code_entity, NICE.snomedCode, Literal(snomed_code)))
    g.add((code_entity, NICE.confidenceTier, Literal(confidence_tier)))
    
    # Activity: the agent run
    g.add((run_activity, RDF.type, PROV.Activity))
    g.add((run_activity, PROV.startedAtTime,
           Literal(datetime.now().isoformat(), datatype=XSD.dateTime)))
    g.add((run_activity, NICE.researchQuestion, Literal(query)))
    
    # Agent: the AI system
    g.add((ai_agent, RDF.type, PROV.SoftwareAgent))
    g.add((ai_agent, RDFS.label, Literal("NICE Clinical Code Recommendation Agent")))
    
    # Provenance chains: link sources to the entity
    for source in sources:
        source_entity = NICE[f"source/{source.get('source_type', 'unknown')}"
                              f"/{source.get('source_version', 'unknown')}"]
        g.add((source_entity, RDF.type, PROV.Entity))
        g.add((source_entity, RDFS.label,
               Literal(source.get('source_type', 'unknown'))))
        g.add((code_entity, PROV.wasDerivedFrom, source_entity))
    
    g.add((code_entity, PROV.wasGeneratedBy, run_activity))
    g.add((run_activity, PROV.wasAssociatedWith, ai_agent))
    
    # Serialise to JSON-LD for archiving
    jsonld_record = g.serialize(format='json-ld')
    
    # -----------------------------------------------------------------------
    # Generate the plain-English XAI report
    # This is what the NICE analyst actually reads during their review.
    # The format is deliberately non-technical: it reads like a reasoned
    # clinical argument rather than a data dump.
    # -----------------------------------------------------------------------
    tier_descriptions = {
        'HIGH': 'Recommended with HIGH confidence. Approved for inclusion pending clinical sign-off.',
        'MEDIUM': 'Recommended with MEDIUM confidence. Requires analyst review before final inclusion.',
        'REVIEW': 'Flagged for mandatory clinical review. Do not include without explicit approval.'
    }
    
    source_summary_lines = []
    for source in sources:
        source_type = source.get('source_type', '')
        if source_type == 'QOF_BUSINESS_RULE':
            source_summary_lines.append(
                f"  ✓ QOF Business Rules v49 2024-25 — Indicator "
                f"{source.get('indicator_id', 'unknown')}: "
                f"\"{source.get('indicator_description', '')}\" "
                f"(Authority: HIGHEST)"
            )
        elif source_type == 'OPENCODECOUNTS':
            count = source.get('annual_count', 0)
            pct = source.get('percentile_rank', 0)
            source_summary_lines.append(
                f"  ✓ NHS England Primary Care Usage Data — "
                f"Recorded {count:,} times annually "
                f"({pct:.0f}th percentile nationally). "
                f"Trend: {source.get('trend', 'unknown').upper()}"
            )
        elif source_type == 'NHS_REFERENCE_SET':
            source_summary_lines.append(
                f"  ✓ NHS England Reference Set — "
                f"{source.get('refset_name', 'unknown')} "
                f"(version: {source.get('source_version', 'unknown')})"
            )
    
    shap_lines = []
    for feat, contribution in sorted(shap_contributions.items(),
                                      key=lambda x: abs(x[1]), reverse=True)[:3]:
        direction = "↑ supports inclusion" if contribution > 0 else "↓ reduces confidence"
        shap_lines.append(f"  {feat}: {contribution:+.3f} ({direction})")
    
    xai_report = f"""
═══════════════════════════════════════════════════════════════════
CLINICAL CODE RECOMMENDATION — XAI AUDIT REPORT
═══════════════════════════════════════════════════════════════════
Code:         {snomed_code}
Description:  {description}
Status:       {confidence_tier}
              {tier_descriptions.get(confidence_tier, '')}

RESEARCH QUESTION:
  {query}

EVIDENCE SOURCES:
{chr(10).join(source_summary_lines) if source_summary_lines else '  No authoritative sources — UNGROUNDED (requires human review)'}

MODEL EXPLANATION (SHAP feature contributions):
{chr(10).join(shap_lines)}

PLAIN-ENGLISH RATIONALE:
  {rationale}

AUDIT REFERENCE:
  Run ID: {run_id}
  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  PROV-O record archived in: outputs/provenance/{run_id}/{snomed_code}.jsonld
═══════════════════════════════════════════════════════════════════
"""
    
    return g, xai_report
```

---

## The Final Backtesting Report: Pulling Everything Together

The final deliverable from the backtesting suite is a structured report that presents findings from all six analytical dimensions in a format suitable for presentation to NICE stakeholders. The report has four sections.

The executive summary presents the grade achieved for each backtested condition (from BT-06 Dimension 1) alongside the single most important finding and the most critical action required before the system is ready for supervised analyst use.

The diagnostic breakdown presents the pipeline attribution data from BT-05, the cluster completeness analysis from BT-03, the temporal validity summary from BT-04, and the structural failure analysis from BT-02, each expressed as a condition-level heat map. These four heat maps together show where the system is strong (green), where it is adequate but needs monitoring (amber), and where it has systematic failures requiring engineering attention (red).

The grounding and auditability section presents the hallucination rate and factual grounding score from BT-06 Dimension 2, alongside three sample XAI reports — one for a HIGH confidence code, one for a MEDIUM confidence code, and one REVIEW code — to demonstrate that the provenance system is producing readable, traceable outputs.

The recommended actions section translates all of the above into a prioritised list of improvements, each with a clear owner (data engineer, ML engineer, clinical informatician, or governance team), a concrete description of the change required, and an expected impact on the evaluation metrics if the change is made.

The grading rubric defines three outcomes. A **PASS** (composite score ≥ 0.85 across all tested conditions) means the system is ready for supervised pilot deployment — analysts can use its outputs as a starting point with mandatory review before finalisation. A **CONDITIONAL PASS** (composite score 0.70-0.85, with no individual condition below 0.60) means the system is ready for internal testing and calibration but not yet for analyst-facing use — further iteration on the pipeline is needed. A **FAIL** means the system has a fundamental gap (typically a QOF data infrastructure problem or a severe hallucination rate) that must be fixed before any deployment, however supervised.

---

*This completes the backtesting documentation suite. Reading order: BT-00 → BT-01 → BT-02 → BT-03 → BT-04 → BT-05 → BT-06. Cross-references to the main project documentation series (00–10) are provided throughout.*
