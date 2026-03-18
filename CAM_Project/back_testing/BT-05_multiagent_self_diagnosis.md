# BT-05 — Multi-Agent Self-Diagnosis: Tracing Failures Through the Pipeline

> **The question this document answers.** When your agent gets a code wrong — misses a code that should be there, or adds one that shouldn't — which agent in the pipeline made the mistake? Was it the Extraction Agent misreading the research question? The QOF lookup returning incomplete results? The Hierarchy Explorer not traversing deeply enough? The scoring function weighting features incorrectly? Without systematic pipeline tracing, debugging is guesswork. With it, debugging becomes a precise forensic exercise.

---

## Why Pipeline Attribution Matters

Consider the difference between two backtesting outcomes. In the first, you run the agent on the obesity-with-hypertension research question and find that it missed fourteen codes from the NICE gold-standard list. In the second, you run the same test and find that it missed fourteen codes — and eleven of them are codes that the QOF lookup failed to return because the hypertension indicator was parsed incorrectly from the Excel workbook, two are bridging concepts that the Hierarchy Explorer failed to surface because it stopped at depth three when it needed to reach depth five, and one is a legitimately ambiguous code that even expert analysts disagreed about. 

The first outcome tells you there's a problem. The second tells you where the problem is, how serious each part of it is, and exactly how to fix it. That level of diagnostic precision is what pipeline attribution provides, and it is why the audit trace system described in `05_auditability_and_monitoring.md` is not just a governance requirement but an active engineering tool during backtesting.

---

## The Structured Backtest Run: Capturing Every Decision

Before you can do pipeline attribution, the agent needs to run in a mode that captures its full decision trace — every tool call, every intermediate result, every piece of reasoning. LangChain's `return_intermediate_steps=True` parameter gives you the raw material for this. The `RunLogger` class described in the auditability document structures it into a clean, queryable format. For backtesting specifically, you need one additional layer: the ability to replay any failed run and inspect exactly which steps produced the wrong result.

The following code implements the backtesting orchestration layer that runs the agent on each gold-standard condition, captures the full trace, compares the output to the NICE list, and builds a per-condition attribution report.

```python
import json
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class AgentStep:
    """
    Represents one step in the agent's execution trace.
    Each time the agent calls a tool, one of these is recorded.
    The combination of tool_name, input, output, and reasoning tells you
    everything you need to know about what the agent was doing and why.
    """
    step_number: int
    tool_name: str
    input_args: dict
    output_summary: str
    codes_returned: list[str]
    agent_reasoning: str
    timestamp: str
    success: bool
    error_message: Optional[str] = None


@dataclass
class BacktestRun:
    """
    The complete record of one agent run for one research question,
    including its final output and the comparison to the gold standard.
    This is the primary unit of analysis in the backtesting framework.
    """
    run_id: str
    condition: str
    research_question: str
    agent_steps: list[AgentStep] = field(default_factory=list)
    final_codes: set = field(default_factory=set)
    nice_gold_codes: set = field(default_factory=set)
    true_positives: set = field(default_factory=set)
    false_negatives: set = field(default_factory=set)
    false_positives: set = field(default_factory=set)
    pipeline_attribution: dict = field(default_factory=dict)


def attribute_errors_to_pipeline_steps(backtest_run: BacktestRun) -> dict:
    """
    For each code that the agent missed (false negative) or incorrectly added
    (false positive), trace it back to the specific pipeline step that was
    responsible for the error.
    
    The attribution logic works by examining which tools were called and what
    they returned, then asking: at which point in the pipeline could this
    specific code have been found or filtered?
    
    A false negative is attributed to the earliest step that should have
    surfaced the code but didn't. A false positive is attributed to the
    latest step that had the opportunity to filter it but didn't.
    
    Returns a dict with error counts attributed to each pipeline stage.
    """
    attribution = {
        'qof_lookup_failures': [],           # FN: code is in QOF but wasn't returned
        'semantic_search_failures': [],       # FN: code should have appeared in semantic search
        'hierarchy_explorer_failures': [],    # FN: code is a child/bridging concept that wasn't traversed to
        'scoring_function_failures': [],      # FN/FP: code was found but scored incorrectly
        'extraction_agent_failures': [],      # FN: wrong condition keywords prevented the search
        'unattributed': []                    # Errors that cannot be clearly attributed
    }
    
    # Build lookup sets for what each tool step returned
    qof_returned_codes = set()
    semantic_returned_codes = set()
    hierarchy_returned_codes = set()
    all_candidate_codes = set()
    
    for step in backtest_run.agent_steps:
        if step.tool_name == 'qof_lookup':
            qof_returned_codes.update(step.codes_returned)
        elif step.tool_name == 'semantic_code_search':
            semantic_returned_codes.update(step.codes_returned)
        elif step.tool_name == 'hierarchy_explorer':
            hierarchy_returned_codes.update(step.codes_returned)
        elif step.tool_name == 'score_and_rank_candidates':
            all_candidate_codes.update(step.codes_returned)
    
    for code in backtest_run.false_negatives:
        # Determine where the breakdown occurred for this missed code.
        # We check each stage in order — the earliest failure is the attribution.
        
        if code in qof_returned_codes and code not in backtest_run.final_codes:
            # The QOF lookup found this code but the scoring function dropped it.
            # This means the composite score for this code fell below the threshold.
            attribution['scoring_function_failures'].append({
                'code': code, 'found_by': 'qof_lookup',
                'diagnosis': 'Found by QOF lookup but dropped by scoring function'
            })
        
        elif code not in qof_returned_codes and code not in semantic_returned_codes:
            # Neither QOF nor semantic search found this code.
            # Check whether it's a QOF code (suggesting a QOF parsing failure)
            # or a non-QOF code (suggesting a semantic search failure).
            # Note: in production, you would cross-reference against the QOF data
            # to determine which type of failure this is.
            attribution['qof_lookup_failures'].append({
                'code': code, 'found_by': 'none',
                'diagnosis': 'Not found by any early-stage retrieval tool'
            })
        
        elif code not in qof_returned_codes and code in semantic_returned_codes:
            # Semantic search found it but QOF didn't return it.
            # If the code is actually in QOF (check manually), this is a QOF failure.
            # If not, it's correct that QOF didn't return it — this is not an error.
            attribution['semantic_search_failures'].append({
                'code': code, 'found_by': 'semantic_search',
                'diagnosis': 'Found by semantic search, check if also in QOF'
            })
        
        elif code not in all_candidate_codes:
            # The code exists in the SNOMED hierarchy near the concepts we searched,
            # but the hierarchy traversal didn't reach it.
            attribution['hierarchy_explorer_failures'].append({
                'code': code, 'found_by': 'none',
                'diagnosis': 'Not found — likely a polyhierarchical bridging concept'
                             ' that requires deeper hierarchy traversal'
            })
        
        else:
            attribution['unattributed'].append({
                'code': code, 'diagnosis': 'Attribution unclear — requires manual investigation'
            })
    
    # Summarise attribution counts for the condition-level report
    summary = {stage: len(errors) for stage, errors in attribution.items()}
    summary['total_false_negatives'] = len(backtest_run.false_negatives)
    
    return {'detailed': attribution, 'summary': summary}


def run_full_backtest_suite(
        agent_executor,
        research_questions: dict[str, str],
        nice_gold_lists: dict[str, set],
        run_logger) -> list[BacktestRun]:
    """
    Execute the agent against every gold-standard condition and collect
    complete BacktestRun records for analysis.
    
    The research_questions dict maps condition name to the research question
    string that was used to produce the NICE gold-standard list —
    this ensures a fair comparison.
    """
    backtest_runs = []
    
    for condition, research_question in research_questions.items():
        print(f"\n{'='*60}")
        print(f"Backtesting: {condition}")
        print(f"Question: {research_question}")
        print('='*60)
        
        run_id = f"backtest_{condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Run the agent with full intermediate step capture
        result = agent_executor.invoke(
            {"input": research_question},
            return_intermediate_steps=True
        )
        
        # Parse intermediate steps into structured AgentStep objects
        agent_steps = []
        for i, (action, observation) in enumerate(result.get('intermediate_steps', [])):
            step = AgentStep(
                step_number=i + 1,
                tool_name=action.tool,
                input_args=action.tool_input if isinstance(action.tool_input, dict)
                           else {'query': action.tool_input},
                output_summary=str(observation)[:200],  # Truncate for storage
                codes_returned=extract_codes_from_output(observation),
                agent_reasoning=action.log[:300] if hasattr(action, 'log') else '',
                timestamp=datetime.now().isoformat(),
                success=True
            )
            agent_steps.append(step)
        
        # Extract final recommended codes from the agent's output
        final_output = result.get('output', '')
        final_codes = extract_codes_from_output(final_output)
        
        gold_codes = nice_gold_lists[condition]
        
        bt_run = BacktestRun(
            run_id=run_id,
            condition=condition,
            research_question=research_question,
            agent_steps=agent_steps,
            final_codes=set(final_codes),
            nice_gold_codes=gold_codes,
            true_positives=set(final_codes) & gold_codes,
            false_negatives=gold_codes - set(final_codes),
            false_positives=set(final_codes) - gold_codes
        )
        
        # Attribute errors to specific pipeline stages
        bt_run.pipeline_attribution = attribute_errors_to_pipeline_steps(bt_run)
        
        recall = len(bt_run.true_positives) / max(len(gold_codes), 1)
        precision = len(bt_run.true_positives) / max(len(set(final_codes)), 1)
        
        print(f"\nResults: Recall={recall:.1%} | Precision={precision:.1%} | "
              f"TP={len(bt_run.true_positives)} | FN={len(bt_run.false_negatives)} | "
              f"FP={len(bt_run.false_positives)}")
        print("Attribution:", bt_run.pipeline_attribution['summary'])
        
        backtest_runs.append(bt_run)
    
    return backtest_runs


def extract_codes_from_output(output) -> list[str]:
    """
    Parse SNOMED codes from an agent tool output or final response.
    Handles both structured JSON outputs and free-text with embedded code IDs.
    SNOMED codes are typically 6-18 digit numeric strings.
    """
    if isinstance(output, str):
        try:
            data = json.loads(output)
            if isinstance(data, list):
                return [str(item.get('snomed_code', '')) for item in data
                        if isinstance(item, dict) and 'snomed_code' in item]
        except (json.JSONDecodeError, TypeError):
            pass
        # Fallback: extract numeric strings that look like SNOMED codes
        import re
        return re.findall(r'\b\d{6,18}\b', output)
    return []


def generate_attribution_report(backtest_runs: list[BacktestRun]) -> pd.DataFrame:
    """
    Aggregate attribution results across all backtest runs into a single
    DataFrame showing where errors are concentrated across the pipeline.
    
    This is the most actionable output from the backtesting suite — it tells
    you directly which component of the pipeline needs the most attention.
    """
    attribution_rows = []
    
    for run in backtest_runs:
        summary = run.pipeline_attribution.get('summary', {})
        attribution_rows.append({
            'condition': run.condition,
            'total_gold_codes': len(run.nice_gold_codes),
            'recall': len(run.true_positives) / max(len(run.nice_gold_codes), 1),
            'precision': len(run.true_positives) / max(len(run.final_codes), 1),
            'qof_lookup_failures': summary.get('qof_lookup_failures', 0),
            'semantic_search_failures': summary.get('semantic_search_failures', 0),
            'hierarchy_failures': summary.get('hierarchy_explorer_failures', 0),
            'scoring_failures': summary.get('scoring_function_failures', 0),
            'unattributed': summary.get('unattributed', 0)
        })
    
    report_df = pd.DataFrame(attribution_rows)
    
    print("\n" + "="*70)
    print("PIPELINE ATTRIBUTION REPORT — BACKTEST RESULTS ACROSS ALL CONDITIONS")
    print("="*70)
    print(report_df.to_string(index=False))
    
    # Identify the dominant failure mode across all conditions
    failure_cols = ['qof_lookup_failures', 'semantic_search_failures',
                    'hierarchy_failures', 'scoring_failures']
    total_failures = report_df[failure_cols].sum()
    dominant_failure = total_failures.idxmax()
    
    print(f"\n>>> DOMINANT FAILURE MODE: {dominant_failure} ({total_failures[dominant_failure]} errors)")
    print(f"    This is where to focus engineering effort for the next iteration.")
    
    return report_df
```

---

## Interpreting the Attribution Report

The attribution report is the bridge between backtesting and iterative improvement. Its value lies not in the overall recall and precision numbers — those tell you how the agent performs but not why — but in the breakdown of failure modes across the pipeline stages. Each column points to a specific component and a specific remediation strategy.

A high count in the **QOF lookup failures** column means the QOF parsing is incomplete — some indicators are not being loaded, or the code-to-indicator mapping is missing rows. The fix is to re-examine the Excel parsing logic in Feature 0.2 of the project plan and validate the row count of the parsed output against the total number of codes in the published Business Rules v49 document.

A high count in the **hierarchy explorer failures** column means the polyhierarchy traversal is not reaching bridging concepts. The fix is to extend the hierarchy tool's search depth and add an explicit ancestor-traversal step that climbs upward from each primary concept to find shared parent nodes, then searches for shared descendants. The BT-02 document's polyhierarchy recall analysis will show you at which in-degree level the failures are concentrated, helping you set the right traversal depth.

A high count in the **scoring function failures** column is a subtler problem — it means the agent is finding the right codes but then discarding them. This typically happens because the weighting of features in the composite score is misaligned: the usage frequency penalty is too aggressive and is filtering out legitimate low-usage codes for rare sub-conditions, or the semantic similarity threshold is too high and is excluding codes that have moderate similarity but strong QOF or reference set backing. Tuning the weights in `config/scoring_weights.yaml` and re-running the backtest is the remedy.

---

*Next: BT-06 — Evaluation Framework, Explainability, and the NICE Grading Rubric.*
