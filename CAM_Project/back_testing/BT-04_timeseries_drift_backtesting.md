# BT-04 — Time Series & Semantic Drift: Backtesting Temporal Validity

> **The question no one wants to ask.** What if our agent produces a code list that was correct in 2022 but is subtly wrong in 2025 because three codes have since been deprecated and replaced with newer equivalents? Without time series backtesting, this failure is invisible — the code list looks complete, the documentation looks thorough, and the error only surfaces when a cohort query returns zero patients for a code that used to be widely used. This document explains how to use temporal analysis as an automated auditor that catches these silent failures before they reach production.

---

## The Nature of Semantic Drift in Clinical Coding

Clinical code systems are not static. The SNOMED CT UK Edition is updated annually — typically in April — with each release adding new concepts, retiring obsolete ones, and sometimes restructuring the hierarchy to reflect advances in clinical understanding. The QOF Business Rules are similarly versioned: each year's rules may add new indicators, modify existing ones, retire old ones, or change the set of SNOMED codes that satisfy each indicator. The NHS England Reference Sets are updated as clinical practice evolves.

This creates what the research document calls "semantic drift" — a gradual, systematic change in the meaning and validity of a clinical code list over time. A code list that was perfectly accurate when produced can drift toward inaccuracy over the following years as its component codes change status in ways that are not immediately visible. The codes themselves don't disappear from the system; they simply become inactive, with patient records no longer being written to them. A query that searches for those codes in a current dataset will find only historical records, not new ones — silently underestimating cohort size.

For NICE, semantic drift is particularly consequential because the guidance NICE produces is intended to be applied for years after it is written. A code list that drifts toward inaccuracy within eighteen months of publication undermines the entire value of the guidance. The time series backtesting framework is the mechanism for detecting drift early, monitoring existing lists proactively, and flagging when an agent-recommended code has usage characteristics consistent with impending deprecation.

---

## The Three Temporal Failure Modes

Understanding how to backtest for temporal validity requires first understanding the three distinct ways that temporal factors can cause a code list to be wrong.

The first failure mode is **deprecated code inclusion** — the agent recommends a code whose usage has already fallen to near-zero nationally because it has been superseded by a newer code. This is a direct accuracy error: the deprecated code will return zero or near-zero patient matches in any current analysis. The time series backtesting check for this is the simplest: run the deprecation detection algorithm (see the code below) on every code the agent recommends, and flag any code where usage has shown a structural break followed by a sustained decline to low levels.

The second failure mode is **emerging code omission** — the agent fails to recommend a recently created code whose usage is growing rapidly. This can happen when the vector store has not been updated since the new code was added to the SNOMED UK Edition, or when the new code's description is sufficiently different from established synonyms that the embedding-based search doesn't surface it. This failure mode is the mirror image of the first: the deprecated code is one the agent wrongly includes; the emerging code is one the agent wrongly omits. Detecting it requires tracking which codes show rapid recent growth and checking whether those growth codes are represented in the agent's outputs.

The third failure mode is **QOF timing mismatch** — the agent was trained or configured against one version of the QOF Business Rules, but the backtest is being evaluated against a different version. In the year when a new QOF indicator is introduced or an existing one is modified, the "correct" code list changes. An agent that was optimised against QOF v48 (2023-24) may systematically miss codes introduced in QOF v49 (2024-25) because its QOF lookup table is stale. This is a data infrastructure failure rather than a modelling failure, but it has the same observable effect.

---

## Time Series Backtesting Implementation

```python
import pandas as pd
import numpy as np
from prophet import Prophet
import ruptures as rpt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')


def backtest_temporal_validity(
        agent_codes: set,
        usage_history_df: pd.DataFrame,
        current_date: str,
        lookback_years: int = 4) -> pd.DataFrame:
    """
    Test every code the agent recommended for temporal validity by fitting
    a time series model to its usage history and classifying its status.
    
    This function is the automated auditor for deprecated code inclusion —
    it runs Prophet on every agent-recommended code and returns a table
    classifying each one as: ACTIVE, DECLINING, DEPRECATED, or EMERGING.
    
    Args:
        agent_codes: Set of SNOMED codes recommended by the agent
        usage_history_df: DataFrame with columns ['snomed_code', 'reporting_date',
                          'usage_count'] — multiple rows per code, one per period
        current_date: The evaluation date (string, e.g. '2025-04-01')
        lookback_years: How many years of history to use for the time series model
    
    Returns:
        DataFrame with one row per code and temporal validity classification
    """
    results = []
    current_dt = pd.to_datetime(current_date)
    cutoff_dt = current_dt - pd.DateOffset(years=lookback_years)
    
    for code in agent_codes:
        code_history = usage_history_df[
            (usage_history_df['snomed_code'] == code) &
            (pd.to_datetime(usage_history_df['reporting_date']) >= cutoff_dt)
        ].sort_values('reporting_date')
        
        # Insufficient history — cannot assess temporal validity
        if len(code_history) < 4:
            results.append({
                'snomed_code': code,
                'temporal_status': 'INSUFFICIENT_HISTORY',
                'latest_usage': 0,
                'trend_slope': 0.0,
                'deprecation_probability': 0.0,
                'action_required': 'VERIFY_MANUALLY'
            })
            continue
        
        # -----------------------------------------------------------------------
        # Change-point detection with ruptures
        # We use the Pelt algorithm (Pruned Exact Linear Time) which finds the
        # optimal number of breakpoints rather than requiring us to specify K.
        # A breakpoint followed by a sustained low-usage plateau is the
        # statistical signature of code deprecation.
        # -----------------------------------------------------------------------
        usage_series = code_history['usage_count'].values
        log_usage_series = np.log1p(usage_series)
        
        algo = rpt.Pelt(model="rbf", min_size=2).fit(log_usage_series)
        breakpoints = algo.predict(pen=3)  # pen controls sensitivity — tune for your data
        
        # Detect the deprecation pattern: breakpoint followed by near-zero usage
        has_deprecation_pattern = False
        deprecation_prob = 0.0
        
        if len(breakpoints) > 1:  # At least one internal breakpoint
            last_breakpoint_idx = breakpoints[-2]  # Last internal breakpoint
            post_break_usage = np.mean(usage_series[last_breakpoint_idx:])
            pre_break_usage = np.mean(usage_series[:last_breakpoint_idx])
            
            if pre_break_usage > 0:
                usage_drop_fraction = 1 - (post_break_usage / pre_break_usage)
                # A >90% drop after a breakpoint is a strong deprecation signal
                if usage_drop_fraction > 0.90:
                    has_deprecation_pattern = True
                    deprecation_prob = min(1.0, usage_drop_fraction)
        
        # -----------------------------------------------------------------------
        # Prophet trend analysis for smooth trend classification
        # We use Prophet's trend component to classify the overall direction,
        # independent of any specific breakpoint structure.
        # -----------------------------------------------------------------------
        prophet_df = code_history[['reporting_date', 'usage_count']].rename(
            columns={'reporting_date': 'ds', 'usage_count': 'y'}
        )
        prophet_df['y'] = np.log1p(prophet_df['y'])
        prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])
        
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                    daily_seasonality=False, changepoint_prior_scale=0.5)
        m.fit(prophet_df)
        
        # Predict one extra period to get a clean trend estimate
        future = m.make_future_dataframe(periods=1, freq='M')
        forecast = m.predict(future)
        
        # Trend slope: change in log usage over the full historical window
        trend_start = forecast['trend'].iloc[0]
        trend_end = forecast['trend'].iloc[-2]  # Exclude the single future period
        trend_slope = (trend_end - trend_start) / len(forecast)
        
        latest_usage = int(usage_series[-1])
        
        # Classify the code's temporal status
        if has_deprecation_pattern and latest_usage < 500:
            status = 'DEPRECATED'
            action = 'EXCLUDE_OR_FIND_SUCCESSOR'
        elif trend_slope < -0.03 and latest_usage < 5000:
            status = 'DECLINING'
            action = 'REVIEW_BEFORE_INCLUSION'
        elif trend_slope > 0.03:
            status = 'EMERGING'
            action = 'CONFIRM_SUCCESSOR_OR_INCLUSION'
        else:
            status = 'ACTIVE'
            action = 'INCLUDE_AS_NORMAL'
        
        results.append({
            'snomed_code': code,
            'temporal_status': status,
            'latest_usage': latest_usage,
            'trend_slope': round(float(trend_slope), 4),
            'deprecation_probability': round(deprecation_prob, 3),
            'action_required': action
        })
    
    results_df = pd.DataFrame(results)
    
    # Summary report
    status_counts = results_df['temporal_status'].value_counts()
    print("\n=== Temporal Validity Backtest Summary ===")
    print(status_counts.to_string())
    
    deprecated_codes = results_df[results_df['temporal_status'] == 'DEPRECATED']
    if not deprecated_codes.empty:
        print(f"\n⚠ CRITICAL: {len(deprecated_codes)} deprecated codes found in agent output.")
        print("  These codes will return near-zero patient matches in current NHS data.")
        print("  Each must be replaced with its active successor code.")
        print(deprecated_codes[['snomed_code', 'latest_usage', 'deprecation_probability']].to_string(index=False))
    
    return results_df


def backtest_emerging_code_coverage(
        nice_codes: set,
        agent_codes: set,
        usage_history_df: pd.DataFrame,
        growth_threshold: float = 0.05) -> dict:
    """
    Find codes that are present in the NICE gold-standard list AND show
    strong recent growth trends — and check whether the agent found them.
    
    Emerging codes are often the hardest for the agent to find because:
    1. They may not yet be in the QOF Business Rules
    2. Their usage counts are still growing and may not yet rank highly
       in frequency-based scoring
    3. Their descriptions may be newer formulations not yet well-represented
       in the embedding model's training data
    
    Identifying which emerging codes the agent misses, and analysing their
    properties, tells you specifically what type of content the current system
    is weakest at discovering.
    """
    missed_emerging = []
    
    nice_only = nice_codes - agent_codes  # Codes the agent failed to find
    
    for code in nice_only:
        code_history = usage_history_df[
            usage_history_df['snomed_code'] == code
        ].sort_values('reporting_date')
        
        if len(code_history) < 4:
            continue
        
        # Simple trend test: is the most recent half of the history
        # substantially higher than the first half?
        mid = len(code_history) // 2
        early_mean = np.mean(code_history['usage_count'].iloc[:mid])
        late_mean = np.mean(code_history['usage_count'].iloc[mid:])
        
        if early_mean > 0:
            growth_rate = (late_mean - early_mean) / early_mean
            if growth_rate > growth_threshold:
                missed_emerging.append({
                    'snomed_code': code,
                    'growth_rate': round(growth_rate, 3),
                    'latest_usage': int(code_history['usage_count'].iloc[-1]),
                    'earliest_usage': int(code_history['usage_count'].iloc[0])
                })
    
    missed_emerging_df = pd.DataFrame(missed_emerging).sort_values(
        'growth_rate', ascending=False
    ) if missed_emerging else pd.DataFrame()
    
    print(f"\n=== Emerging Code Coverage ===")
    print(f"Codes in NICE but not agent: {len(nice_only)}")
    print(f"Of those, strongly growing codes: {len(missed_emerging_df)}")
    
    if not missed_emerging_df.empty:
        print("\n⚠ Agent is missing rapidly growing codes — these represent")
        print("  new clinical practice patterns that the current system is not capturing.")
        print("  Consider updating the vector store and QOF lookup to the latest versions.")
        print(missed_emerging_df.head(10).to_string(index=False))
    
    return {
        'missed_emerging_codes': missed_emerging_df,
        'total_nice_only': len(nice_only),
        'emerging_fraction': len(missed_emerging_df) / max(len(nice_only), 1)
    }


def plot_temporal_backtest_summary(temporal_results: pd.DataFrame) -> None:
    """
    Produce a dashboard-style summary of the temporal backtesting results,
    showing the distribution of temporal status classifications and the
    breakdown of actions required.
    
    This chart is designed to be included in the backtesting report presented
    to NICE stakeholders — it provides an immediate visual sense of how many
    of the agent's recommendations have temporal validity concerns.
    """
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)
    
    # Status distribution pie chart
    ax1 = fig.add_subplot(gs[0])
    status_counts = temporal_results['temporal_status'].value_counts()
    colors = {'ACTIVE': '#2ecc71', 'EMERGING': '#3498db',
              'DECLINING': '#f39c12', 'DEPRECATED': '#e74c3c',
              'INSUFFICIENT_HISTORY': '#95a5a6'}
    pie_colors = [colors.get(s, '#bdc3c7') for s in status_counts.index]
    ax1.pie(status_counts.values, labels=status_counts.index,
            colors=pie_colors, autopct='%1.0f%%', startangle=90)
    ax1.set_title("Temporal Status of Agent-Recommended Codes")
    
    # Action distribution bar chart
    ax2 = fig.add_subplot(gs[1])
    action_counts = temporal_results['action_required'].value_counts()
    bar_colors = ['#e74c3c' if 'EXCLUDE' in a or 'REVIEW' in a else '#2ecc71'
                  for a in action_counts.index]
    ax2.barh(action_counts.index, action_counts.values,
             color=bar_colors, edgecolor='black')
    ax2.set_xlabel("Number of Codes")
    ax2.set_title("Actions Required Based on Temporal Analysis")
    
    plt.suptitle("Temporal Validity Backtest Dashboard", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/backtest_temporal_validity_dashboard.png", dpi=150, bbox_inches='tight')
    plt.show()
```

---

## QOF Version Drift: Testing Across Annual Rule Updates

The QOF timing mismatch failure mode requires a specific backtesting approach that compares code lists generated against different QOF versions. To test for this, you create a version-controlled QOF lookup table that stores the mapping between indicator IDs and their SNOMED reference sets for each annual release (v47, v48, v49, etc.), and then run the agent's QOF lookup step using each version in turn.

The resulting comparison shows you which codes appear in the NICE gold-standard list because they were introduced in a specific QOF release. If the agent only has QOF v48 data but is being backtested against a NICE list that was built using v49, all the codes that were added or modified in the v48→v49 transition will appear as false negatives — the agent "missed" them not because its search failed but because it didn't have the right version of the rules. Quantifying this version-specific failure is important because it tells you whether the issue is a modelling problem (fix the algorithm) or a data infrastructure problem (update the QOF lookup table), which require completely different remediation strategies.

---

*Next: BT-05 — Multi-Agent Workflow Self-Diagnosis.*
