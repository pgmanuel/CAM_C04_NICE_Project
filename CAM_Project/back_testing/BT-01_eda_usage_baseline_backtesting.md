# BT-01 — EDA & Usage Baseline: Establishing Clinical Ground Truth for Backtesting

> **The core idea in one sentence.** Before you can judge whether your agent is recommending the *right* codes, you need an independent, empirical measure of what "right" looks like in real clinical practice — and the OpenCodeCounts usage frequency data is that measure.

---

## Why Frequency Is a Proxy for Clinical Validity

When a NICE analyst decides to include a SNOMED code in a research code list, they are making a clinical judgment that this code is actually used to record the condition in GP patient records. A code that is theoretically correct — it maps to the right concept in the SNOMED hierarchy — but has never been recorded in a single patient record in England is not a useful code for an analysis of English primary care data. It will match zero patients and contribute nothing to the cohort.

The OpenCodeCounts dataset, published by the Bennett Institute and underpinned by over 62 million patient records from NHS England, tells you exactly how often each SNOMED code is actually used. This is not a theoretical measure — it is the empirical record of what happens when thousands of GPs sit down at their clinical systems and record patient conditions. For backtesting purposes, this creates a powerful test: any code the agent recommends should have a plausible usage footprint, and the distribution of usage frequencies across the agent's recommended codes should resemble the distribution in the NICE gold-standard lists.

Think of it this way. If the NICE analysts, in building their obesity code list, chose codes that are collectively used in tens of millions of records annually — because they chose the most clinically mainstream codes — and your agent produces a list where half the recommended codes have usage in the hundreds or fewer, that is a systematic signal that the agent is finding theoretically plausible but clinically dormant codes. EDA surfaces this pattern immediately and quantitatively.

---

## Building the Usage Baseline: What to Measure

The usage baseline is not a single number but a set of distributions and statistics that characterise the "fingerprint" of a well-formed NICE code list. You compute these statistics on the NICE gold-standard lists and then use them as reference targets during backtesting.

The first statistic is the **median and quartile distribution of log usage frequency** across all codes in each condition's list. Because usage follows a power law (a few codes are used millions of times; most codes are used much less), you work in log-transformed space. A well-formed NICE code list will have a certain characteristic distribution shape — concentrated in the high-to-mid usage range, with a tail of legitimately rare or specialist codes. When you run the same calculation on your agent's output, the shape should roughly match.

The second statistic is the **proportion of codes above the 50th, 75th, and 90th usage percentile** nationally. For a well-formed code list focused on a mainstream condition like obesity or hypertension, you expect a high proportion of codes to be in the top half of national usage. A code list where fewer than 30% of codes exceed the national median usage frequency is a candidate for investigation — either the condition is genuinely rare, or the agent is surfacing obscure codes that don't reflect mainstream clinical practice.

The third, and most diagnostically useful, statistic is the **usage frequency distribution comparison between codes the agent got right (present in both the agent list and the NICE list), codes the agent missed (present in NICE but not agent), and codes the agent hallucinated (present in agent but not NICE)**. This three-way comparison is the core EDA backtesting diagnostic. If the missed codes are systematically high-frequency (the agent failed to find codes that are very commonly used), that is a failure of coverage — the semantic search or QOF lookup is missing widely-used codes. If the hallucinated codes are systematically low-frequency (the agent is recommending codes that are rarely used), that is a failure of precision — the agent is surfacing theoretically related but clinically inactive codes.

---

## Co-occurrence Correlation: Validating Comorbidity Structure

A second EDA backtesting technique examines the correlation structure of the agent's code recommendations across conditions. The logic mirrors the EDA work in Phase 1 of the project plan: when obesity codes are present in a patient record, hypertension codes and T2DM codes tend to also be present. This co-occurrence pattern is a real, measurable property of NHS primary care data, and it creates an independent validation test for the agent's comorbidity handling.

To apply this as a backtesting tool, you take the agent's recommended codes for each condition and compute a pairwise overlap matrix — how many codes does the agent's obesity list share with its hypertension list? How does that compare to the same overlap in the NICE gold-standard lists? If the agent systematically underestimates overlap between metabolically related conditions, it suggests the agent is treating each condition as completely independent when it should be sharing codes across related condition queries, particularly for the bridging concepts that sit at the intersection of two or more conditions in the SNOMED polyhierarchy.

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import jaccard


def compute_overlap_matrix(code_lists: dict[str, set]) -> pd.DataFrame:
    """
    Compute pairwise Jaccard similarity between code lists for different conditions.
    
    Jaccard similarity = |A ∩ B| / |A ∪ B|
    A score of 1.0 means two lists are identical; 0.0 means no overlap at all.
    
    We use Jaccard (rather than raw overlap count) because it normalises for
    list size — a large list sharing 10 codes with a small list is a different
    relationship to two lists of the same size sharing 10 codes.
    
    Args:
        code_lists: dict mapping condition name to set of SNOMED code strings
    
    Returns:
        Square DataFrame where each cell is the Jaccard similarity between
        the two conditions' code lists.
    """
    conditions = list(code_lists.keys())
    n = len(conditions)
    matrix = np.zeros((n, n))
    
    for i, cond_a in enumerate(conditions):
        for j, cond_b in enumerate(conditions):
            set_a = code_lists[cond_a]
            set_b = code_lists[cond_b]
            if not set_a and not set_b:
                matrix[i, j] = 0.0
            else:
                # scipy.jaccard returns *distance* (1 - similarity), so we invert
                matrix[i, j] = 1.0 - jaccard(
                    list(set_a | set_b),
                    list(set_b | set_a)
                )
    
    return pd.DataFrame(matrix, index=conditions, columns=conditions)


def backtest_overlap_structure(
        nice_lists: dict[str, set],
        agent_lists: dict[str, set]) -> dict:
    """
    Compare the pairwise overlap structure of agent code lists to NICE gold standard.
    
    The key diagnostic is whether the agent preserves the correlation structure
    of the NICE lists. Conditions that share many codes in NICE (high Jaccard)
    should also share many codes in the agent output.
    If the agent's obesity-hypertension overlap is far lower than NICE's,
    it is likely missing the cardiovascular bridging codes that link them.
    
    Returns a dict with the two matrices and a discrepancy analysis.
    """
    nice_matrix = compute_overlap_matrix(nice_lists)
    agent_matrix = compute_overlap_matrix(agent_lists)
    
    # Discrepancy matrix — positive = agent has MORE overlap than NICE (over-sharing)
    # negative = agent has LESS overlap than NICE (missing bridging codes)
    discrepancy = agent_matrix - nice_matrix
    
    # Find the most under-shared condition pair — this is your top priority
    np.fill_diagonal(discrepancy.values, np.nan)
    min_idx = np.unravel_index(np.nanargmin(discrepancy.values), discrepancy.shape)
    most_under_shared = (discrepancy.index[min_idx[0]], discrepancy.columns[min_idx[1]])
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for ax, matrix, title in zip(axes,
                                  [nice_matrix, agent_matrix, discrepancy],
                                  ["NICE Gold Standard Overlap",
                                   "Agent Overlap",
                                   "Discrepancy (Agent - NICE)"]):
        sns.heatmap(matrix, annot=True, fmt='.2f', ax=ax,
                    cmap='coolwarm' if 'Discrepancy' in title else 'Blues',
                    vmin=-1 if 'Discrepancy' in title else 0, vmax=1)
        ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig("outputs/backtest_overlap_structure.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    return {
        "nice_overlap_matrix": nice_matrix,
        "agent_overlap_matrix": agent_matrix,
        "discrepancy_matrix": discrepancy,
        "most_under_shared_pair": most_under_shared,
        "most_under_shared_score": float(discrepancy.loc[most_under_shared])
    }


def backtest_usage_frequency_profile(
        nice_codes: set,
        agent_codes: set,
        usage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify agent codes into four categories relative to the NICE gold standard
    and compare their usage frequency distributions.
    
    The four categories are:
    - TRUE POSITIVE:  in both agent and NICE list — correct inclusions
    - FALSE NEGATIVE: in NICE but not agent — codes the agent missed
    - FALSE POSITIVE: in agent but not NICE — codes the agent added
    - TRUE NEGATIVE:  in neither (not directly relevant to this analysis)
    
    The key diagnostic insight: if FALSE NEGATIVES have high usage frequency
    and FALSE POSITIVES have low usage frequency, the agent is missing
    mainstream codes and adding obscure ones — a systematic quality problem.
    If FALSE POSITIVES have HIGH usage frequency, they may be genuine improvements
    to the NICE list that are worth clinical review.
    """
    # Classify each code relative to the NICE gold standard
    usage_df = usage_df.copy()
    usage_df['category'] = 'not_relevant'
    
    usage_df.loc[usage_df['snomed_code'].isin(nice_codes & agent_codes),
                 'category'] = 'TRUE_POSITIVE'
    usage_df.loc[usage_df['snomed_code'].isin(nice_codes - agent_codes),
                 'category'] = 'FALSE_NEGATIVE'
    usage_df.loc[usage_df['snomed_code'].isin(agent_codes - nice_codes),
                 'category'] = 'FALSE_POSITIVE'
    
    relevant = usage_df[usage_df['category'] != 'not_relevant'].copy()
    relevant['log_usage'] = np.log1p(relevant['usage_count'])
    
    # Summary statistics per category — this is the core diagnostic table
    summary = relevant.groupby('category')['log_usage'].agg(['mean', 'median', 'std', 'count'])
    summary.columns = ['Mean Log Usage', 'Median Log Usage', 'Std', 'Code Count']
    
    # Visualise distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    sns.boxplot(data=relevant, x='category', y='log_usage', ax=ax1,
                order=['TRUE_POSITIVE', 'FALSE_NEGATIVE', 'FALSE_POSITIVE'],
                palette={'TRUE_POSITIVE': '#2ecc71',
                         'FALSE_NEGATIVE': '#e74c3c',
                         'FALSE_POSITIVE': '#f39c12'})
    ax1.set_title("Usage Frequency by Classification Category")
    ax1.set_ylabel("Log National Usage Frequency")
    ax1.set_xlabel("Category (relative to NICE gold standard)")
    
    sns.histplot(data=relevant, x='log_usage', hue='category', ax=ax2,
                 kde=True, bins=30, alpha=0.6)
    ax2.set_title("Usage Frequency Distribution by Category")
    ax2.set_xlabel("Log National Usage Frequency")
    
    plt.tight_layout()
    plt.savefig("outputs/backtest_usage_frequency_profile.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n=== Usage Frequency Backtesting Summary ===")
    print(summary.to_string())
    
    # Interpret the results automatically
    fn_median = summary.loc['FALSE_NEGATIVE', 'Median Log Usage'] if 'FALSE_NEGATIVE' in summary.index else 0
    fp_median = summary.loc['FALSE_POSITIVE', 'Median Log Usage'] if 'FALSE_POSITIVE' in summary.index else 0
    tp_median = summary.loc['TRUE_POSITIVE', 'Median Log Usage'] if 'TRUE_POSITIVE' in summary.index else 0
    
    print("\n=== Interpretation ===")
    if fn_median > tp_median:
        print("⚠ WARNING: Missed codes (FALSE NEGATIVE) have HIGHER median usage than correct")
        print("  codes (TRUE POSITIVE). The agent is systematically missing high-frequency,")
        print("  mainstream codes. Check QOF lookup coverage and semantic search recall.")
    if fp_median < tp_median - 1.0:
        print("⚠ WARNING: Added codes (FALSE POSITIVE) have much LOWER median usage than")
        print("  correct codes. The agent is surfacing clinically dormant codes.")
        print("  Consider raising the usage frequency threshold in the scoring function.")
    if fp_median > tp_median:
        print("✓ NOTE: Added codes (FALSE POSITIVE) have HIGHER median usage than the NICE")
        print("  list. These may be genuine improvements worth clinical review — the agent")
        print("  may be finding clinically active codes that NICE missed.")
    
    return summary
```

---

## What to Do With the Results

The EDA backtesting analysis produces four types of actionable insight, each pointing to a specific part of the pipeline that needs attention.

If **false negatives are high-frequency** (the agent is missing commonly-used codes), the problem almost certainly lies in the QOF lookup step or the semantic search step. High-frequency codes are mainstream clinical codes — the QOF lookup should catch the most mandated ones, and the semantic search should catch the rest. Systematic misses here mean either the QOF parsing is incomplete (some indicators are not being loaded correctly) or the embedding model is failing to surface high-frequency codes for certain query phrasings. The fix is to add more query variants to the semantic search step and validate the QOF parsing against the raw source file.

If **false positives are low-frequency** (the agent is recommending obscure codes), the problem usually lies in the scoring function's weighting of usage frequency. The composite score needs to assign more penalty to low-frequency candidates, particularly those that are found only through semantic similarity and not through any official source. Raising the minimum usage threshold for MEDIUM-confidence codes is often sufficient.

If **the overlap structure between conditions is weaker in the agent output than in NICE** (the agent is under-sharing bridging codes between comorbid conditions), the problem lies in the Hierarchy Explorer tool. The agent is not traversing far enough up the SNOMED hierarchy to find the parent concepts that two conditions share, and therefore is not discovering their common child concepts. The fix is to extend the hierarchy exploration depth and add a dedicated "bridging concept search" step for comorbidity queries.

---

*Next: BT-02 — Feature Engineering and Structural Mapping for Backtesting.*
