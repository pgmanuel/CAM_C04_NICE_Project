# BT-03 — Unsupervised Clustering: Backtesting Multimorbidity Completeness

> **The core question this document answers.** Chronic conditions don't occur independently — obesity clusters with T2DM; hypertension clusters with cardiovascular disease; sleep apnoea clusters with both. If our agent builds a code list for "obesity with hypertension" but misses the cardiovascular cluster that links them, it will produce a cohort that is systematically incomplete in a clinically predictable way. This document explains how KMeans, community detection, and t-SNE become automated completeness auditors during backtesting.

---

## The Multimorbidity Blind Spot

One of the NICE project brief's explicit pain points is that defining cohorts for patients with multiple co-occurring conditions is disproportionately hard — not just twice as hard as defining a single-condition cohort, but multiplicatively harder because the conditions interact. The codes you need are not just the union of "obesity codes" and "hypertension codes"; you also need the codes that exist specifically *at their intersection* — codes for conditions that only arise when both are present, codes for measurements that become clinically relevant only in the comorbid context, and codes for treatment pathways that are triggered specifically by the combination.

The academic literature on multimorbidity (cited in your research document) has established that chronic conditions in NHS primary care data form recognisable clusters — a cardiovascular cluster, a metabolic cluster, a mixed cardiometabolic cluster, and others. These clusters are not arbitrary; they reflect the underlying biology of co-morbid conditions and the clinical workflows that generate the coding patterns. An agent that has genuinely understood the comorbidity structure of NHS primary care data should produce code lists that reflect these known clusters. An agent that has not will produce lists that are either incomplete (missing cluster-adjacent codes) or incoherent (mixing codes from clinically unrelated clusters).

The unsupervised clustering work from Phase 3 of the main project plan — the KMeans analysis, the community detection graph, the UMAP visualisation — is not just exploratory analysis for the team's understanding. It is a reference model of the correct clinical cluster structure that the agent's outputs can be formally tested against during backtesting.

---

## Using KMeans Clusters as a Completeness Reference

The logic here requires some care, so it is worth building the argument step by step. When you run KMeans on the SNOMED embedding matrix, you produce a set of clusters where codes that are semantically similar end up in the same cluster. If your embedding model is well-calibrated for clinical text, these clusters should roughly correspond to clinically coherent concept groupings — a "blood glucose monitoring" cluster, a "cardiovascular medication" cluster, a "body weight measurement" cluster, and so on.

For each NICE condition code list, you can then ask: which clusters are represented, and in what proportions? The obesity code list should draw heavily from the "body weight" cluster, the "metabolic measurement" cluster, and the "endocrine disorder" cluster. A backtest of the agent's obesity code list should show a similar cluster distribution. If the agent's obesity list is heavily weighted toward the metabolic cluster but nearly absent from the body weight measurement cluster, that is a completeness signal: the agent understands the diagnostic codes for obesity but has missed the monitoring and measurement codes, which are also important for cohort definition in an analytical context.

For comorbidity queries, the cluster distribution test becomes even more powerful. The NICE "obesity with hypertension" query should produce a code list that draws from *three* cluster families: the obesity family, the hypertension family, and the cardiovascular bridging cluster that links them. If the agent's output draws from only two cluster families, you know exactly which clinical domain is under-represented, and you can trace that back to a specific retrieval failure.

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def compute_cluster_distribution(
        code_list: set,
        feature_matrix: pd.DataFrame) -> dict[int, int]:
    """
    Given a set of SNOMED codes and a feature matrix that includes
    pre-computed cluster assignments (from the Phase 3 KMeans model),
    return a dictionary counting how many codes fall in each cluster.
    
    This becomes the "cluster fingerprint" of a code list — a vector
    showing which parts of the clinical code space the list draws from.
    """
    codes_in_list = feature_matrix[
        feature_matrix['snomed_code'].isin(code_list)
    ]
    cluster_counts = Counter(codes_in_list['cluster_id'].tolist())
    return dict(cluster_counts)


def backtest_cluster_completeness(
        nice_lists: dict[str, set],
        agent_lists: dict[str, set],
        feature_matrix: pd.DataFrame,
        cluster_labels: dict[int, str]) -> pd.DataFrame:
    """
    For each condition, compare the cluster distribution of the NICE code list
    to the agent's code list. Identifies which clinical clusters the agent
    under-represents (potential completeness gaps) or over-represents
    (potential spurious code inclusions).
    
    The cluster_labels dict maps cluster IDs to human-readable clinical names
    (e.g., {4: "Metabolic & Endocrine Disorders", 7: "Cardiovascular Medication"}).
    These labels come from the cluster characterisation step in Feature 3.1
    of the project plan — you name each cluster based on its most central codes.
    
    Args:
        nice_lists: dict of condition → set of SNOMED codes (gold standard)
        agent_lists: dict of condition → set of SNOMED codes (agent output)
        feature_matrix: DataFrame with 'snomed_code' and 'cluster_id' columns
        cluster_labels: dict mapping cluster_id (int) to descriptive label (str)
    
    Returns:
        DataFrame with under- and over-represented clusters per condition
    """
    results = []
    
    for condition in nice_lists.keys():
        nice_dist = compute_cluster_distribution(nice_lists[condition], feature_matrix)
        agent_dist = compute_cluster_distribution(agent_lists.get(condition, set()), feature_matrix)
        
        # Normalise to proportions so we're comparing relative emphasis,
        # not absolute counts (the two lists may have different total lengths)
        nice_total = sum(nice_dist.values()) or 1
        agent_total = sum(agent_dist.values()) or 1
        
        all_cluster_ids = set(nice_dist.keys()) | set(agent_dist.keys())
        
        for cluster_id in all_cluster_ids:
            nice_prop = nice_dist.get(cluster_id, 0) / nice_total
            agent_prop = agent_dist.get(cluster_id, 0) / agent_total
            discrepancy = agent_prop - nice_prop
            
            results.append({
                'condition': condition,
                'cluster_id': cluster_id,
                'cluster_label': cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
                'nice_proportion': round(nice_prop, 4),
                'agent_proportion': round(agent_prop, 4),
                'discrepancy': round(discrepancy, 4),
                'gap_direction': ('UNDER_REPRESENTED' if discrepancy < -0.05
                                  else 'OVER_REPRESENTED' if discrepancy > 0.05
                                  else 'BALANCED')
            })
    
    results_df = pd.DataFrame(results)
    
    # Visualise as a grouped heatmap of discrepancies
    pivot = results_df.pivot_table(
        index='cluster_label', columns='condition',
        values='discrepancy', aggfunc='mean'
    )
    
    plt.figure(figsize=(max(10, len(nice_lists) * 2), max(8, len(pivot) * 0.6)))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, vmin=-0.3, vmax=0.3,
                linewidths=0.5,
                cbar_kws={'label': 'Discrepancy (Agent - NICE proportion)'})
    plt.title("Cluster Completeness Audit: Agent vs. NICE Gold Standard\n"
              "Green = Agent over-represents this cluster | Red = Agent under-represents",
              fontsize=12)
    plt.xlabel("Condition")
    plt.ylabel("Clinical Cluster")
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    plt.savefig("outputs/backtest_cluster_completeness.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print the most significant gaps for immediate action
    under_rep = results_df[results_df['gap_direction'] == 'UNDER_REPRESENTED'].sort_values('discrepancy')
    if not under_rep.empty:
        print("\n=== TOP UNDER-REPRESENTED CLUSTERS (Agent Completeness Gaps) ===")
        print(under_rep[['condition', 'cluster_label', 'nice_proportion',
                         'agent_proportion', 'discrepancy']].head(10).to_string(index=False))
    
    return results_df


def backtest_multimorbidity_bridge_detection(
        primary_condition_codes: set,
        comorbidity_codes: set,
        agent_comorbidity_codes: set,
        feature_matrix: pd.DataFrame) -> dict:
    """
    Specifically test whether the agent has found the "bridging codes" that
    exist at the intersection of two conditions' cluster families.
    
    Bridging codes are the most important codes for comorbidity cohort definition
    and the hardest for the agent to find, because they may not be semantically
    similar enough to either condition query string in isolation to surface
    in the top-K semantic search results.
    
    A bridging code is defined as one that belongs to a cluster that is
    meaningfully represented in BOTH condition code lists.
    
    Returns a dict identifying the bridge clusters, the expected bridging codes
    from the NICE list, and which ones the agent successfully found.
    """
    primary_cluster_dist = compute_cluster_distribution(primary_condition_codes, feature_matrix)
    comorbidity_cluster_dist = compute_cluster_distribution(comorbidity_codes, feature_matrix)
    
    # Find clusters that appear in both conditions above a minimal threshold
    # — these are the "bridge" clusters
    primary_clusters = {c for c, count in primary_cluster_dist.items() if count >= 2}
    comorbidity_clusters = {c for c, count in comorbidity_cluster_dist.items() if count >= 2}
    bridge_clusters = primary_clusters & comorbidity_clusters
    
    # Identify codes from the NICE comorbidity list that belong to bridge clusters
    comorbidity_df = feature_matrix[feature_matrix['snomed_code'].isin(comorbidity_codes)]
    bridge_codes_nice = set(
        comorbidity_df[comorbidity_df['cluster_id'].isin(bridge_clusters)]['snomed_code']
    )
    
    # Check how many bridge codes the agent found
    bridge_codes_agent = bridge_codes_nice & agent_comorbidity_codes
    bridge_recall = len(bridge_codes_agent) / len(bridge_codes_nice) if bridge_codes_nice else 1.0
    
    print(f"\n=== Multimorbidity Bridge Detection ===")
    print(f"Bridge clusters identified: {bridge_clusters}")
    print(f"Bridging codes in NICE list: {len(bridge_codes_nice)}")
    print(f"Bridging codes found by agent: {len(bridge_codes_agent)}")
    print(f"Bridge code recall: {bridge_recall:.1%}")
    
    if bridge_recall < 0.75:
        print(f"\n⚠ LOW BRIDGE RECALL: The agent is missing {len(bridge_codes_nice) - len(bridge_codes_agent)}"
              f" bridging codes that connect the two conditions.")
        print("  Action: Extend the Hierarchy Explorer to search for concepts in clusters")
        print("  that appear in both conditions' top cluster families.")
    
    return {
        'bridge_clusters': list(bridge_clusters),
        'bridge_codes_nice': bridge_codes_nice,
        'bridge_codes_agent': bridge_codes_agent,
        'bridge_recall': bridge_recall,
        'missed_bridge_codes': bridge_codes_nice - agent_comorbidity_codes
    }
```

---

## Community Detection as an Independent Completeness Auditor

Community detection goes beyond KMeans to find condition clusters at the disease level rather than the code level. The multimorbidity co-occurrence graph (described in `07_snomed_graph_architecture.md`) represents diseases as nodes and their co-occurrence frequency as edge weights. Community detection algorithms applied to this graph will find groups of conditions that tend to occur together in patient populations — the cardiovascular community, the metabolic community, the respiratory community, and so on.

For backtesting, community detection provides a top-level completeness check. When the agent builds a code list for a comorbidity query, you can ask: does the set of conditions the agent's codes implicitly cover match the community structure of the primary condition? A code list for "obesity with hypertension" should include codes from the cardiovascular community that surrounds hypertension, as well as the metabolic community that surrounds obesity. If community detection reveals that the agent's code list only touches the metabolic community and barely touches the cardiovascular community, that is a systematic completeness failure that the EDA frequency analysis alone would not have detected — because the missing cardiovascular community codes might each individually have only moderate usage frequency, making them invisible to frequency-based analysis even though their collective absence represents a major clinical gap.

This multi-level approach to completeness testing — individual code frequency, cluster distribution, and community membership — is what makes the backtesting framework robust. Each level catches a different type of incompleteness, and together they provide a comprehensive picture of where the agent succeeds and where it needs to be improved.

---

*Next: BT-04 — Time Series and Semantic Drift Backtesting.*
