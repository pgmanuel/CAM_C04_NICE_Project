# BT-02 — Feature Engineering & Structural Mapping: Backtesting the Hierarchy

> **The key question this document answers.** The SNOMED polyhierarchy means that a code can be a child of multiple parent concepts simultaneously. When the agent fails to find a code, is that failure random — or is it concentrated in codes that sit at the intersection of two or more condition branches? Feature engineering gives us the tools to answer this precisely.

---

## Why Structural Features Make Backtesting Explainable

There is a difference between knowing that your agent missed a code and knowing *why* it missed that code. The first is just a score. The second is a diagnosis. And a diagnosis is what you need to fix the problem systematically rather than patching individual errors case by case.

Structural features — computed from the shape and position of each code within the SNOMED knowledge graph — are the primary diagnostic tool for understanding *why* a code was missed. Every SNOMED code has a set of structural properties: how deep it sits in the hierarchy (its depth from the root concept), how many parent concepts it has (its in-degree, which identifies polyhierarchical bridging concepts), how many child concepts branch from it (its out-degree, which measures generality), and how large a subtree sits beneath it (its descendant count, which indicates how much semantic territory the concept covers). These properties are not arbitrary — they encode clinically meaningful information about the concept's role in the knowledge graph.

When you compute these features for all codes in the NICE gold-standard lists and then stratify the agent's errors by structural category, consistent patterns emerge that point to specific pipeline failures. Codes with high in-degree (polyhierarchical bridging concepts) that the agent systematically misses tell you the Hierarchy Explorer is not climbing far enough up the graph to find shared ancestors. Codes at extreme depths — either very shallow (broad categories) or very deep (hyper-specific sub-types) — that appear in agent outputs but not in NICE lists tell you the scoring function is not applying sufficient penalty to structurally inappropriate codes.

---

## The Polyhierarchy Failure Mode: Why Bridging Concepts Get Missed

Consider the clinical concept "obesity hypertension" — a code that describes a patient who has both obesity and hypertension recorded together as a combined concept. In the SNOMED polyhierarchy, this code is a child of *both* the obesity branch and the hypertension branch simultaneously. An agent that starts from the obesity concept and traverses only downward through its children will find this code. An agent that starts from the hypertension concept and traverses only downward through *its* children will also find it. But an agent that searches for "obesity" using semantic similarity and for "hypertension" using semantic similarity as two separate queries — without also exploring what concepts they share in the hierarchy — may fail to find this bridging code either time, because its description may not be semantically similar enough to either query string in isolation to surface in the top-K results.

This failure is predictable and detectable. If you compute the in-degree of every code in the NICE gold-standard lists and then look at the agent's recall rate as a function of in-degree, you will often find that recall is near-perfect for in-degree-1 codes (simple hierarchical concepts with one parent) and substantially lower for in-degree-2 or in-degree-3 codes (polyhierarchical bridging concepts with multiple parents). This pattern is the diagnostic signature of an agent whose Hierarchy Explorer is not performing full polyhierarchical traversal.

```python
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


def build_structural_feature_profile(
        graph: nx.DiGraph,
        nice_codes: set,
        agent_codes: set,
        usage_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute structural graph features for every code in the union of the
    NICE and agent code lists, then classify each code by whether the agent
    correctly included it, missed it, or hallucinated it.
    
    The returned DataFrame is the primary input for structural backtesting —
    it lets you ask "are the agent's errors concentrated in codes with specific
    structural properties?"
    """
    all_codes = nice_codes | agent_codes
    feature_rows = []
    
    for code in all_codes:
        if code not in graph.nodes:
            # Code exists in the lists but not in our graph snapshot —
            # this itself is worth flagging (possible new code post-SNOMED-update)
            feature_rows.append({
                'snomed_code': code,
                'in_degree': 0, 'out_degree': 0,
                'subtree_size': 0, 'degree_centrality': 0.0,
                'in_nice': code in nice_codes,
                'in_agent': code in agent_codes,
                'in_graph': False
            })
            continue
        
        feature_rows.append({
            'snomed_code': code,
            # in_degree > 1 = polyhierarchical bridging concept
            'in_degree': graph.in_degree(code),
            # out_degree = 0 means leaf node (most specific sub-type)
            'out_degree': graph.out_degree(code),
            # subtree_size: how many more specific concepts exist below this one
            'subtree_size': len(nx.descendants(graph, code)),
            'degree_centrality': nx.degree_centrality(graph).get(code, 0.0),
            'in_nice': code in nice_codes,
            'in_agent': code in agent_codes,
            'in_graph': True
        })
    
    df = pd.DataFrame(feature_rows)
    
    # Add binary outcome columns for analysis
    df['is_true_positive'] = (df['in_nice'] & df['in_agent']).astype(int)
    df['is_false_negative'] = (df['in_nice'] & ~df['in_agent']).astype(int)
    df['is_false_positive'] = (~df['in_nice'] & df['in_agent']).astype(int)
    
    # Merge in usage data for combined structural-usage analysis
    df = df.merge(usage_df[['snomed_code', 'log_usage']], on='snomed_code', how='left')
    df['log_usage'] = df['log_usage'].fillna(0)
    
    return df


def analyse_polyhierarchy_failure(profile_df: pd.DataFrame) -> None:
    """
    Test the polyhierarchy failure hypothesis: do codes with higher in-degree
    (more parent concepts = bridging concepts) have lower agent recall?
    
    If yes, the Hierarchy Explorer needs to be extended to do full
    polyhierarchical traversal, not just downward traversal from each
    condition's primary concept.
    """
    # Only analyse codes that should be in the list (NICE codes)
    nice_df = profile_df[profile_df['in_nice']].copy()
    
    # Group by in_degree and compute recall at each level
    # Recall = true positives / (true positives + false negatives)
    recall_by_indegree = (
        nice_df.groupby('in_degree')
        .agg(
            total_codes=('in_nice', 'sum'),
            agent_found=('is_true_positive', 'sum')
        )
        .assign(recall=lambda x: x['agent_found'] / x['total_codes'])
        .reset_index()
    )
    
    print("\n=== Recall by SNOMED Polyhierarchy Depth (In-Degree) ===")
    print(recall_by_indegree.to_string(index=False))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of recall by in-degree — the key polyhierarchy diagnostic
    colors = ['#e74c3c' if r < 0.7 else '#f39c12' if r < 0.9 else '#2ecc71'
              for r in recall_by_indegree['recall']]
    ax1.bar(recall_by_indegree['in_degree'].astype(str),
            recall_by_indegree['recall'], color=colors, edgecolor='black')
    ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90% recall target')
    ax1.set_xlabel("Number of Parent Concepts (SNOMED In-Degree)")
    ax1.set_ylabel("Agent Recall Rate")
    ax1.set_title("Agent Recall by Polyhierarchical Complexity\n"
                  "(Lower recall at high in-degree = polyhierarchy failure)")
    ax1.legend()
    ax1.set_ylim(0, 1.05)
    
    # Show count of codes at each in-degree level as context
    ax2.bar(recall_by_indegree['in_degree'].astype(str),
            recall_by_indegree['total_codes'], color='steelblue', edgecolor='black')
    ax2.set_xlabel("Number of Parent Concepts (SNOMED In-Degree)")
    ax2.set_ylabel("Number of Codes in NICE Gold Standard")
    ax2.set_title("Code Count by Polyhierarchical Complexity\n"
                  "(Context: how many codes at each level?)")
    
    plt.tight_layout()
    plt.savefig("outputs/backtest_polyhierarchy_recall.png", dpi=150, bbox_inches='tight')
    plt.show()
    
    # Automated diagnosis
    low_recall_levels = recall_by_indegree[recall_by_indegree['recall'] < 0.80]
    if not low_recall_levels.empty:
        problem_levels = low_recall_levels['in_degree'].tolist()
        print(f"\n⚠ DIAGNOSIS: Agent recall falls below 80% for codes with "
              f"in-degree = {problem_levels}.")
        print("  These are polyhierarchical bridging concepts. The Hierarchy Explorer")
        print("  tool needs to be configured to traverse upward to shared ancestor nodes,")
        print("  not just downward from each primary condition concept.")
    else:
        print("\n✓ PASS: Agent recall is above 80% at all polyhierarchy levels.")
        print("  The Hierarchy Explorer is successfully capturing bridging concepts.")


def analyse_depth_appropriateness(profile_df: pd.DataFrame) -> None:
    """
    Test whether the agent's false positives are concentrated at structurally
    inappropriate depths — either very broad (shallow) concepts that are too
    general for a targeted code list, or very deep (leaf) concepts that are
    hyper-specific and rarely used in practice.
    
    NICE code lists tend to cluster at mid-range depths (3-7 levels from root).
    Systematic errors at depth extremes indicate a scoring function that is not
    penalising structurally inappropriate concepts.
    """
    # Compare depth distribution of true positives vs false positives
    tp_depth = profile_df[profile_df['is_true_positive'] == 1]['subtree_size']
    fp_depth = profile_df[profile_df['is_false_positive'] == 1]['subtree_size']
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log1p(tp_depth), bins=30, alpha=0.6, color='#2ecc71',
            label=f'Correct inclusions (n={len(tp_depth)})', density=True)
    ax.hist(np.log1p(fp_depth), bins=30, alpha=0.6, color='#f39c12',
            label=f'Incorrect additions (n={len(fp_depth)})', density=True)
    ax.set_xlabel("Log Subtree Size (proxy for hierarchical generality)")
    ax.set_ylabel("Density")
    ax.set_title("Structural Depth: Correct vs Incorrect Codes\n"
                 "Right-shifted orange = agent adding overly broad concepts\n"
                 "Left-shifted orange = agent adding overly specific leaf concepts")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/backtest_depth_appropriateness.png", dpi=150, bbox_inches='tight')
    plt.show()
```

---

## Embedding Space Alignment: Backtesting Semantic Feature Quality

The second structural backtesting dimension concerns the embedding space itself. The entire semantic search pipeline relies on the assumption that codes which are clinically relevant to a research question will have embeddings that are geometrically close to the embedded research question. If this assumption fails — if the embedding space is poorly aligned for certain types of clinical concepts — then semantic search will systematically fail for those concept types regardless of how well everything else in the pipeline is designed.

To backtest this assumption, you take each condition from the NICE gold-standard lists and compute two things. First, the cosine similarity between the embedded research question for that condition and the embedded descriptions of every code in the NICE list — this tells you how "visible" the correct codes are to the semantic search tool. Second, the cosine similarity between the embedded research question and every code *not* in the NICE list — this tells you how much "noise" exists in the neighbourhood of the correct codes.

A well-aligned embedding space produces a clear separation: correct codes cluster at high similarity scores (above 0.80), while incorrect codes cluster at lower similarity scores (below 0.65). A poorly aligned space produces extensive overlap — many incorrect codes have similarity scores similar to the correct ones, which means the semantic search tool has no good way of distinguishing them. When you observe this overlap pattern for a specific condition type, the diagnosis is that you need a better or more domain-specific embedding model for that clinical domain.

The model `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` is generally strong for common conditions. For rare or highly specialised conditions — those involving specific drug names, rare metabolic disorders, or laboratory measurement codes — you may find that the general biomedical model loses alignment. Flagging these cases during backtesting tells you where to consider condition-specific fine-tuning.

---

## Interpreting the Structural Backtesting Results

The structural backtesting analysis produces a tiered set of findings that each point to a different level of the pipeline.

A finding that recall drops at in-degree-2 and above (polyhierarchical codes being missed) points to the Hierarchy Explorer tool. The fix is to extend it to perform ancestor traversal — not just looking at children of the target concept, but climbing to its ancestors and then searching for shared descendants that represent bridging concepts.

A finding that false positives cluster at extreme subtree sizes (either very large, indicating overly broad concepts, or very small, indicating hyper-specific leaf concepts) points to the scoring function. The fix is to add a structural appropriateness feature — a penalty term that reduces the composite score for codes whose subtree size places them outside the typical range of NICE code list entries for the target condition type.

A finding that embedding similarity distributions overlap heavily between correct and incorrect codes for certain condition types points to the embedding model or the query construction. The fix is to either switch to a more specialised embedding model for those condition types, or to improve the query text passed to the semantic search tool by adding more clinical synonyms and context.

---

*Next: BT-03 — Unsupervised Clustering for Multimorbidity Backtesting.*
