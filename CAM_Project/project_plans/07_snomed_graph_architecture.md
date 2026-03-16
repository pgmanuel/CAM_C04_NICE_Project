# 07 — SNOMED CT as a Graph: Polyhierarchy, DAGs, and NetworkX

> **Why this matters before you write a single line of agent code.** The most common mistake teams make when approaching this project is treating SNOMED CT as a flat lookup table — a dictionary where you search for a word and get a code back. It is not. SNOMED is a *knowledge graph*, and understanding its graph structure is what separates a naive keyword-search system from one that is genuinely comprehensive and defensible. This document teaches you the graph theory you need to work with SNOMED correctly.

---

## Part 1 — What Is a Directed Acyclic Graph (DAG)?

Before we get to SNOMED specifically, it helps to build the concept from the ground up using familiar examples. A **graph** in computer science is simply a set of **nodes** (also called vertices) connected by **edges** (also called links or arcs). A family tree is a graph: each person is a node, and the parent-child relationships are edges.

A **directed** graph is one where every edge has a direction — an arrow pointing from one node to another. In a family tree, the edge points from parent to child, not bidirectionally. This direction matters because it encodes meaning: "A is the parent of B" is a different statement to "B is the parent of A."

An **acyclic** graph is one where you can never follow a sequence of directed edges and return to your starting node. There are no loops. In a family tree, this makes sense biologically — you cannot be your own ancestor. If you start at "Grandmother" and follow the arrows downward, you will never circle back to "Grandmother."

A **Directed Acyclic Graph (DAG)**, therefore, is exactly what you'd expect: a directed graph with no cycles. DAGs appear throughout computer science and data science — they are used to represent dependencies in build systems, execution orders in machine learning pipelines, and causal relationships in probabilistic models. They are the right data structure for knowledge hierarchies because they naturally encode the idea that some concepts are more general (closer to the root) and some are more specific (closer to the leaves).

---

## Part 2 — Why SNOMED CT Is a *Poly*hierarchy

Most of us are familiar with simple hierarchies — a tree structure where every node has exactly one parent. The Linnaean biological classification is a tree: every species belongs to exactly one genus, every genus belongs to exactly one family, and so on. You can ask "what is the parent of this concept?" and get exactly one answer.

SNOMED CT is different. It is a **polyhierarchy**, which means a single concept can have **multiple parent nodes** — multiple different ways of being classified. This is not a bug; it is a deliberate design decision that reflects the genuine complexity of medical knowledge.

Consider the example from the research document: **"Bacterial pneumonia"**. In the SNOMED hierarchy, this concept has two distinct parent classifications simultaneously:

- It is a child of **"Infectious disease"** — because pneumonia caused by bacteria is, by definition, an infection
- It is *also* a child of **"Disorder of lung"** — because pneumonia, regardless of its cause, is a condition that affects the lungs

A tree structure could only place "Bacterial pneumonia" under one parent. SNOMED's DAG structure allows it to be correctly classified under both, because both classifications are clinically true and useful for different purposes. A clinician searching for "all infectious diseases" and a clinician searching for "all lung disorders" should both find "Bacterial pneumonia" in their results — and with a polyhierarchy, they do.

This is why the research document describes SNOMED as a "polyhierarchical directed acyclic graph" rather than a tree. The "directed" part means edges point from parent to child (from general to specific). The "acyclic" part means there are no circular relationships. The "poly" part means a single node can have more than one parent.

---

## Part 3 — Why This Matters for Code List Completeness

The practical consequence of SNOMED's polyhierarchy is profound for our project. When NICE wants to define a code list for "obesity", it cannot simply say "give me the single subtree rooted at the obesity concept." Because obesity-related concepts appear across multiple branches of the SNOMED hierarchy — under metabolic disorders, under nutritional disorders, under cardiovascular risk factors — a simple single-path traversal will miss a significant fraction of the relevant codes.

Furthermore, for the comorbidity use case (obesity with hypertension, obesity with T2DM), the relevant codes are not just the direct children of the target concepts. They also include **bridging concepts** — concepts that are children of *both* relevant parent categories. A code like "obesity hypertension" or "type 2 diabetes mellitus in obese patient" is a child of both the obesity subtree and the hypertension or diabetes subtree respectively. These bridging concepts are exactly the ones that are most relevant for cohort definition — they describe patients who have both conditions recorded simultaneously — and they are the ones most likely to be missed by a system that does not model the polyhierarchy.

This is why the **Hierarchy Explorer** tool in our agent is not just a nice-to-have. It is essential for completeness. When the agent finds a concept through semantic search, it must also traverse the polyhierarchy around that concept — looking at parents, children, and siblings — to ensure that related concepts in other branches of the hierarchy are captured.

---

## Part 4 — Modelling SNOMED with NetworkX

NetworkX is the standard Python library for creating, manipulating, and analysing graph structures. For our purposes, it allows us to model the SNOMED hierarchy as a Python object, navigate its relationships programmatically, and compute graph-theoretic features that become part of the code's feature vector.

Here is how to build and query a simplified SNOMED-style graph. This example uses a small subset of concepts, but the same code scales to the full SNOMED hierarchy.

```python
import networkx as nx

# -----------------------------------------------------------------------
# Building the SNOMED polyhierarchy as a Directed Acyclic Graph
# -----------------------------------------------------------------------
# Nodes represent clinical concepts; edges represent IS-A relationships,
# pointing FROM parent (more general) TO child (more specific).
# In SNOMED's RF2 format, this relationship type has ID 116680003.
# -----------------------------------------------------------------------

G = nx.DiGraph()

# Add nodes with metadata — in practice you'd load these from the SNOMED
# RF2 Concepts table, which has ~350,000 active concepts in the UK edition
G.add_node("44054006",  description="Type 2 diabetes mellitus",             depth=4)
G.add_node("73211009",  description="Diabetes mellitus",                    depth=3)
G.add_node("362969004", description="Disorder of endocrine system",         depth=2)
G.add_node("414916001", description="Obesity",                              depth=3)
G.add_node("238136002", description="Morbid obesity",                       depth=4)
G.add_node("57653000",  description="Obesity hypertension",                 depth=5)
G.add_node("38341003",  description="Hypertension",                         depth=3)
G.add_node("49436004",  description="Disorder of cardiovascular system",    depth=2)

# Add directed edges (parent → child, i.e., general → specific)
# T2DM inherits from the general diabetes concept
G.add_edge("73211009",  "44054006")  # Diabetes mellitus → T2DM
# General diabetes inherits from endocrine disorder
G.add_edge("362969004", "73211009")  # Endocrine → Diabetes
# Obesity inherits from endocrine system AND from nutritional disorder
G.add_edge("362969004", "414916001") # Endocrine → Obesity (one parent)
G.add_edge("238136002", "57653000")  # Morbid obesity → Obesity hypertension
G.add_edge("414916001", "238136002") # Obesity → Morbid obesity
# Obesity hypertension inherits from BOTH obesity AND hypertension
# — this is the polyhierarchy in action
G.add_edge("414916001", "57653000")  # Obesity → Obesity hypertension
G.add_edge("38341003",  "57653000")  # Hypertension → Obesity hypertension
G.add_edge("49436004",  "38341003")  # Cardiovascular → Hypertension


# -----------------------------------------------------------------------
# Traversal: Finding all descendants of a concept
# -----------------------------------------------------------------------
# nx.descendants() returns ALL nodes reachable by following directed edges
# from the given source node. This gives us the complete subtree.
# -----------------------------------------------------------------------

def get_all_subtypes(graph, concept_code):
    """
    Returns all more-specific concepts reachable from concept_code
    by traversing IS-A relationships downward (parent → child direction).
    This is equivalent to 'SNOMED CT Subsumption Query' and is the
    correct way to find all codes that count as a given condition.
    """
    return nx.descendants(graph, concept_code)

obesity_subtypes = get_all_subtypes(G, "414916001")
print("All subtypes of Obesity:", obesity_subtypes)
# → {'238136002', '57653000'}  (Morbid obesity + Obesity hypertension)


# -----------------------------------------------------------------------
# Polyhierarchy check: Finding all parents of a concept
# -----------------------------------------------------------------------
# In a simple tree, every node has one parent.
# In SNOMED's DAG, a node can have multiple. We check for this explicitly.
# -----------------------------------------------------------------------

def get_parents(graph, concept_code):
    """Returns all direct parent concepts (predecessors in the DAG)."""
    return list(graph.predecessors(concept_code))

parents_of_obesity_hypertension = get_parents(G, "57653000")
print("Parents of 'Obesity hypertension':", parents_of_obesity_hypertension)
# → ['238136002', '414916001', '38341003']
# Confirms polyhierarchy: three distinct parent classifications!


# -----------------------------------------------------------------------
# Graph Feature Engineering: Centrality Metrics
# -----------------------------------------------------------------------
# Degree centrality measures how many connections a node has, normalised
# to [0,1]. In SNOMED, high-centrality concepts are "hubs" — concepts
# that are relevant across many different clinical contexts.
# A hub concept is more likely to be a core part of multiple code lists.
# -----------------------------------------------------------------------

centrality = nx.degree_centrality(G)
print("\nDegree centrality (most connected concepts):")
for code, score in sorted(centrality.items(), key=lambda x: x[1], reverse=True):
    desc = G.nodes[code].get('description', code)
    print(f"  {code} ({desc}): {score:.3f}")


# -----------------------------------------------------------------------
# Graph Feature Engineering: Depth (Distance from Root)
# -----------------------------------------------------------------------
# Depth in the SNOMED hierarchy indicates specificity.
# Shallow nodes are broad clinical categories (rarely in code lists).
# Deep nodes are very specific sub-types (may be too granular for some lists).
# Mid-range depth (3-6) is typically the "sweet spot" for NICE code lists.
# -----------------------------------------------------------------------

def get_depth(graph, root_node, target_node):
    """
    Computes shortest path length from root to target.
    In the undirected sense this approximates hierarchy depth.
    """
    try:
        return nx.shortest_path_length(graph, root_node, target_node)
    except nx.NetworkXNoPath:
        return -1  # Not reachable — different branch entirely

# This depth score becomes a feature in our supervised classifier
depth_obesity_hypertension = get_depth(G, "362969004", "57653000")
print(f"\nDepth of 'Obesity hypertension' from root: {depth_obesity_hypertension}")
# → 3 (Endocrine → Obesity → Obesity hypertension)


# -----------------------------------------------------------------------
# Multimorbidity Network: Disease Co-occurrence as a Separate Graph
# -----------------------------------------------------------------------
# This is a different type of graph from the SNOMED hierarchy.
# Here, nodes are CONDITIONS (not individual codes) and edges represent
# how frequently the two conditions co-occur in patient records.
# Edge weights come from the NICE example code list overlap matrix
# computed in Phase 3 (Feature 1.3 in the project plan).
# -----------------------------------------------------------------------

# Build a weighted undirected graph of condition co-occurrence
comorbidity_graph = nx.Graph()
comorbidity_graph.add_edge("obesity", "type_2_diabetes",  weight=0.82)
comorbidity_graph.add_edge("obesity", "hypertension",     weight=0.71)
comorbidity_graph.add_edge("obesity", "dyslipidaemia",    weight=0.65)
comorbidity_graph.add_edge("hypertension", "type_2_diabetes", weight=0.58)
comorbidity_graph.add_edge("hypertension", "dyslipidaemia",   weight=0.61)

# Community detection: which conditions cluster together?
# The Louvain algorithm (via the community package) is preferred for this.
# Here we use nx's built-in greedy modularity as a simple approximation.
communities = nx.algorithms.community.greedy_modularity_communities(comorbidity_graph)
print("\nMultimorbidity communities detected:")
for i, community in enumerate(communities):
    print(f"  Cluster {i+1}: {community}")
```

---

## Part 5 — What Graph Features to Add to the Feature Vector

Now that you understand the graph structure, you can extract graph-theoretic features for each SNOMED code and add them to the feature matrix used by the supervised classifier and the scoring function. Each of these features captures something distinct about a code's structural position in the hierarchy.

**Hierarchy depth** (distance from the SNOMED concept hierarchy root) captures specificity. Very shallow codes are broad categories like "Clinical finding" that no one would put in a targeted code list. Very deep codes are hyper-specific sub-types that may only apply to a handful of patients nationally. Mid-range depth tends to correspond to the clinically actionable level at which NICE works.

**In-degree** (number of parents) directly measures whether a concept is in a polyhierarchical position. A code with in-degree 1 is in a simple hierarchical position. A code with in-degree 3 or more is a bridging concept that appears in multiple clinical contexts — these are disproportionately important for comorbidity code lists because they describe the intersection of two or more conditions.

**Out-degree** (number of children) measures generality at the local level. A code with many children is a broad category; a code with zero children is a leaf node representing a highly specific concept. Both extremes have implications for code list construction: you usually want the concept at the right level of specificity, not its parent category and not its most granular sub-type unless that granularity is clinically justified.

**Betweenness centrality** measures how often a concept sits on the shortest path between other pairs of concepts. High betweenness means the concept is a "connector" — removing it would disconnect parts of the graph. These connector concepts tend to be clinically important bridging terms that are relevant across multiple disease domains.

**Subtree size** (the number of descendants) tells you how many more specific concepts exist beneath this one. A large subtree is a signal to the agent that it should use the Hierarchy Explorer tool to search for relevant children, rather than assuming the parent concept alone is sufficient.

These five graph features, combined with the numerical and embedding features from Phase 2, give the supervised classifier a rich, multi-dimensional view of each code — one that captures not just its clinical meaning but its structural role in the knowledge graph.

---

## Part 6 — New Packages This Introduces

**`networkx>=3.1`** is the standard Python library for graph operations. You need it for building and querying the SNOMED polyhierarchy, computing centrality metrics, running community detection, and building the multimorbidity co-occurrence network. It is pure Python and installs without complications.

**`ruptures>=1.1.9`** (add to requirements) is needed for the change-point detection on time series that detects code deprecation events. It is mentioned in the monitoring document but not yet in the requirements file.

The `community` package provides the Louvain algorithm for community detection and is available as `python-louvain` on PyPI. This is worth adding for the multimorbidity cluster analysis in Feature 3.3.

---

*Next: See `08_embeddings_and_semantic_search.md` to understand how code descriptions are converted into vectors and searched by meaning.*
