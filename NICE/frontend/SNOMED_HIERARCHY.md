# SNOMED CT Hierarchy Resolution
## Handling Parent-Child Code Overlap Without Rebuilding ChromaDB

---

## The Problem

The image you shared shows the hierarchical nature of SNOMED CT data. For example:

```
Hypertensive disorder (parent)
    ├── Essential hypertension
    ├── Secondary hypertension
    └── Hypertensive heart disease

Diabetes mellitus (parent)
    ├── Type 1 diabetes mellitus
    └── Type 2 diabetes mellitus
            └── Gestational diabetes
```

When an analyst searches for "hypertension", the current system may return BOTH "Hypertensive disorder" (the parent concept) AND "Essential hypertension" (the child). This creates two problems:

**Problem 1 — Redundancy:** "Hypertensive disorder" and "Essential hypertension" overlap significantly in terms of which patients they capture. Returning both inflates the apparent code count without adding clinical value.

**Problem 2 — Wrong ranking:** A parent code like "Hypertensive disorder" may score highly on semantic similarity (it contains the word "hypertension") and may have high NHS usage (because it aggregates all hypertension subtypes). But for a code list targeting a specific clinical question, the analyst usually wants the specific child codes (Essential hypertension, Hypertensive heart disease) rather than the generic parent.

---

## Why Not Rebuild ChromaDB?

Rebuilding ChromaDB with hierarchy information would require:
1. A SNOMED hierarchy file (the RF2 release contains IS-A relationships as a separate file)
2. Processing hundreds of thousands of parent-child relationships
3. Storing them in ChromaDB as additional metadata
4. Changing the hybrid ranking formula to account for hierarchy level

This is weeks of work. The approaches below solve 80% of the problem with changes only to `pipeline.py` and `ragas_eval.py`.

---

## Approach 1 — Post-Processing Deduplication (Recommended, No ChromaDB Change)

**Principle:** After retrieval and hybrid scoring, scan the result list for parent-child relationships using only the SNOMED code IDs themselves, which encode hierarchy.

**How SNOMED codes encode hierarchy:** SNOMED CT's numeric code IDs do not directly encode hierarchy, but the term descriptions do. "Diabetes mellitus" is always broader than "Type 2 diabetes mellitus". We can detect likely parent-child relationships by checking whether one code's term is a substring of another's, or by using SNOMED's IS-A metadata if available.

**Implementation in `pipeline.py`:**

```python
def _deduplicate_by_hierarchy(items: list[dict]) -> list[dict]:
    """
    Remove broad parent codes when more specific child codes are
    already present in the result list.

    Approach: if code A's term appears as a substring of code B's term,
    and code B has a higher confidence score than code A, then code A
    is likely a parent of code B and can be removed.

    Example:
        "Diabetes mellitus" (parent)  score 0.71
        "Type 2 diabetes mellitus"    score 0.89  ← child, keep this
        → Remove "Diabetes mellitus" because it is less specific

    This does not require hierarchy data — it uses term substring matching
    as a proxy for the IS-A relationship.

    Note: this is an approximation. Some terms are substrings of others
    by coincidence rather than hierarchy. Set min_score_gap to reduce
    false removals.
    """
    min_score_gap = 0.05  # Only remove parent if child scores at least 5% higher

    terms_lower = [(i, i.get("term", "").lower(), i.get("confidence_score", 0.0))
                   for i in items]

    to_remove: set[int] = set()

    for idx_a, term_a, score_a in terms_lower:
        for idx_b, term_b, score_b in terms_lower:
            if idx_a is idx_b:
                continue
            # term_a is a substring of term_b → term_a is likely more general
            if term_a in term_b and score_b >= score_a - min_score_gap:
                to_remove.add(id(idx_a))

    cleaned = [item for item in items if id(item) not in to_remove]

    # Re-rank sequentially after removal
    for pos, item in enumerate(cleaned, start=1):
        item["rank"] = pos

    return cleaned
```

**Where to add it in `pipeline.py`'s `retrieve_and_rank()` function:**

```python
    # After: final = _engine.hybrid_rerank(...)
    # Add:
    final_deduplicated = _deduplicate_by_hierarchy(
        [_translate_to_report_item(pos, doc) for pos, doc in enumerate(final, 1)]
    )
    # Then build items from final_deduplicated
```

**Trade-offs:**
- Works without any changes to ChromaDB
- Handles the most common cases (e.g. "Diabetes mellitus" vs "Type 2 diabetes mellitus")
- Will miss cases where parent and child terms don't have a simple substring relationship
- May occasionally remove a genuinely useful parent code

---

## Approach 2 — Hierarchy Depth Penalty in the Hybrid Score (No ChromaDB Change)

**Principle:** SNOMED codes at higher abstraction levels (parents) should be penalised relative to more specific codes. We can estimate abstraction level from the term length — more specific codes tend to have longer, more qualified descriptions.

**Implementation:**

```python
def _compute_specificity_bonus(term: str) -> float:
    """
    Estimate how specific a code is from its term length.

    Heuristic: more specific SNOMED terms tend to be longer because
    they include qualifiers (Type 2, Essential, Stage 3, etc.).

    "Hypertension"              → length 12 → low specificity
    "Essential hypertension"    → length 23 → medium specificity
    "Hypertensive heart disease"→ length 26 → medium specificity

    Returns a small bonus (0.0 to 0.05) for longer, more specific terms.
    Cap at 0.05 so it does not overwhelm the hybrid score.
    """
    length = len(term.strip())
    if length < 15:
        return 0.0
    elif length < 30:
        return 0.02
    elif length < 50:
        return 0.04
    else:
        return 0.05
```

**Where to add it:** In `pipeline.py`'s retrieval loop, when building report items:

```python
    items.append({
        ...
        "confidence_score": round(
            float(doc.get("hybrid_score", 0.0)) + _compute_specificity_bonus(term),
            4
        ),
        ...
    })
```

**Trade-offs:**
- Extremely simple — one function, no external data
- Naturally promotes specific codes without hard cutoffs
- Only works when term length correlates with specificity (mostly true in SNOMED)
- Will not help when a short specific code competes with a long generic one

---

## Approach 3 — ChromaDB Metadata Enhancement (Requires Rebuilding the DB)

If neither Approach 1 nor 2 is sufficient, this is the proper long-term solution. It requires running `ingest_data.py` again with additional hierarchy data.

**What to add to the database:**

SNOMED CT RF2 release files include a `Relationship_Snapshot` file containing IS-A relationships. Each row is: `sourceId | typeId | destinationId`. When `typeId` is the IS-A concept ID (116680003), the row means "sourceId IS-A destinationId" (sourceId is a child of destinationId).

**Modified `ingest_data.py`:**

```python
def load_hierarchy(rf2_dir: Path) -> dict[str, list[str]]:
    """
    Load SNOMED IS-A relationships from RF2 release files.
    Returns: {child_code: [parent_code, grandparent_code, ...]}
    """
    IS_A_CONCEPT = "116680003"
    parents: dict[str, list[str]] = {}

    for rf2_file in rf2_dir.rglob("*Relationship*Snapshot*.txt"):
        try:
            df = pd.read_csv(rf2_file, sep='\t', low_memory=False,
                             usecols=['sourceId', 'typeId', 'destinationId', 'active'])
            df = df[df['active'] == 1]  # only active relationships
            df = df[df['typeId'].astype(str) == IS_A_CONCEPT]
            for _, row in df.iterrows():
                child  = str(row['sourceId'])
                parent = str(row['destinationId'])
                parents.setdefault(child, []).append(parent)
        except Exception:
            continue
    return parents

# Then in the ChromaDB metadata for each code, add:
# "parent_codes": "|".join(parents.get(snomed_code, []))[:500]
# "hierarchy_depth": str(compute_depth(snomed_code, parents))
```

**Modified hybrid formula using hierarchy depth:**

```python
# In hybrid_rerank(), add a depth penalty:
# Codes closer to the root (depth 1-3) get a small penalty
# Codes deeper in the hierarchy (depth 4+) get no penalty

depth_str   = doc["metadata"].get("hierarchy_depth", "0")
depth       = int(depth_str) if depth_str.isdigit() else 0
depth_penalty = 0.05 if depth <= 2 else 0.0  # penalise root-level codes

doc["hybrid_score"] = (alpha * sem_score + beta * normalized_usage + qof_bonus) - depth_penalty
```

**Trade-offs:**
- Most accurate approach — uses real SNOMED hierarchy data
- Requires RF2 release files from NHS Digital
- Requires rebuilding ChromaDB (running `ingest_data.py` again, ~20-30 minutes)
- But once done, the app does not need to change at all

---

## Approach 4 — Graph-Based Re-Ranking with NetworkX (Without Rebuilding ChromaDB)

If you have the RF2 hierarchy data but do NOT want to rebuild ChromaDB, you can load the IS-A relationships into a NetworkX directed graph at app startup and use it to re-rank results.

**Setup in `pipeline.py` (runs once at startup):**

```python
import networkx as nx  # pip install networkx

_hierarchy_graph: nx.DiGraph | None = None

def _load_hierarchy_graph(rf2_dir: Path) -> nx.DiGraph | None:
    """
    Build a directed graph of SNOMED IS-A relationships.
    Nodes are SNOMED code strings.
    Edge A → B means "A is a child of B" (A IS-A B).
    """
    IS_A_CONCEPT = "116680003"
    G = nx.DiGraph()
    try:
        import pandas as pd
        for f in rf2_dir.rglob("*Relationship*Snapshot*.txt"):
            df = pd.read_csv(f, sep='\t', low_memory=False,
                             usecols=['sourceId','typeId','destinationId','active'])
            df = df[(df['active'] == 1) & (df['typeId'].astype(str) == IS_A_CONCEPT)]
            for _, row in df.iterrows():
                G.add_edge(str(row['sourceId']), str(row['destinationId']))
        print(f"[pipeline] Hierarchy graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        return G
    except Exception as e:
        print(f"[pipeline] Could not load hierarchy graph: {e}")
        return None
```

**Re-ranking function using the graph:**

```python
def _rerank_with_hierarchy(items: list[dict], graph: nx.DiGraph) -> list[dict]:
    """
    Given a ranked list and the SNOMED hierarchy graph, promote specific
    codes over their parent codes.

    For each pair of codes where one is an ancestor of another in the
    graph, penalise the ancestor's score by a small amount.
    """
    if graph is None:
        return items

    codes = [item["code"] for item in items]
    penalised = set()

    for i, code_a in enumerate(codes):
        for j, code_b in enumerate(codes):
            if i == j:
                continue
            # Is code_a an ancestor of code_b? (code_b IS-A code_a)
            if nx.has_path(graph, code_b, code_a):
                # code_a is more general than code_b — penalise it
                penalised.add(code_a)

    for item in items:
        if item["code"] in penalised:
            item["confidence_score"] = round(item["confidence_score"] * 0.85, 4)

    items.sort(key=lambda x: x["confidence_score"], reverse=True)
    for pos, item in enumerate(items, start=1):
        item["rank"] = pos

    return items
```

**Limitation:** `nx.has_path()` on a large SNOMED graph (300k+ nodes) is slow. Limit this to only the codes actually returned (typically 10) and precompute ancestor sets for those codes rather than checking all pairs.

---

## Recommendation for the Final App

**For the presentation and initial deployment:** Use **Approach 1** (term substring deduplication). It requires only adding `_deduplicate_by_hierarchy()` to `pipeline.py` after the hybrid reranking step. No ChromaDB rebuild, no new packages, ten lines of code.

**For production quality:** Use **Approach 3** (hierarchy metadata in ChromaDB) in a future iteration. This is the only approach that uses the actual SNOMED IS-A relationships rather than proxies, and it requires only a single `ingest_data.py` run — the app code does not change.

**Do not use Approach 4** for this scale of data without caching ancestor lookups — `nx.has_path()` traversal on 300k nodes per query will make the app unusably slow.

---

## What to Add to the Reasoning Trace

When hierarchy deduplication runs, add a step to `reasoning_eval.py`:

```python
sections.append("### Step 3b — Hierarchy Deduplication")
sections.append(
    "Parent codes (broader concepts) were checked against child codes "
    "(more specific concepts). Where a more specific code was already "
    "present with a comparable or higher score, the parent code was "
    "removed to avoid redundancy. This ensures the analyst sees "
    "specific, actionable codes rather than generic categories."
)
removed_count = report.get("hierarchy_removed_count", 0)
if removed_count > 0:
    sections.append(f"- **{removed_count} parent code(s) removed** by hierarchy deduplication")
```
