# 08 — Embeddings and Semantic Search: The Mathematical Foundation of Code Discovery

> **This is the engine room of the entire RAG system.** Every time the agent receives a research question and goes looking for relevant clinical codes, it is performing a mathematical operation called semantic similarity search. Understanding *how* this works — from first principles through to implementation — is essential for debugging the system, tuning its performance, and explaining its behaviour to NICE stakeholders. This document builds the concept from the ground up.

---

## Part 1 — Why Keywords Alone Fail in Clinical Text

Before explaining what embeddings *are*, it is worth being clear about what problem they solve. Imagine you ask the system: *"Find codes for a patient who is overweight and has high blood sugar."* A keyword-based search would look for SNOMED code descriptions containing the literal words "overweight" and "high blood sugar." But the clinically correct codes for these concepts are likely described as "obesity" or "body mass index 30+" and "hyperglycaemia" or "type 2 diabetes mellitus" — none of which share keywords with your query.

This is the **vocabulary mismatch problem**, and it is everywhere in clinical informatics. The language used in research questions, the language used in GP clinical notes, the language used in SNOMED concept descriptions, and the language used in QOF business rules are all subtly different dialects of medical English. They mean the same things but say them differently. A system that relies on keyword overlap — including more sophisticated variants like TF-IDF or BM25 (which weight keywords by their rarity) — will systematically miss relevant codes simply because the words don't match, even when the meanings do.

Dense retrieval models, built on transformer neural networks, solve this problem by working with *meaning* rather than *words*. They convert both the query and every code description into vectors — lists of numbers — in a high-dimensional space, where two pieces of text that mean similar things end up as vectors that are geometrically close to each other. The distance in that space is a measure of semantic similarity, not keyword overlap.

---

## Part 2 — What Is a Vector Embedding?

A vector is, at its most basic, an ordered list of numbers. A 2D vector like `[3.0, 7.5]` can be plotted as a point in a two-dimensional plane. A 3D vector like `[3.0, 7.5, -1.2]` can be plotted in three-dimensional space. An embedding vector has many more dimensions — typically 384 or 768 — so we can't visualise it directly, but the geometry still works the same way. Points that are close together in this high-dimensional space represent concepts that are semantically similar.

The "learning" part — how these numbers are chosen — is done by a neural network trained on enormous quantities of text. The model learns to assign numbers to words and sentences such that the geometric relationships in the vector space reflect the semantic relationships in language. Pairs of sentences that mean similar things end up with similar numbers in similar positions. Pairs of sentences that mean unrelated things end up geometrically far apart.

A critical property of good embedding models is that they capture *meaning beyond the literal words*. The sentence "the patient's blood glucose was elevated" and the code description "hyperglycaemia" should produce similar embeddings, even though they share no words, because the model has learned from context that these two phrases refer to the same clinical reality.

---

## Part 3 — Why BioBERT? The Importance of Domain-Specific Pre-Training

Not all embedding models are equally useful for clinical text. General-purpose models like `all-MiniLM-L6-v2` (which is fast and decent for everyday English) are trained primarily on web text, Wikipedia, and books. They have good general language understanding but limited clinical knowledge. When you ask a general-purpose model to embed "hyperglycaemia" and "elevated fasting plasma glucose", it may or may not recognise these as semantically equivalent because these terms are rare in everyday text.

**BioBERT** is a version of BERT (the foundational transformer model from Google) that has been *additionally pre-trained on biomedical literature* — specifically, PubMed abstracts and PMC full-text articles. By training on millions of medical papers, BioBERT develops a rich understanding of clinical language: it knows that "myocardial infarction" and "heart attack" are synonymous, that "T2DM" is an abbreviation for "type 2 diabetes mellitus", and that "BMI 30+" is a way of saying "obese".

The specific model recommended for this project, `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb`, is a BioBERT variant that has been further fine-tuned on natural language inference tasks in scientific and medical domains. This fine-tuning on inference tasks means it is particularly good at recognising entailment — the relationship where one clinical statement implies another. "Patient has type 2 diabetes" implies that "code 44054006 (Type 2 diabetes mellitus)" is relevant, even if the exact phrasing doesn't match. This entailment capability is exactly what you need for query-to-code matching.

**`pritamdeka/S-PubMedBert-MS-MARCO`** is an alternative that has been fine-tuned specifically for semantic search (finding the most relevant document for a query), rather than for general natural language understanding. In practice, this often produces slightly better results for the specific task of "given this research question, which code descriptions are most relevant?" — which is precisely what our vector store retrieval is doing. The evaluation experiment in Feature 2.2 of the project plan (asking a team member to judge the clinical relevance of retrieved neighbours) is how you determine which model is better for your specific data.

---

## Part 4 — Cosine Similarity: The Measurement That Powers Retrieval

Once you have converted a query and a set of code descriptions into vectors, you need a way to measure how similar they are. The standard approach is **cosine similarity**, and it has a mathematical elegance that is worth understanding properly.

The formula, from the research document, is:

$$\text{similarity}(q, c) = \frac{q \cdot c}{\|q\| \cdot \|c\|}$$

Let's unpack each component. The numerator, $q \cdot c$, is the **dot product** of the query vector and the code vector — you multiply corresponding elements and sum the results. The denominator, $\|q\| \cdot \|c\|$, is the product of the two vectors' **magnitudes** (the straight-line distance from the origin to each vector point). Dividing the dot product by the product of magnitudes produces a value between -1 and 1.

Why cosine rather than Euclidean distance (the straight-line distance between two points)? The key insight is that cosine similarity measures the *angle* between two vectors, not the distance between them. This means it is independent of the *magnitude* (length) of the vectors. Two code descriptions, one short ("Obesity") and one long ("Morbid obesity due to excess caloric intake in a patient with inactive lifestyle"), might produce vectors of very different magnitudes simply because one sentence is much longer. If you used Euclidean distance, the longer description might appear artificially different from shorter related descriptions simply because its vector is bigger. Cosine similarity normalises this out — it only cares about the direction the vectors point, which is determined by meaning rather than length.

A cosine similarity of 1.0 means the two vectors point in exactly the same direction — perfect semantic alignment. A cosine similarity of 0.0 means the vectors are perpendicular — the concepts are semantically unrelated in the embedding space. A cosine similarity of -1.0 means the vectors point in opposite directions — the concepts are semantically opposed (though in practice, clinical code embeddings rarely go negative). In practice, a similarity above 0.85 typically indicates a strong semantic match; between 0.65 and 0.85 indicates a moderate match worth considering; below 0.65 suggests weak or incidental similarity.

---

## Part 5 — From Theory to Code: Building and Querying the Vector Store

The following code is a fully annotated implementation of the embedding and retrieval pipeline. Each section includes an explanation of both *what* the code does and *why* that design decision was made.

```python
import numpy as np
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# -----------------------------------------------------------------------
# STEP 1: Choose and Load the Embedding Model
# -----------------------------------------------------------------------
# We load the biomedical-domain model rather than a general-purpose one.
# The first call will download the model (~500MB) from HuggingFace Hub.
# Subsequent calls load from local cache, so the download only happens once.
#
# 'device="cuda"' uses GPU if available — embeddings are ~50x faster on GPU.
# If you don't have a GPU, remove the device argument; it will use CPU.
# -----------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "pritamdeka/S-PubMedBert-MS-MARCO"

model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
# Typically 768 for PubMedBERT-based models


# -----------------------------------------------------------------------
# STEP 2: Prepare Clinical Codes for Embedding
# -----------------------------------------------------------------------
# We embed the code DESCRIPTION rather than the code number itself.
# The description carries the semantic meaning; the number is arbitrary.
#
# Important: augment the description with context before embedding.
# "Obesity" alone is less informative than "Clinical condition: Obesity
# — disorder of body weight regulation". The extra context helps the
# model produce more discriminative embeddings.
# -----------------------------------------------------------------------

def prepare_text_for_embedding(code: str, description: str, source: str) -> str:
    """
    Construct an enriched text representation of a clinical code for embedding.
    The format is chosen to match the style of text the model was trained on.
    
    Research shows that adding a domain prefix like "clinical condition:"
    or "medical diagnosis:" improves retrieval performance on biomedical
    models because it helps the model contextualise the concept correctly.
    """
    return f"Clinical code: {description}. Source: {source}. SNOMED: {code}."


# -----------------------------------------------------------------------
# STEP 3: Batch Embedding — Processing All Codes Efficiently
# -----------------------------------------------------------------------
# We embed in batches because:
# (a) sending all codes at once can exceed GPU memory
# (b) batch processing is much faster than encoding one code at a time
#
# The 'show_progress_bar=True' flag displays a tqdm progress bar — essential
# when you have 50,000+ codes to embed (this will take several minutes).
# -----------------------------------------------------------------------

def embed_code_list(df: pd.DataFrame, 
                    model: SentenceTransformer,
                    batch_size: int = 256) -> np.ndarray:
    """
    Embed all clinical codes in a DataFrame.
    
    Args:
        df: DataFrame with columns ['snomed_code', 'description', 'source']
        model: Loaded SentenceTransformer model
        batch_size: Number of texts to embed at once (tune based on GPU memory)
    
    Returns:
        numpy array of shape (n_codes, embedding_dim)
    """
    texts = [
        prepare_text_for_embedding(row.snomed_code, row.description, row.source)
        for row in df.itertuples()
    ]
    
    # encode() returns a numpy array of shape (n_texts, embedding_dim)
    # normalise_embeddings=True ensures vectors have unit length (magnitude=1),
    # which means dot product and cosine similarity become equivalent —
    # simplifying the similarity computation inside ChromaDB.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # Important: unit-normalise for cosine search
    )
    
    return embeddings


# -----------------------------------------------------------------------
# STEP 4: Store Embeddings in ChromaDB with Rich Metadata
# -----------------------------------------------------------------------
# ChromaDB is a local vector database that handles similarity search.
# We store not just the embedding vector but also structured metadata
# for each code — this metadata is used by the agent's scoring function
# and for filtering results (e.g., "only return codes that are in QOF").
# -----------------------------------------------------------------------

def build_vector_store(df: pd.DataFrame,
                       embeddings: np.ndarray,
                       persist_path: str = "./data/vectorstore") -> chromadb.Collection:
    """
    Persist all embeddings and metadata into a ChromaDB collection.
    
    The collection is stored on disk at persist_path, so it only needs
    to be built once. Subsequent agent runs load from disk in seconds.
    """
    # PersistentClient saves to disk — unlike the in-memory Client(),
    # data survives between Python sessions.
    client = chromadb.PersistentClient(path=persist_path)
    
    # Delete existing collection if we're rebuilding (e.g., after a model update)
    try:
        client.delete_collection("snomed_codes")
    except Exception:
        pass  # Collection didn't exist yet — that's fine
    
    collection = client.create_collection(
        name="snomed_codes",
        # cosine distance = 1 - cosine_similarity, so closer = more similar
        metadata={"hnsw:space": "cosine"}
    )
    
    # ChromaDB requires all fields to be lists and IDs to be strings
    collection.add(
        ids=[str(row.snomed_code) for row in df.itertuples()],
        embeddings=embeddings.tolist(),
        documents=[row.description for row in df.itertuples()],
        metadatas=[
            {
                "snomed_code":    str(row.snomed_code),
                "description":    row.description,
                "in_qof":         bool(row.in_qof),
                "log_usage":      float(row.log_usage) if not pd.isna(row.log_usage) else 0.0,
                "source_count":   int(row.source_count),
                "cluster_id":     int(row.cluster_id) if not pd.isna(row.cluster_id) else -1,
                "deprecated_flag": bool(row.deprecated_flag),
                "usage_trend":    str(row.usage_trend),   # "growing", "stable", "declining"
                "qof_indicators": str(row.qof_indicators) # comma-separated indicator IDs
            }
            for row in df.itertuples()
        ]
    )
    
    print(f"Vector store built: {len(df)} codes stored at {persist_path}")
    return collection


# -----------------------------------------------------------------------
# STEP 5: Querying the Vector Store — Semantic Retrieval in Practice
# -----------------------------------------------------------------------
# This is what the agent calls via the semantic_code_search() tool.
# The query string is embedded with the same model, and ChromaDB returns
# the most semantically similar codes using approximate nearest-neighbour
# search (HNSW algorithm — fast even at 100k+ vectors).
# -----------------------------------------------------------------------

def semantic_search(query: str, 
                    collection: chromadb.Collection,
                    model: SentenceTransformer,
                    top_k: int = 20,
                    qof_only: bool = False) -> list[dict]:
    """
    Search the vector store for clinical codes semantically similar to query.
    
    Args:
        query: Plain-English clinical concept or research question excerpt
        collection: Loaded ChromaDB collection
        model: Same SentenceTransformer used when building the vector store
        top_k: Number of results to return
        qof_only: If True, only return codes that are in QOF (highest authority tier)
    
    Returns:
        List of dicts with code, description, similarity score, and metadata
    """
    # Embed the query with the same model and normalisation as the index
    query_embedding = model.encode(
        [prepare_text_for_embedding("", query, "research_question")],
        normalize_embeddings=True
    ).tolist()
    
    # Build optional metadata filter — ChromaDB supports structured filtering
    # alongside vector search, which is much more efficient than post-filtering
    where_filter = {"in_qof": True} if qof_only else None
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        where=where_filter,
        include=["metadatas", "distances", "documents"]
    )
    
    # Convert to a clean, consistent format for the agent to work with
    # ChromaDB returns cosine DISTANCE (0=identical, 2=opposite), so
    # we convert to similarity (1 = identical, -1 = opposite) for clarity
    output = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        similarity = 1.0 - distance  # Convert distance to similarity
        
        meta = results["metadatas"][0][i]
        output.append({
            "snomed_code":     meta["snomed_code"],
            "description":     meta["description"],
            "similarity":      round(similarity, 4),
            "in_qof":          meta["in_qof"],
            "log_usage":       meta["log_usage"],
            "cluster_id":      meta["cluster_id"],
            "deprecated_flag": meta["deprecated_flag"],
            "usage_trend":     meta["usage_trend"]
        })
    
    return output


# -----------------------------------------------------------------------
# STEP 6: Evaluating Retrieval Quality — The Manual Evaluation Protocol
# -----------------------------------------------------------------------
# Before using the vector store in the agent, we need to know whether
# the retrieval is actually finding clinically relevant codes.
#
# This function implements the manual evaluation from Feature 2.2:
# take known NICE codes, search for them, and check whether results
# are clinically reasonable.
# -----------------------------------------------------------------------

def evaluate_retrieval_on_nice_examples(
        nice_codes_df: pd.DataFrame,
        collection: chromadb.Collection,
        model: SentenceTransformer,
        top_k: int = 20) -> pd.DataFrame:
    """
    For each known NICE gold-standard code, embed its description as a query
    and check whether it retrieves itself and similar codes in the top-K results.
    
    This is 'recall@K' evaluation: what fraction of relevant codes appear
    in the top K results when we query using their own description?
    A well-functioning retrieval system should return a code when you
    search for exactly what that code describes.
    """
    results_log = []
    
    for row in nice_codes_df.itertuples():
        # Use the code's own description as the query
        search_results = semantic_search(
            query=row.description,
            collection=collection,
            model=model,
            top_k=top_k
        )
        
        # Check if the target code appears in the top-K
        retrieved_codes = [r["snomed_code"] for r in search_results]
        target_found = str(row.snomed_code) in retrieved_codes
        
        # Find rank of target code if present
        rank = retrieved_codes.index(str(row.snomed_code)) + 1 if target_found else None
        
        results_log.append({
            "snomed_code":    row.snomed_code,
            "description":    row.description,
            "target_found":   target_found,
            "rank":           rank,
            "top_similarity": search_results[0]["similarity"] if search_results else 0
        })
    
    results_df = pd.DataFrame(results_log)
    recall_at_k = results_df["target_found"].mean()
    mean_rank = results_df.loc[results_df["rank"].notna(), "rank"].mean()
    
    print(f"Recall@{top_k}: {recall_at_k:.2%}")
    print(f"Mean rank of target code (when found): {mean_rank:.1f}")
    
    return results_df
```

---

## Part 6 — Sparse vs. Dense: When to Use Each

Now that you understand dense retrieval, it is worth positioning it against sparse methods (TF-IDF, BM25) so you understand when each is appropriate. The research document describes this trade-off clearly, and it has direct implications for how the agent's search tool works.

**Sparse retrieval** (TF-IDF, BM25) converts text into sparse vectors where each dimension represents a word in the vocabulary. Most dimensions are zero — only the words that actually appear in the text are non-zero. Sparse retrieval is fast, interpretable, and very good when the query uses exactly the same terminology as the document. If a NICE analyst types the exact SNOMED code description, sparse retrieval will find it with near-perfect accuracy. It is also easy to explain: "I found this code because the query contained the words X and Y."

**Dense retrieval** (BioBERT embeddings) converts text into dense vectors where every dimension is non-zero and the values are determined by the model's learned understanding of meaning. Dense retrieval is better when the query and the document use different words to describe the same concept — the vocabulary mismatch problem described at the start of this document. It is also harder to explain, because the numbers in the embedding vector don't have obvious human-interpretable meanings.

The best production systems use a **hybrid approach**: perform both sparse and dense retrieval, merge the results, and re-rank using a cross-encoder model that scores query-document pairs more carefully than either retrieval method alone. For this project at its current stage, dense retrieval alone (BioBERT + ChromaDB) is the right starting point — it addresses the vocabulary mismatch problem that causes the most errors in clinical code discovery, and adding the hybrid layer can be a future iteration once the baseline system is validated.

---

## Part 7 — Context Windows and the Embedding Budget

One practical constraint you will encounter when working with transformer-based embedding models is the **context window limit**. BioBERT and similar models can only process a limited number of tokens (subword units) at once — typically 512 tokens for BERT-based models. A token is roughly three-quarters of a word, so 512 tokens is approximately 380 words.

Most SNOMED code descriptions are short (under 20 words) and will fit comfortably within this window. However, if you enrich the text before embedding (as the `prepare_text_for_embedding()` function above does), make sure the enriched text still fits within 512 tokens. If you ever embed longer clinical documents (such as QOF indicator descriptions or NHS Data Dictionary entries), you will need to either truncate or chunk the text, embedding each chunk separately and aggregating the resulting vectors (typically by averaging them).

---

*Next: See `09_multi_agent_patterns.md` to understand how LangChain and LangGraph orchestrate these components into a working agentic system.*
