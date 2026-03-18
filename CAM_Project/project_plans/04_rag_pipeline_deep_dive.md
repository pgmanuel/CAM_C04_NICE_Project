# 04 — The RAG/Agentic Pipeline: How Every Previous Technique Feeds Into It

> **The big picture:** Every technique in this project — correlation analysis, KMeans clustering, t-SNE, logistic regression, SHAP, time series — is not a standalone exercise. Each one is either building a component that the agent uses as a tool, producing metadata that improves the agent's scoring, or validating that the system is working correctly. This document explains exactly how every piece connects.

---

## The Mental Model: A Research Assistant with a Filing System

Before diving into the technical connections, it helps to have a clear mental model of what the final agent is actually doing. Imagine a highly capable research assistant who is tasked with building a clinical code list. They have access to a well-organised filing system containing:

- A comprehensive index of all clinical codes, sorted by meaning (the vector store built from embeddings)
- A highlighted policy manual showing exactly which codes are required by NHS rules (QOF data)
- A usage ledger showing how often each code appears in real patient records (OpenCodeCounts)
- A pre-trained instinct for which codes tend to go together for a given condition (the supervised classifier)
- A map of the clinical code landscape showing which concepts cluster together (KMeans results)
- A flag system marking which codes are trending up or down in usage over time (time series analysis)

The agent doesn't generate code recommendations from thin air. Every recommendation traces back to one of these components. This is what makes the system defensible — there is always a "why" behind every code, and that "why" comes from one of the above sources. Let's walk through exactly how each component was built and how the agent uses it.

---

## How Correlation Analysis Feeds the Pipeline

You might wonder why we spend time on correlation analysis when the final product is an AI agent. The reason is that correlation analysis is how we *validate our features* before the agent uses them.

Recall that in Phase 2, we engineer several numerical features for each SNOMED code: its usage frequency, the number of sources it appears in, whether it is in QOF, its hierarchy depth in SNOMED, and its description length. Before we hand these features to any model, we need to know whether they actually predict whether a code belongs in a NICE code list.

Correlation analysis answers this precisely. We compute the Pearson correlation between each feature and the binary label "included in NICE list" using the DAAR_2025_004 example files as ground truth. What we expect to find — and what we need to confirm — is that `in_qof` has the highest positive correlation with inclusion, followed by `log_frequency`, followed by `source_count`. If the correlations are weak or counterintuitive, that is a signal to rethink the feature engineering before building anything more complex.

There is a second, equally important use of correlation analysis: detecting multicollinearity among the features themselves. If `in_qof` and `log_frequency` are very highly correlated with each other (which they might be, because QOF-mandated codes tend to be frequently used), then feeding both features into a logistic regression or gradient boosting model will cause those models to split their weight between two near-identical features rather than assigning full weight to one clean signal. By catching this early through a correlation heatmap, we can decide to either drop one feature or create a combined feature. The agent's scoring function ultimately uses these features, so getting them right at this stage directly improves the quality of every code recommendation the agent makes.

---

## How Linear and Logistic Regression Feed the Pipeline

Linear regression is the starting point for understanding the quantitative relationship between code features and a continuous outcome — in our case, we can use it to predict `log_frequency` from code metadata, which teaches us which structural properties of a code are associated with mainstream clinical use.

Logistic regression is more directly useful. We train a logistic regression classifier on the NICE example code lists to predict "included vs. not included" from our engineered features. This model, once trained, becomes one of the scoring signals available to the agent. When the agent calls `score_and_rank_candidates()`, part of the composite score comes directly from the logistic regression probability output.

The reason we use logistic regression as a component (rather than a more powerful model) is interpretability. The coefficient for `in_qof` in the logistic regression model has a direct, readable meaning: it tells you exactly how much being a QOF code increases the log-odds of inclusion in a NICE list. This kind of interpretable, quantitative statement is valuable when explaining the agent's decisions to NICE stakeholders. You can say: "The logistic regression component of our scoring assigns 2.4 times more weight to QOF membership than to usage frequency, which reflects the known authority structure of these data sources." That is a defensible, understandable claim.

---

## How KMeans Clustering Feeds the Pipeline

KMeans clustering is perhaps the most directly integrated of all the earlier techniques. To understand why, consider what happens when an analyst asks: *"Build me a code list for obesity with hypertension."* The agent needs to know not just which codes relate to obesity and which relate to hypertension — it also needs to know which codes exist in the *overlap* between the two condition spaces, and whether there are any related condition clusters it should also consider (for example, cardiovascular risk, which is relevant to both obesity and hypertension).

The KMeans clusters computed in Phase 3 are stored as metadata in the vector database alongside each code. Every SNOMED code in our knowledge base has a `cluster_id` field indicating which cluster it belongs to. When the agent retrieves candidate codes through semantic search, it can inspect the cluster distribution of those candidates. If all the high-scoring candidates for "obesity" fall into Cluster 4 (the metabolic disease cluster), but a few are in Cluster 7 (the cardiovascular medication cluster), the agent knows to also query "what other codes are in Cluster 7 that relate to this research question?" This is how KMeans enables the agent to discover codes it might not have found through semantic search alone.

There is also a defensive use of cluster membership. If the agent retrieves a code through semantic search that has high similarity to the research question but falls into a completely unrelated cluster (say, it describes a respiratory condition when we are looking for metabolic codes), the cluster mismatch acts as a flag. The agent will lower the confidence tier for that code and add a note: "Semantic similarity is high but cluster membership suggests potential mismatch — recommend clinical review." This is exactly the kind of uncertainty surfacing that NICE needs.

For the multimorbidity use case specifically, the document we've reviewed from your uploaded research brief references a critically important concept: multimorbidity clusters. Research has shown that chronic conditions form distinct co-occurrence clusters in patient populations — a "cardiovascular cluster" of heart disease, hypertension, and dyslipidaemia; a "metabolic cluster" of obesity, T2DM, and sleep apnoea; and a "mixed cardiometabolic cluster" that overlaps them both. Our KMeans analysis of the SNOMED code embedding space should reproduce a version of these clinical clusters. If it does — and if the boundaries align roughly with what clinical literature says about comorbidity patterns — that is a validation that our embeddings are capturing genuine clinical structure, not just surface-level text similarity.

---

## How t-SNE and UMAP Feed the Pipeline

t-SNE and UMAP are primarily validation and communication tools rather than direct components of the agent's reasoning loop. Their role is to make the embedding space *visible* so that the team and NICE stakeholders can sanity-check it.

After you have run t-SNE on a sample of embedded SNOMED codes, you should be able to point to a 2D scatter plot and say: "Look, here is a dense island of diabetes-related codes. Here is a separate island of cardiovascular codes. Notice how there is a small bridge between them — that is where codes like 'cardiovascular disease in a patient with diabetes' live." If the t-SNE plot looks like random noise with no discernible structure, that tells you your embedding model is not capturing clinical meaning and you need to switch to a different pre-trained model (from `all-MiniLM-L6-v2` to `pritamdeka/S-PubMedBert-MS-MARCO`, for example, which is fine-tuned on biomedical text).

There is also a direct practical output from t-SNE that feeds into the agent: the **neighbourhood relationships** revealed by the low-dimensional projection can be used to identify "boundary codes" — codes that sit between two clinical clusters and could plausibly belong to either. These are exactly the codes that require the highest level of clinical scrutiny, and flagging them proactively is part of our uncertainty surfacing requirement. The agent can query: "Does this code have near-neighbours in multiple clusters in the t-SNE space?" If yes, it's a boundary code, and it gets flagged accordingly.

---

## How Supervised Learning (Random Forest + SHAP) Feeds the Pipeline

The supervised classifier trained in Phase 4 is the most direct contributor to the agent's scoring function. It is trained on the NICE example code lists as labelled data, with all of our engineered features as inputs. For any new SNOMED code the agent is considering, the classifier outputs a probability — something like 0.87 for "this code is likely to be included in a NICE list for condition X."

But the *prediction* is less important than the *explanation*. This is where SHAP values become essential. For each prediction, SHAP decomposes it into the contribution of each feature. A SHAP output for a code might look like this:

- Base rate (average inclusion probability across all codes): 0.12
- `in_qof` = True: +0.45 (this is the strongest push toward inclusion)
- `log_frequency` = 10.2 (high): +0.18
- `cluster_id` = 4 (metabolic cluster, matching our target): +0.11
- `hierarchy_depth` = 5 (mid-range, not too broad or too narrow): +0.04
- `description_length` = 62 (moderate specificity): -0.02
- Final prediction: 0.88

The agent takes this SHAP decomposition and converts it directly into the human-readable rationale that appears in the output: *"Included: mandated by QOF 2024-25 (strongest signal), high national usage frequency (second signal), semantically aligned with target condition cluster (supporting signal)."* Every word of that rationale is grounded in a specific SHAP feature contribution. This is not a narrative the LLM makes up — it is a structured translation of the quantitative SHAP output. This is the technical mechanism behind the system's auditability.

---

## How Time Series Analysis Feeds the Pipeline

Time series analysis produces two types of metadata that the agent uses as warning flags rather than positive signals.

The first type is a **deprecation flag**. When we fit a structural break detection algorithm to a code's usage time series and detect a sudden cliff-drop in usage (say, from 400,000 annual occurrences to near-zero), this is a strong indicator that the code has been deprecated and replaced by a newer code. The agent checks this flag for every candidate code and, if it is raised, adds a warning to the output: *"Usage data shows structural decline consistent with code deprecation — verify whether a successor code exists before including."*

The second type is a **trend signal**. Codes whose usage is growing year-on-year are becoming more mainstream. If the agent is evaluating two semantically similar codes for the same concept — an older code with stable but declining usage and a newer code with rapidly growing usage — the trend signal tells it to prefer the newer code. This is particularly relevant for conditions where clinical practice is evolving, such as obesity pharmacotherapy, where new medication classes (like GLP-1 agonists) are driving the creation and rapid adoption of new clinical codes.

Both of these signals are stored as metadata fields in the vector database alongside each code, computed once during the data preparation phase and updated whenever new NHS SNOMED usage statistics are published. The agent doesn't need to run time series models on the fly — it just reads pre-computed flags.

---

## Bringing It All Together: What Happens When the Agent Runs

To make the connections fully concrete, here is a step-by-step account of a single agent run for the query *"obesity with type 2 diabetes comorbidity"*, showing which component contributes at each step.

The agent begins by calling `qof_lookup("obesity")` and `qof_lookup("type 2 diabetes")`. The data for these calls comes from our parsed QOF Business Rules v49 table. This yields approximately 15–20 anchor codes per condition. These anchor codes immediately get set to HIGH confidence with QOF as the cited source.

Next, the agent calls `semantic_code_search("obesity BMI body weight overweight", top_k=50)`. The vector store, built from sentence-transformer embeddings computed in Phase 2, returns 50 candidate codes ranked by cosine similarity. Each returned code carries its pre-computed cluster ID, SHAP-ready feature vector, and time series flags. The agent reviews the similarity scores and cluster IDs together, discarding any candidates that have a cluster mismatch despite high similarity.

The agent then passes all candidate codes to `score_and_rank_candidates()`. This function runs the supervised classifier from Phase 4, retrieves SHAP values for each prediction, and combines them with usage frequency and source count into a composite score. The output is a ranked list with per-code feature contributions.

Any code with a deprecation flag from the time series analysis is moved to a REVIEW tier with an automatic warning. Any code that sits in the t-SNE boundary zone between two condition clusters is also flagged for review. The remaining codes are assigned HIGH, MEDIUM, or REVIEW confidence based on their composite scores.

Finally, the LLM synthesises the scored list into the structured output, translating SHAP contributions into human-readable rationale for each code. The result is a fully sourced, tiered, explainable code list where every decision traces back to a quantitative signal from one of the six analytical components we built across the project.

---

*Next: See `05_auditability_and_monitoring.md` for the full tracing, logging, and monitoring strategy.*
