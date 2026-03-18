# 00 — NICE Clinical Code Assistant: Plain English Project Brief

> **Who this is for:** Anyone on the team who wants to understand *what* we are building and *why*, without needing to be a clinical informatician or a machine learning engineer on day one.

---

## What Problem Are We Solving?

When NICE wants to answer a healthcare question — for example, *"How many patients in England are obese and also have hypertension?"* — they first have to define exactly what "obese" and "hypertension" mean inside a dataset. Healthcare datasets don't store plain English words. They store thousands of numeric and alphanumeric **clinical codes** from systems like SNOMED CT.

A clinical code is essentially a number (or short code) that stands for a very precise medical concept. For example, SNOMED code `44054006` means "Type 2 diabetes mellitus." But a real analysis might need not just that one code, but 80, 100, or even several hundred related codes to capture every way a condition is recorded across GP surgeries, hospitals, and clinics.

**The current process looks like this:**
A NICE analyst manually searches through reference sources — NHS reference sets, Quality and Outcomes Framework (QOF) business rules, the NHS Data Dictionary — and hand-picks every code they think is relevant. For a single condition this might take days. For a complex condition with multiple co-morbidities (e.g. obesity + type 2 diabetes + hypertension + dyslipidaemia), it can take weeks, and different analysts may produce different lists for the same question.

**The gap we're filling:**  
We want to build a system that, given a plain-English research question, can automatically suggest a defensible, well-sourced list of relevant clinical codes — and explain exactly *why* each code was included. The system needs to be auditable, transparent, and suitable for expert clinical review. It is not replacing the analyst; it is doing the heavy lifting so the analyst can focus on validation and edge cases.

---

## What Does "Defensible" Mean?

This is the most important word in the project brief, and it's worth dwelling on it. NICE's guidance influences NHS spending and patient care, so if a code list is wrong — even subtly — it can lead to:

- **Missed patients** (cohort is too narrow → treatment benefit is underestimated)
- **Misclassified patients** (cohort is too broad → spurious conclusions)
- **Rework and delays** (errors found late in the analysis pipeline → guidance is delayed)
- **Reputational risk** (NICE's credibility depends on methodological rigour)

A "defensible" code recommendation is one that can be traced back to a specific, authoritative source and accompanied by a clear rationale. Think of it like academic referencing: every code we recommend should have a citation and a justification, not just a model's best guess.

---

## The Data We Are Working With

Think of our data in three layers:

**Layer 1 — The "Gold Standard" Policy Rules (QOF Business Rules)**  
These are the official NHS rules that GP practices must follow to receive quality payments. They specify exact codes that *must* be recorded for a condition to count officially. These are our most authoritative source — if a code is in QOF, it is essentially mandatory.

**Layer 2 — The "Evidence of Use" Data (OpenCodeCounts / NHS SNOMED Usage)**  
These datasets tell us how often each code is actually used in primary care. A code that appears in millions of patient records is almost certainly clinically important. A code that has only ever been used three times nationally is likely obscure, deprecated, or a data quality artefact. This usage frequency is our signal for prioritising codes.

**Layer 3 — The "Reference and Context" Sources (NHS Data Dictionary, NHS Reference Sets)**  
These sources provide the definitions, hierarchies, and relationships between codes. They help us understand that code A is a more specific version of code B, which is a child concept within the broader clinical hierarchy.

We also have **example code lists from a real NICE project** (the DAAR_2025_004 series) covering obesity, hypertension, T2DM, dyslipidaemia, and several others. These are our "gold standard" test cases — we can use them to measure whether our system would have produced something similar.

---

## Why Is This a Data Science Project?

At first glance this looks like a search or retrieval problem — just find the right codes. But the real complexity comes from several layers:

**Semantic complexity.** The same clinical concept can be described dozens of different ways across different coding versions, GP systems, and time periods. Fuzzy keyword search is not good enough; we need *semantic* (meaning-based) similarity.

**Scale.** There are hundreds of thousands of SNOMED codes. Brute-force comparison is not feasible. We need dimensionality reduction and smart indexing.

**Uncertainty.** Some codes are clearly in or out. Many sit in a grey zone — they're related but not definitive. Our system needs to surface and communicate this uncertainty rather than hide it behind false confidence.

**Temporal drift.** Code systems are updated periodically. A code list that was correct in 2022 may be subtly wrong in 2025 because some codes were deprecated and replaced. We may be able to model this as a time series problem.

**Auditability.** Every recommendation must carry a reason. This is not just a nice-to-have; it is a hard requirement for NICE stakeholders.

---

## What We Are Building (in Plain English)

At the end of this project, the goal is a system where an analyst can type something like:

> *"I need a code list for patients with obesity who also have type 2 diabetes"*

...and receive a structured output that looks something like:

```
CONDITION: Obesity
  ✓ SNOMED 414916001 — Obesity (BMI 30+)    Source: QOF 2024-25, NHS Ref Set     Confidence: HIGH
  ✓ SNOMED 6077007   — Morbid obesity        Source: NHS England Reference Set    Confidence: HIGH
  ⚠ SNOMED 162864005 — Body mass index 40+  Source: OpenCodeCounts (n=45,000)     Confidence: MEDIUM
    Note: Often used interchangeably with morbid obesity; recommend clinical review

CONDITION: Type 2 Diabetes (comorbidity)
  ✓ SNOMED 44054006  — Type 2 diabetes mellitus   Source: QOF 2024-25           Confidence: HIGH
  ✓ SNOMED 237599002 — T2DM without complication  Source: NHS Reference Set     Confidence: HIGH
  ...
```

This output should be accompanied by a full audit trail showing which sources were searched, what reasoning was applied, and which borderline codes were flagged for human review.

---

## How the Project Phases Fit Together

The data science journey we are taking is deliberately structured so that each phase teaches us something that feeds into the next:

**Exploratory Data Analysis (EDA)** teaches us the shape of the data — how many codes are there, how are they distributed, what do the descriptions look like, where are the gaps and outliers.

**Correlation and Linear Regression** help us understand which features of a code (frequency, hierarchy level, source type) are predictive of whether it belongs in a NICE code list. This builds intuition before we go near a complex model.

**KMeans and t-SNE / UMAP** let us visualise the clinical code space. We can embed all SNOMED codes as vectors (based on their descriptions) and then cluster them — codes about "glucose monitoring" will cluster together; codes about "blood pressure medication" will cluster separately. This is how we discover natural groupings without being told what to look for.

**Supervised Learning** lets us train a classifier to predict "should this code be included for condition X?" using the NICE example code lists as labelled training data.

**The RAG / Agentic Pipeline** brings everything together: a large language model is given tools (vector search, frequency lookup, QOF rule lookup) and orchestrated to build a code list from scratch, explaining every decision in a chain of reasoning.

**Time Series Analysis** can be applied to code usage data over time — understanding whether a code is growing in use (becoming more standard), declining (potentially being replaced), or seasonal.

---

## Success Criteria

We will know the system is working if it can reproduce a significant proportion of the NICE example code lists (the DAAR_2025_004 series) from scratch, and if the codes it recommends but that don't appear in the NICE list can be justified on clinical grounds (i.e. they are genuinely relevant codes that NICE may have missed, rather than irrelevant noise).

The human analyst's job becomes reviewing the output, not generating it.

---

*Next: See `01_data_science_roadmap.md` for the full technical analysis plan.*
