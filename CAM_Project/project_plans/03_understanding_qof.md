# 03 — Understanding QOF: The "Gold Standard" Data Source Explained

> **The single most important thing to understand before building this system.** QOF appears in virtually every design decision we make. This document explains what it is, why it has the authority it does, and exactly how our pipeline uses it.

---

## What Is QOF in Plain English?

The Quality and Outcomes Framework (QOF) is a voluntary annual reward and incentive scheme for GP practices in England. The NHS pays GP practices extra money for meeting defined quality indicators — for example, ensuring that a certain percentage of their patients with Type 2 diabetes have had an HbA1c blood test in the last twelve months.

To claim that payment, the GP practice must prove the patient has the relevant condition *recorded in their system using specific, pre-approved clinical codes*. This is the crucial detail for our project. **QOF doesn't just describe conditions — it mandates which exact SNOMED codes GPs must use to record them**, and it publishes these mandates openly each year in what are called the **QOF Business Rules**.

Think of QOF as the NHS saying: *"These are the codes that count. If you used a different code to record this condition, the patient doesn't appear in our national statistics."* That is an extraordinary level of authority. It means QOF codes are the backbone of primary care data in England — they are the codes that actually appear in patient records at scale, because GPs are financially incentivised to use exactly them and nothing else.

---

## The Structure of QOF Business Rules

Each year, NHS Digital publishes a new version of the QOF Business Rules. For the 2024–25 cycle, this is Business Rules v49. The document is large — it runs to hundreds of pages across multiple Excel workbooks — and it is organised into **clinical domains** and **indicators**.

A domain is a broad clinical area, such as Cardiovascular, Diabetes, Respiratory, or Mental Health. Within each domain there are multiple indicators. An indicator is a specific, measurable quality target. For example:

- **DM012**: The percentage of patients with diabetes in whom the last HbA1c measurement (taken in the 15 months before the achievement date) is 58 mmol/mol or less.

To evaluate whether a patient meets this indicator, NICE analysts need to know: what codes mean "this patient has diabetes"? What codes mean "this patient had an HbA1c test"? What codes capture the actual result value? The business rules specify all of this precisely, and they do so using **Reference Sets** — curated lists of SNOMED codes that have been pre-validated for each indicator.

For our project, the QOF Business Rules serve as the single most authoritative source of "confirmed correct" codes. When our agent retrieves a code from QOF, it assigns it the highest possible confidence tier, because that code has been nationally mandated and is used in real patient records at scale across the NHS.

---

## Why QOF Is Central to the Agentic Pipeline

Our RAG agent is designed around a tiered source hierarchy, and QOF sits at the top of that hierarchy. Here is how this manifests in practice.

When the agent receives a research question — for example, *"Build a code list for patients with obesity who also have hypertension"* — the very first tool it should call is `qof_lookup("obesity")` and `qof_lookup("hypertension")`. The codes returned from this lookup become the **anchor codes**: they are almost certainly in the final list regardless of any other evidence. Everything else the agent finds through semantic search, hierarchy exploration, or usage frequency lookup is evaluated relative to how well it complements or extends these anchor codes.

This matters enormously for defensibility. If a NICE stakeholder challenges why a particular code is in the list, the strongest possible answer is: *"This code is mandated by QOF Business Rules v49 2024–25, Indicator OB001, and it appears in X million GP patient records nationally."* That is an answer that needs no further justification.

Conversely, if the agent recommends a code that does *not* appear in QOF but does appear in an NHS Reference Set and has a high usage frequency, it will be labelled MEDIUM confidence with the explanation: *"Not mandated by QOF but present in NHS England Reference Set and recorded in Y national patient records."* If a code appears in none of the official sources and only has semantic similarity to the research question, it gets flagged for REVIEW with an explicit note that it requires clinical validation.

---

## QOF, Primary Care, and the Obesity Comorbidity Use Case

The NICE project is specifically focused on obesity with comorbidities — this is exactly where QOF becomes complex and fascinating. QOF has a dedicated obesity indicator domain, but when a patient has obesity *alongside* Type 2 diabetes, they also appear in the diabetes domain, the hypertension domain, the lipid domain, and potentially others. This is not duplication — it is the same patient being counted across multiple quality frameworks simultaneously.

This creates what we call **code list overlap**, and it is one of the key challenges the agent must handle. The same SNOMED code for "body mass index 30+" might appear in the QOF obesity rules, the diabetes monitoring rules (because BMI is relevant to diabetes management), and the cardiovascular risk assessment rules. Our agent needs to recognise this and not treat it as a conflict — rather, a code that appears in multiple QOF indicator domains is an even stronger signal that it belongs in a comorbidity code list.

The hierarchical clustering analysis we do in Phase 3 (see `01_data_science_roadmap.md`) will actually reveal this overlap visually: when we cluster the NICE example code lists from the DAAR_2025_004 series, we will see that the hypertension list and the obesity list share a cluster of cardiovascular risk codes. This is not noise — it is clinically meaningful signal that our agent should exploit.

---

## QOF vs. Other Sources: How to Prioritise

To be explicit about the source hierarchy our agent follows, here is the reasoning:

**QOF Business Rules** carry the highest authority because they represent live NHS payment policy — these codes are in use in patient records right now, and they have been agreed by NHS England, clinical advisory groups, and GP federations. They define what "counts" for national monitoring.

**NHS England Reference Sets** are the second tier. These are curated by NHS Digital and go beyond QOF to cover a broader set of operational reporting needs. They include the Primary Care Domain Reference Sets used by the CPRD, EMIS, TPP, and other GP system suppliers. Not everything in a Reference Set is QOF-mandated, but everything has been reviewed by NHS clinical informatics experts.

**OpenCodeCounts Usage Data** is the third tier — it is empirical rather than normative. It tells you not "which codes should be used" but "which codes are actually used". High-frequency codes in OpenCodeCounts that are not in QOF or Reference Sets often represent legitimate clinical practice that falls outside the formal framework — things like locally agreed codes, older code variants that GPs haven't migrated from, or highly specific codes that are technically correct but not required by the framework.

**Semantic similarity search** is the fourth tier — the exploratory layer. It finds codes that are related in meaning to the research question even if they don't appear in any authoritative source. Every code found only through semantic search requires human clinical review before inclusion in a final code list.

---

## A Practical Note: Getting the QOF Data

The QOF Business Rules are publicly available from NHS Digital. The 2024–25 rules are published as Business Rules v49. The data arrives as a set of Excel workbooks, one per clinical domain, each containing multiple tabs for different rule types (cluster rules, exclusion rules, denominator rules, etc.).

The most important tab for our purposes is the one listing **SNOMED reference sets** per indicator — this is what maps indicator codes (like DM012) to the specific SNOMED codes that satisfy them. Parsing this data requires careful Excel handling with `openpyxl` because the structure is complex — codes are often in merged cells, with indicator descriptions in one column and the corresponding SNOMED code list in a separate linked reference set file.

Once parsed into a clean DataFrame with columns `[indicator_id, condition_domain, snomed_code, code_description, refset_id]`, this becomes the most valuable lookup table in our entire system.

---

*Next: See `04_rag_pipeline_deep_dive.md` to understand how QOF, KMeans, correlation, and all previous work converge in the final pipeline.*
