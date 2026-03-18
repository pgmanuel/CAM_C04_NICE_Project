# BT-00 — Backtesting Framework: Master Overview

> **The central question this document answers:** How do we know whether our AI agent is actually good at its job before we put it in front of a NICE analyst? The answer is backtesting — and this document explains what that means in the specific context of clinical code list generation, why it is harder than it sounds, and how the six analytical pillars in this documentation suite each contribute a different type of evidence to the overall answer.

---

## What Backtesting Means in This Context

The word "backtesting" comes from quantitative finance, where it describes the practice of running a trading strategy against historical market data to see how it *would have* performed before risking real money. The logic is: if the strategy works on history, it has at least demonstrated it can handle real-world conditions. If it fails on history, you've learned something important without paying the real-world price of failure.

For the NICE clinical code recommendation project, backtesting means something analogous. We have a set of code lists that NICE analysts have already produced manually — the DAAR_2025_004 series covering obesity, hypertension, type 2 diabetes, dyslipidaemia, and related conditions. These lists represent the ground truth: expert clinical judgment, applied carefully, producing a defensible output. Our question is: if we had *not* had those expert analysts, and had instead run our AI agent on the same research questions, how closely would the agent's output have matched? Where would it have done well? Where would it have fallen short, and why?

This is a far more demanding test than simply checking whether the agent produces outputs that look reasonable. "Looks reasonable" is not good enough for NICE. What we need is a systematic, quantitative, explainable comparison between what the agent recommends and what expert analysts actually decided — across every condition, every code, and every confidence tier.

---

## Why Backtesting Clinical Code Lists Is Harder Than It Looks

In other domains, backtesting is relatively straightforward because there is a single, unambiguous ground truth. A chess engine is either better or worse than human players based on win rates. A fraud detection model is either correctly flagging fraudulent transactions or not. There is a clear, binary notion of correct.

Clinical code list generation does not have this luxury, and understanding why is essential before you design a backtesting framework. There are at least three reasons why the ground truth is inherently fuzzy.

The first is **expert disagreement**. If you gave the same research question to three different NICE analysts independently, they would not produce identical code lists. They would agree on the core codes — the ones that are mandated by QOF, the ones that appear in every reference set, the ones that are unambiguously central to the condition. But for codes at the margins — codes that are related but not definitively required, codes that capture rare clinical sub-types, codes whose inclusion depends on judgement calls about the research question's scope — analysts will differ. The NICE example code lists represent *one* set of expert decisions, not *the* objectively correct answer. A code our agent recommends but that doesn't appear in the NICE list might be a genuine improvement; a code the agent misses might be legitimately optional.

The second is **temporal validity**. The NAAR_2025_004 code lists were produced at a specific point in time against a specific version of QOF and SNOMED. If we are backtesting against them in 2026 using QOF v49 2024-25 data, some codes will have changed status in the intervening period — codes that were correct when the list was built may have been deprecated, and new codes may have been created that would now be more appropriate. Our backtesting framework needs to account for this temporal dimension, not just treat the example lists as timeless ground truth.

The third is **asymmetric error costs**. In a standard classification problem, a false positive (recommending a code that shouldn't be included) and a false negative (missing a code that should be included) are roughly symmetric. In clinical code list generation, they are not. Missing a code that captures 50,000 patients — not including it in a cohort definition — potentially skews an entire NICE analysis and leads to underestimates of disease prevalence or treatment eligibility. Adding a spurious code that brings in a handful of irrelevant patients is a much smaller problem; a human reviewer will likely notice and remove it. Our evaluation metrics must reflect this asymmetry.

---

## The Six Analytical Pillars of Backtesting

The remainder of this documentation suite treats each major analytical technique not as a standalone exercise but as a specific lens for evaluating agent performance. Each pillar addresses a different failure mode and a different question.

**EDA and the Usage Baseline** (BT-01) addresses the question: is the agent recommending codes that people actually use in clinical practice, or is it finding codes that are theoretically correct but clinically dormant? The OpenCodeCounts frequency data is the ground truth for this question. An agent that systematically recommends low-frequency codes is making a type of error that EDA can diagnose precisely and quantify across the backtest corpus.

**Feature Engineering and Structural Mapping** (BT-02) addresses the question: is the agent correctly navigating the SNOMED polyhierarchy, or is it missing codes because it is only traversing one branch of a multi-parent concept? Feature engineering produces the structural metadata — hierarchy depth, in-degree, subtree size — that lets us diagnose whether the agent's failures are concentrated in polyhierarchical bridging concepts, which would indicate a systematic gap in the Hierarchy Explorer tool.

**Unsupervised Clustering for Multimorbidity** (BT-03) addresses the question: is the agent identifying the right clinical clusters for comorbidity conditions, or is it producing code lists that are clinically incoherent — mixing concepts from unrelated disease clusters? The KMeans and community detection results provide an independent reference for what a coherent comorbidity code set should look like, which can be used to audit agent outputs automatically.

**Time Series and Semantic Drift** (BT-04) addresses the question: is the agent aware of when codes are becoming outdated, and does it correctly prefer currently active codes over deprecated ones? Time series backtesting is the only way to detect temporal blind spots — cases where the agent would recommend a code that was perfectly valid two years ago but has since been deprecated or superseded.

**Multi-Agent Workflow Self-Diagnosis** (BT-05) addresses the question: when the agent produces a wrong answer, which specific agent in the pipeline is responsible? The LangGraph state machine and audit trace allow us to replay any failed backtest run and pinpoint exactly which step — extraction, QOF lookup, semantic search, hierarchy exploration, or scoring — produced the error. This turns debugging from guesswork into a systematic process.

**Evaluation Framework, Explainability, and Auditability** (BT-06) is the master grading rubric. It defines how we translate all the evidence from the previous five pillars into a single, structured assessment of agent quality — one that NICE stakeholders can read, understand, and act on.

---

## The Backtesting Data Setup

Before running any backtesting experiment, you need to construct the evaluation dataset carefully. The NICE example code lists (DAAR_2025_004 series) become your **gold standard labels**, but they require careful preparation before use.

The first preparation step is to treat the code lists as a multi-label dataset rather than a single binary problem. Each condition has its own code list, and a code can appear in multiple lists (as we know from the hierarchical clustering analysis showing high overlap between metabolic conditions). For backtesting purposes, the agent is evaluated separately on each condition — its performance on obesity is measured against the obesity list, its performance on hypertension is measured against the hypertension list, and so on. This gives you condition-level performance metrics that reveal whether the agent is systematically better or worse at certain condition types.

The second step is to construct a meaningful negative set — codes that could plausibly have been in the NICE list but were not. This requires the same hard-negative mining strategy described in Phase 4 of the project plan: sample negatives from the same semantic neighbourhood as the positives, not from completely unrelated clinical domains. A system that only distinguishes "obesity codes" from "astrophysics concepts" is not impressive; a system that correctly distinguishes "obesity codes" from "metabolic codes that are related but not appropriate for this specific research question" is genuinely useful.

The third step is to record the QOF and SNOMED version numbers at the time the NICE example lists were produced, and to track any version changes between then and the time of backtesting. Codes that changed status in that window should be handled carefully — they are not reliable ground truth and should be flagged as temporally ambiguous in the evaluation.

---

## Reading Order for This Suite

Work through the documents in numbered order: BT-01 through BT-06. Each document assumes familiarity with the concepts introduced in the earlier ones. The progression moves from the most concrete and empirically grounded analysis (EDA frequency baselines) through increasingly abstract and architectural considerations (multi-agent self-diagnosis) before arriving at the evaluation framework that synthesises everything.

After reading the suite, you should be able to design and run a complete backtesting experiment — produce a condition-level performance report, identify the agent's systematic failure modes, and trace each failure back to a specific component of the pipeline that needs improvement.

---

*Document suite: BT-00 through BT-06. Cross-references to the main project documentation use the format `[see 04_rag_pipeline_deep_dive.md]`.*
