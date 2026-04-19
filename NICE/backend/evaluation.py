"""Evaluation helpers ported from backend_with_Eval.ipynb.

This module is intentionally not wired into main.py. It provides opt-in
evaluation utilities for retrieval, gate, cross-encoder, final decisions, and
final formatted responses.

All judge-model calls in this module use Nebius's OpenAI-compatible API. The
production pipeline remains local and does not import or call this module.
"""

from __future__ import annotations

import json
import os
from typing import Any

from query_planning import normalize_query


NEBIUS_OPENAI_BASE_URL = "https://api.studio.nebius.ai/v1/"
NEBIUS_API_KEY_ENV_VAR = "NEBIUS_API_KEY"

DEFAULT_EVAL_MODEL = os.environ.get("NEBIUS_EVAL_MODEL", "Qwen/Qwen3-32B")
DEFAULT_RAGAS_MODEL = os.environ.get("NEBIUS_RAGAS_MODEL", "OpenAI/gpt-oss-20b")
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en"


def get_nebius_client():
    api_key = os.environ.get(NEBIUS_API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            "Evaluation requires Nebius credentials. "
            f"Set {NEBIUS_API_KEY_ENV_VAR} in your environment before running "
            "custom eval or RAGAS evaluation."
        )

    from openai import OpenAI

    return OpenAI(
        base_url=NEBIUS_OPENAI_BASE_URL,
        api_key=api_key,
    )


def _strip_json_fences(raw_text: str) -> str:
    raw_text = raw_text.strip()
    if raw_text.startswith("```json"):
        return raw_text.replace("```json", "").replace("```", "").strip()
    if raw_text.startswith("```"):
        return raw_text.replace("```", "").strip()
    return raw_text


def _conditions_from_structured_query(structured_query: dict[str, Any]) -> list[str]:
    raw_conditions = [structured_query.get("primary_condition", "")]
    raw_conditions.extend(structured_query.get("secondary_conditions", []))

    conditions: list[str] = []
    seen: set[str] = set()
    for condition in raw_conditions:
        text = normalize_query(condition)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            conditions.append(text)
    return conditions


def _judge_json(prompt: str, eval_model: str) -> dict[str, Any]:
    client = get_nebius_client()
    response = client.chat.completions.create(
        model=eval_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw_text = _strip_json_fences(response.choices[0].message.content)
    return json.loads(raw_text)


def build_retrieval_eval_rows(
    search_jobs: list[dict[str, Any]],
    retrieval_batches: list[list[dict[str, Any]]],
    max_contexts_per_job: int = 5,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for job, batch in zip(search_jobs, retrieval_batches):
        query_text = str(job.get("query_text", "")).strip()
        query_type = str(job.get("query_type", "")).strip()
        clinical_focus = str(job.get("clinical_focus", "")).strip()

        contexts: list[str] = []
        top_concepts: list[str] = []
        seen_codes: set[str] = set()

        for candidate in batch[:max_contexts_per_job]:
            snomed_code = str(candidate.get("snomed_code", "")).strip()
            if not snomed_code or snomed_code in seen_codes:
                continue
            seen_codes.add(snomed_code)

            term = str(candidate.get("term", "")).strip()
            semantic_tag = str(candidate.get("semantic_tag", "")).strip()
            in_qof = bool(candidate.get("in_qof", False))
            in_opencodelists = bool(candidate.get("in_opencodelists", False))

            context = f"{term} ({semantic_tag}) | QOF={in_qof} | OpenCodelists={in_opencodelists}"
            contexts.append(context)
            top_concepts.append(f"{term} ({semantic_tag})")

        answer = (
            "Top retrieved concepts: " + "; ".join(top_concepts) + "."
            if top_concepts
            else "No concepts retrieved."
        )

        rows.append({
            "question": query_text,
            "contexts": contexts,
            "answer": answer,
            "stage": "retrieval_only",
            "query_type": query_type,
            "clinical_focus": clinical_focus,
        })

    return rows


def run_retrieval_evaluation(
    retrieval_eval_rows: list[dict[str, Any]],
    eval_model: str = DEFAULT_EVAL_MODEL,
) -> dict[str, Any]:
    if not retrieval_eval_rows:
        return {
            "stage": "retrieval_only",
            "num_rows": 0,
            "average_custom_score": None,
            "details": [],
        }

    results: list[dict[str, Any]] = []
    total_score = 0.0

    for row in retrieval_eval_rows:
        focus = (row.get("clinical_focus") or row.get("question") or "").strip()
        contexts_str = "\n".join(f"- {c}" for c in row.get("contexts", []))

        prompt = f"""You are evaluating retrieval quality for a clinical concept search system.

Focus condition: "{focus}"

Retrieved concepts:
{contexts_str}

Assess the retrieved concepts using these boolean checks:
1. "main_condition_retrieved": Is the main condition clearly present?
2. "top_results_relevant": Are the top retrieved concepts strongly relevant to the focus condition?
3. "noise_low": Are there no clearly off-topic concepts in the retrieved list?

Return ONLY valid JSON with exactly these keys:
{{
  "main_condition_retrieved": true/false,
  "top_results_relevant": true/false,
  "noise_low": true/false
}}
"""

        try:
            eval_dict = _judge_json(prompt, eval_model)
            checks = [
                bool(eval_dict.get("main_condition_retrieved", False)),
                bool(eval_dict.get("top_results_relevant", False)),
                bool(eval_dict.get("noise_low", False)),
            ]
            score = sum(checks) / 3.0
        except Exception as exc:
            eval_dict = {"error": str(exc)}
            score = 0.0

        results.append({
            "focus": focus,
            "score": round(score, 3),
            "details": eval_dict,
        })
        total_score += score

    return {
        "stage": "retrieval_only",
        "num_rows": len(results),
        "average_custom_score": round(total_score / len(results), 3),
        "details": results,
    }


def build_post_gate_eval_rows(
    structured_query: dict[str, Any],
    gate_candidates: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    conditions = _conditions_from_structured_query(structured_query)
    rows: list[dict[str, Any]] = []

    for condition in conditions:
        cond_norm = condition.lower().strip()
        condition_candidates = [
            c for c in gate_candidates
            if cond_norm in [m.lower().strip() for m in c.get("matched_conditions_from_gate", [])]
            or str(c.get("dominant_condition_from_gate", "")).lower().strip() == cond_norm
        ]
        condition_candidates.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        contexts: list[str] = []
        seen_codes: set[str] = set()
        for candidate in condition_candidates[:top_n]:
            code = str(candidate.get("snomed_code", "")).strip()
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            term = str(candidate.get("term", "")).strip()
            tag = str(candidate.get("semantic_tag", "")).strip()
            contexts.append(f"{term} ({tag})")

        answer = (
            "Top post-gate concepts: " + "; ".join(contexts) + "."
            if contexts
            else "No concepts survived gate."
        )

        rows.append({
            "focus_condition": condition,
            "all_conditions": conditions,
            "contexts": contexts,
            "answer": answer,
            "stage": "post_gate",
        })

    return rows


def run_post_gate_evaluation(
    post_gate_eval_rows: list[dict[str, Any]],
    eval_model: str = DEFAULT_EVAL_MODEL,
) -> dict[str, Any]:
    if not post_gate_eval_rows:
        return {
            "stage": "post_gate",
            "num_rows": 0,
            "average_custom_score": None,
            "details": [],
        }

    results: list[dict[str, Any]] = []
    total_score = 0.0

    for row in post_gate_eval_rows:
        focus = row.get("focus_condition", "").strip()
        all_conds_str = ", ".join(row.get("all_conditions", []))
        contexts_str = "\n".join(f"- {c}" for c in row.get("contexts", []))

        prompt = f"""You are evaluating candidate quality after a relevance gating stage in a clinical concept search system.

Full user query conditions: {all_conds_str}
Focus condition for this evaluation: "{focus}"

Surviving post-gate concepts:
{contexts_str}

Assess the surviving concepts using these boolean checks:
1. "main_condition_still_present": Is the target condition for that focus still clearly represented after gating?
2. "top_candidates_relevant": Are the highest-ranked surviving candidates genuinely relevant to the focus condition?
3. "noise_low_after_gate": Are obvious junk, weak matches, or semantically loose concepts mostly absent from this surviving list?
4. "cross_condition_contamination_low": Are candidates for this condition mostly free of heavy pollution from concepts belonging to the OTHER unrelated conditions in the query?

Return ONLY valid JSON with exactly these keys:
{{
  "main_condition_still_present": true/false,
  "top_candidates_relevant": true/false,
  "noise_low_after_gate": true/false,
  "cross_condition_contamination_low": true/false
}}
"""

        try:
            eval_dict = _judge_json(prompt, eval_model)
            checks = [
                bool(eval_dict.get("main_condition_still_present", False)),
                bool(eval_dict.get("top_candidates_relevant", False)),
                bool(eval_dict.get("noise_low_after_gate", False)),
                bool(eval_dict.get("cross_condition_contamination_low", False)),
            ]
            score = sum(checks) / 4.0
        except Exception as exc:
            eval_dict = {"error": str(exc)}
            score = 0.0

        results.append({
            "focus": focus,
            "score": round(score, 3),
            "details": eval_dict,
        })
        total_score += score

    return {
        "stage": "post_gate",
        "num_rows": len(results),
        "average_custom_score": round(total_score / max(1, len(results)), 3),
        "details": results,
    }


def build_post_ce_eval_rows(
    structured_query: dict[str, Any],
    ce_reranked_candidates: list[dict[str, Any]],
    top_n: int = 5,
) -> list[dict[str, Any]]:
    conditions = _conditions_from_structured_query(structured_query)
    rows: list[dict[str, Any]] = []

    for condition in conditions:
        cond_norm = condition.lower().strip()
        condition_candidates = [
            c for c in ce_reranked_candidates
            if cond_norm in [m.lower().strip() for m in c.get("matched_conditions_from_ce", [])]
            or str(c.get("dominant_condition_from_ce", "")).lower().strip() == cond_norm
        ]
        condition_candidates.sort(key=lambda x: x.get("ce_score", 0.0), reverse=True)

        contexts: list[str] = []
        seen_codes: set[str] = set()
        for candidate in condition_candidates[:top_n]:
            code = str(candidate.get("snomed_code", "")).strip()
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            term = str(candidate.get("term", "")).strip()
            tag = str(candidate.get("semantic_tag", "")).strip()
            contexts.append(f"{term} ({tag})")

        answer = (
            "Top post-CE concepts: " + "; ".join(contexts) + "."
            if contexts
            else "No concepts survived cross-encoder."
        )

        rows.append({
            "focus_condition": condition,
            "contexts": contexts,
            "answer": answer,
            "stage": "post_cross_encoder",
        })

    return rows


def run_post_ce_evaluation(
    post_ce_eval_rows: list[dict[str, Any]],
    eval_model: str = DEFAULT_EVAL_MODEL,
) -> dict[str, Any]:
    if not post_ce_eval_rows:
        return {
            "stage": "post_cross_encoder",
            "num_rows": 0,
            "average_custom_score": None,
            "details": [],
        }

    results: list[dict[str, Any]] = []
    total_score = 0.0

    for row in post_ce_eval_rows:
        focus = row.get("focus_condition", "").strip()
        contexts = row.get("contexts", [])

        if not contexts:
            results.append({
                "focus": focus,
                "score": 0.0,
                "details": {
                    "main_condition_still_present": False,
                    "top_candidates_relevant": False,
                    "ranking_improved_after_ce": False,
                    "noise_low_after_ce": False,
                    "error": "No concepts survived cross-encoder.",
                },
            })
            continue

        contexts_str = "\n".join(f"- {c}" for c in contexts)
        prompt = f"""You are evaluating candidate quality after a cross-encoder reranking stage in a clinical concept search system.

Focus condition for this evaluation: "{focus}"

Surviving post-CE concepts (in ranked order):
{contexts_str}

Assess the surviving concepts using these boolean checks:
1. "main_condition_still_present": Is the target condition for that focus still clearly represented after CE reranking?
2. "top_candidates_relevant": Are the highest-ranked surviving candidates genuinely relevant to the focus condition?
3. "ranking_improved_after_ce": Does the ordering appear highly logical and semantically refined for the focus condition?
4. "noise_low_after_ce": Are weak or misleading concepts reduced/absent in this top CE-ranked set?

Return ONLY valid JSON with exactly these keys:
{{
  "main_condition_still_present": true/false,
  "top_candidates_relevant": true/false,
  "ranking_improved_after_ce": true/false,
  "noise_low_after_ce": true/false
}}
"""

        try:
            eval_dict = _judge_json(prompt, eval_model)
            checks = [
                bool(eval_dict.get("main_condition_still_present", False)),
                bool(eval_dict.get("top_candidates_relevant", False)),
                bool(eval_dict.get("ranking_improved_after_ce", False)),
                bool(eval_dict.get("noise_low_after_ce", False)),
            ]
            score = sum(checks) / 4.0
        except Exception as exc:
            eval_dict = {"error": str(exc)}
            score = 0.0

        results.append({
            "focus": focus,
            "score": round(score, 3),
            "details": eval_dict,
        })
        total_score += score

    return {
        "stage": "post_cross_encoder",
        "num_rows": len(results),
        "average_custom_score": round(total_score / max(1, len(results)), 3),
        "details": results,
    }


def build_final_decision_eval_rows(
    user_query: str,
    structured_query: dict[str, Any],
    candidate_groups: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    conditions = _conditions_from_structured_query(structured_query)

    def extract_clean_strings(candidates: list[dict[str, Any]]) -> list[str]:
        return [
            f"{c.get('term', '')} ({c.get('semantic_tag', '')})"
            if c.get("semantic_tag")
            else f"{c.get('term', '')}"
            for c in candidates
        ]

    include_candidates = extract_clean_strings(candidate_groups.get("include_candidates", []))
    review_candidates = extract_clean_strings(candidate_groups.get("review_candidates", []))
    specific_variants = extract_clean_strings(candidate_groups.get("specific_variants", []))
    suppressed_candidates = extract_clean_strings(candidate_groups.get("suppressed_candidates", []))

    answer = (
        f"Include ({len(include_candidates)}): {'; '.join(include_candidates)}\n"
        f"Review ({len(review_candidates)}): {'; '.join(review_candidates)}\n"
        f"Specific ({len(specific_variants)}): {'; '.join(specific_variants)}\n"
        f"Suppressed ({len(suppressed_candidates)}): {'; '.join(suppressed_candidates)}"
    )

    return [{
        "question": user_query,
        "conditions": conditions,
        "include_candidates": include_candidates,
        "review_candidates": review_candidates,
        "specific_variants": specific_variants,
        "suppressed_candidates": suppressed_candidates,
        "answer": answer,
        "stage": "final_decision",
    }]


def run_final_decision_evaluation(
    final_decision_eval_rows: list[dict[str, Any]],
    eval_model: str = DEFAULT_EVAL_MODEL,
) -> dict[str, Any]:
    if not final_decision_eval_rows:
        return {
            "stage": "final_decision",
            "num_rows": 0,
            "average_custom_score": None,
            "details": [],
        }

    results: list[dict[str, Any]] = []
    total_score = 0.0

    for row in final_decision_eval_rows:
        question = row.get("question", "")
        conditions_str = ", ".join(row.get("conditions", []))
        include_str = "\n".join(f"- {c}" for c in row.get("include_candidates", []))
        review_str = "\n".join(f"- {c}" for c in row.get("review_candidates", []))
        specific_str = "\n".join(f"- {c}" for c in row.get("specific_variants", []))
        suppressed_str = "\n".join(f"- {c}" for c in row.get("suppressed_candidates", []))

        prompt = f"""You are evaluating the final bucketed output of a clinical concept search system.

Original query: "{question}"
Identified conditions: {conditions_str}

Final Bucketed Output:
INCLUDE (Core Anchors):
{include_str if include_str else '- None'}

REVIEW (Variants):
{review_str if review_str else '- None'}

SPECIFIC (Deeper/Narrower Concepts):
{specific_str if specific_str else '- None'}

SUPPRESSED (Filtered):
{suppressed_str if suppressed_str else '- None'}

Assess the final decisions using these boolean checks:
1. "anchor_correct": Does the INCLUDE bucket contain the correct broad anchor concept(s) for the condition(s)?
2. "wrong_core_concepts_absent": Are clearly wrong, over-specific, causal, or misleading concepts absent from the INCLUDE bucket?
3. "bucket_assignment_reasonable": Are anchors in INCLUDE, relevant variants in REVIEW, deeper/narrower concepts in SPECIFIC, and is obvious junk absent?
4. "cross_condition_balance_correct": For multi-condition queries, is each real condition represented sensibly without one condition completely overwhelming the others?

Return ONLY valid JSON with exactly these keys:
{{
  "anchor_correct": true/false,
  "wrong_core_concepts_absent": true/false,
  "bucket_assignment_reasonable": true/false,
  "cross_condition_balance_correct": true/false
}}
"""

        try:
            eval_dict = _judge_json(prompt, eval_model)
            checks = [
                bool(eval_dict.get("anchor_correct", False)),
                bool(eval_dict.get("wrong_core_concepts_absent", False)),
                bool(eval_dict.get("bucket_assignment_reasonable", False)),
                bool(eval_dict.get("cross_condition_balance_correct", False)),
            ]
            score = sum(checks) / 4.0
        except Exception as exc:
            eval_dict = {"error": str(exc)}
            score = 0.0

        results.append({
            "question": question,
            "score": round(score, 3),
            "details": eval_dict,
        })
        total_score += score

    return {
        "stage": "final_decision",
        "num_rows": len(results),
        "average_custom_score": round(total_score / max(1, len(results)), 3),
        "details": results,
    }


def build_final_response_eval_rows(
    user_query: str,
    candidate_groups: dict[str, list[dict[str, Any]]],
    formatted_output: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    contexts: list[str] = []
    for bucket in ["include_candidates", "review_candidates"]:
        for candidate in candidate_groups.get(bucket, []):
            term = candidate.get("term", "")
            tag = candidate.get("semantic_tag", "")

            if "evidence" in candidate:
                qof = candidate["evidence"].get("in_qof", False)
                oc = candidate["evidence"].get("in_opencodelists", False)
                usage = candidate["evidence"].get("usage_count_nhs", 0.0)
            else:
                qof = candidate.get("in_qof", False)
                oc = candidate.get("in_opencodelists", False)
                usage = candidate.get("usage_count_nhs", 0.0)

            if tag and f"({tag})" not in term:
                context = f"Concept: {term} ({tag}) | QOF={qof} | OpenCodelists={oc} | Usage={usage}"
            else:
                context = f"Concept: {term} | QOF={qof} | OpenCodelists={oc} | Usage={usage}"
            contexts.append(context)

    response_lines = ["Recommended SNOMED concepts:"]
    for bucket in ["include_candidates", "review_candidates", "specific_variants", "suppressed_candidates"]:
        items = formatted_output.get(bucket, [])
        if items:
            response_lines.append(f"\n{bucket.upper()}:")
            for item in items:
                term = item.get("term", "Unknown")
                rationale = item.get("rationale", "No rationale provided.")
                response_lines.append(f"- {term}: {rationale}")

    return [{
        "user_input": user_query,
        "retrieved_contexts": contexts,
        "response": "\n".join(response_lines),
    }]


def run_final_response_ragas_evaluation(
    final_response_eval_rows: list[dict[str, Any]],
    evaluator_llm: Any = None,
    ragas_model: str = DEFAULT_RAGAS_MODEL,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    if not final_response_eval_rows:
        return {
            "stage": "final_response_ragas",
            "num_rows": 0,
            "faithfulness": None,
            "response_relevancy": None,
            "details": [],
        }

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import llm_factory
        from ragas.metrics import answer_relevancy, faithfulness

        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError:
            from langchain_community.embeddings import HuggingFaceEmbeddings

        dataset = Dataset.from_list(final_response_eval_rows)
        client = get_nebius_client()
        ragas_eval_llm = evaluator_llm or llm_factory(ragas_model, client=client)
        ragas_embeddings = LangchainEmbeddingsWrapper(
            HuggingFaceEmbeddings(model_name=embedding_model)
        )

        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=ragas_eval_llm,
            embeddings=ragas_embeddings,
        )
        df_res = result.to_pandas()
        details = df_res.to_dict(orient="records")
        avg_faithfulness = float(df_res["faithfulness"].mean()) if "faithfulness" in df_res else None
        avg_relevancy = float(df_res["answer_relevancy"].mean()) if "answer_relevancy" in df_res else None
    except Exception as exc:
        return {
            "stage": "final_response_ragas",
            "num_rows": len(final_response_eval_rows),
            "error": str(exc),
            "faithfulness": None,
            "response_relevancy": None,
            "details": [],
        }

    return {
        "stage": "final_response_ragas",
        "num_rows": len(final_response_eval_rows),
        "faithfulness": avg_faithfulness,
        "response_relevancy": avg_relevancy,
        "details": details,
    }
