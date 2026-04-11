import math
from collections import defaultdict
from typing import Any

from pipeline_policy import (
    CAUSAL_TRIGGER_TERMS,
    CORE_BLOCK_TERMS,
    NARROW_SUBTYPE_TERMS,
    PREGNANCY_TRIGGER_TERMS,
    QUERY_EXCEPTION_TERMS,
)
from query_planning import normalize_query, tokenize_text


def major_conditions(structured_query: dict[str, Any]) -> list[str]:
    conditions = [structured_query.get("primary_condition", "")]
    conditions.extend(structured_query.get("secondary_conditions", []))
    normalized = []
    seen = set()
    for condition in conditions:
        text = normalize_query(condition)
        if text and text.lower() not in seen:
            seen.add(text.lower())
            normalized.append(text)
    return normalized


def query_context(structured_query: dict[str, Any]) -> dict[str, Any]:
    raw_query = str(structured_query.get("original_query", "")).lower()
    modifier_tokens = set()
    for modifier in structured_query.get("modifiers", []):
        modifier_tokens |= tokenize_text(modifier)
    supporting_tokens = set()
    for term in structured_query.get("supporting_terms", []):
        supporting_tokens |= tokenize_text(term)
    return {
        "raw_query": raw_query,
        "modifier_tokens": modifier_tokens,
        "supporting_tokens": supporting_tokens,
    }


def contains_any_phrase(text: str, phrases: set[str] | tuple[str, ...]) -> bool:
    lowered = str(text).lower()
    return any(phrase in lowered for phrase in phrases)


def allows_resolved_language(query_ctx: dict[str, Any]) -> bool:
    return contains_any_phrase(query_ctx.get("raw_query", ""), QUERY_EXCEPTION_TERMS)


def allows_causal_language(query_ctx: dict[str, Any]) -> bool:
    return contains_any_phrase(query_ctx.get("raw_query", ""), CAUSAL_TRIGGER_TERMS)


def is_resolved_candidate(candidate: dict[str, Any]) -> bool:
    return "resolved" in str(candidate.get("term", "")).lower()


def is_causal_variant(candidate: dict[str, Any]) -> bool:
    return contains_any_phrase(str(candidate.get("term", "")).lower(), CAUSAL_TRIGGER_TERMS)


def is_pregnancy_variant(candidate: dict[str, Any]) -> bool:
    return contains_any_phrase(str(candidate.get("term", "")).lower(), PREGNANCY_TRIGGER_TERMS)


def has_primary_modifier_alignment(
    candidate: dict[str, Any],
    matched_conditions: list[str],
    conditions: list[str],
    query_ctx: dict[str, Any],
) -> bool:
    if not conditions or conditions[0] not in matched_conditions:
        return False
    term = str(candidate.get("term", "")).lower()
    modifier_tokens = query_ctx.get("modifier_tokens", set())
    supporting_tokens = query_ctx.get("supporting_tokens", set())
    if "severe" in modifier_tokens and any(token in term for token in ["severe", "morbid", "extreme"]):
        return True
    if {"poorly", "controlled"} & supporting_tokens and any(
        token in term for token in ["poorly controlled", "uncontrolled", "control"]
    ):
        return True
    return False


def is_generic_anchor(candidate: dict[str, Any], condition: str) -> bool:
    term_tokens = tokenize_text(candidate.get("term", ""))
    condition_tokens = tokenize_text(condition)
    if not condition_tokens:
        return False
    extra_tokens = term_tokens - condition_tokens - {"disorder", "finding"}
    return len(extra_tokens) == 0


def is_broad_core_refinement(candidate: dict[str, Any], condition: str) -> bool:
    term = str(candidate.get("term", "")).lower()
    normalized_condition = normalize_query(condition).lower()
    if normalized_condition == "hypertension":
        return "essential hypertension" in term
    return False


def is_narrow_subtype_variant(candidate: dict[str, Any], query_ctx: dict[str, Any]) -> bool:
    term = str(candidate.get("term", "")).lower()
    raw_query = query_ctx.get("raw_query", "")
    return any(phrase in term and phrase not in raw_query for phrase in NARROW_SUBTYPE_TERMS)


def is_preferred_obesity_refinement(candidate: dict[str, Any]) -> bool:
    term = str(candidate.get("term", "")).lower()
    return "obesity" in term and any(token in term for token in ("severe", "morbid"))


def candidate_role(
    candidate: dict[str, Any],
    matched_conditions: list[str],
    conditions: list[str],
    query_ctx: dict[str, Any],
    ranking_components: dict[str, Any],
) -> str:
    term = str(candidate.get("term", "")).lower()
    raw_query = query_ctx.get("raw_query", "")
    core_blocked = any(block_term in term and block_term not in raw_query for block_term in CORE_BLOCK_TERMS)
    if is_resolved_candidate(candidate) and not allows_resolved_language(query_ctx):
        return "suppress"
    if "remission" in term and not allows_resolved_language(query_ctx):
        return "suppress"
    if is_pregnancy_variant(candidate):
        return "suppress"
    if is_causal_variant(candidate) and not allows_causal_language(query_ctx):
        return "suppress"
    if float(ranking_components.get("query_misalignment_penalty", 0.0)) >= 1.5:
        return "suppress"
    if not core_blocked and has_primary_modifier_alignment(candidate, matched_conditions, conditions, query_ctx):
        return "core"
    if not core_blocked and any(is_generic_anchor(candidate, condition) for condition in matched_conditions):
        return "core"
    if not core_blocked and any(is_broad_core_refinement(candidate, condition) for condition in matched_conditions):
        return "core"
    if is_preferred_obesity_refinement(candidate):
        return "variant"
    if is_narrow_subtype_variant(candidate, query_ctx):
        return "specific"
    if matched_conditions:
        return "variant"
    return "specific"


def serialize_candidate_output(row: dict[str, Any], index: int | None = None) -> dict[str, Any]:
    payload = {
        "presentation_score": row["presentation_score"],
        "snomed_code": row["snomed_code"],
        "term": row["term"],
        "semantic_tag": row.get("semantic_tag", ""),
        "confidence_tier": row["confidence_tier"],
        "candidate_role": row.get("candidate_role"),
        "evidence": {
            "in_qof": bool(row["in_qof"]),
            "in_opencodelists": bool(row["in_opencodelists"]),
            "usage_count_nhs": float(row["usage_count_nhs"]),
        },
        "retrieval_features": {
            "fusion_score": float(row["fusion_score"]),
            "best_primary_condition_score": float(row["best_primary_condition_score"]),
            "semantic_score_max": float(row["semantic_score_max"]),
            "bm25_score_max": float(row["bm25_score_max"]),
            "lexical_overlap_max": float(row["lexical_overlap_max"]),
            "query_coverage_count": int(row["query_coverage_count"]),
        },
        "matched_conditions": row["matched_conditions"],
        "dominant_condition": row["dominant_condition"],
        "ranking_components": row["ranking_components"],
        "authority_bucket": int(row["authority_bucket"]),
        "retrieval_trace": row["retrieval_trace"],
    }
    if index is not None:
        payload["presentation_rank"] = index
    return payload


def condition_matches(candidate: dict[str, Any], conditions: list[str]) -> list[str]:
    candidate_terms = tokenize_text(candidate.get("term", ""))
    supported = []
    for condition in conditions:
        condition_terms = tokenize_text(condition)
        if condition_terms and condition_terms & candidate_terms:
            supported.append(condition)

    if supported:
        return supported

    weighted_by_focus: dict[str, float] = defaultdict(float)
    for trace in candidate.get("retrieval_trace", []):
        clinical_focus = normalize_query(trace.get("clinical_focus", ""))
        if clinical_focus in conditions:
            weighted_by_focus[clinical_focus] += float(trace.get("weighted_retrieval_score", 0.0))

    return [focus for focus, score in weighted_by_focus.items() if score > 0]


def dominant_condition(candidate: dict[str, Any], matched_conditions: list[str], conditions: list[str]) -> str:
    if not matched_conditions:
        return normalize_query(conditions[0]) if conditions else "unassigned"

    weighted_by_focus: dict[str, float] = defaultdict(float)
    for trace in candidate.get("retrieval_trace", []):
        clinical_focus = normalize_query(trace.get("clinical_focus", ""))
        if clinical_focus in matched_conditions:
            weighted_by_focus[clinical_focus] += float(trace.get("weighted_retrieval_score", 0.0))

    if weighted_by_focus:
        return max(weighted_by_focus.items(), key=lambda item: item[1])[0]
    return matched_conditions[0]


def compute_authority_score(candidate: dict[str, Any]) -> float:
    usage_count_nhs = float(candidate.get("usage_count_nhs", 0.0))
    usage_component = min(0.55, math.log10(usage_count_nhs + 1.0) / 12.0)
    authority = usage_component
    if bool(candidate.get("in_qof", False)):
        authority += 0.9
    if bool(candidate.get("in_opencodelists", False)):
        authority += 0.35
    return authority


def compute_coverage_component(
    candidate: dict[str, Any],
    matched_conditions: list[str],
    multimorbidity: bool,
) -> float:
    condition_coverage = len(matched_conditions)
    query_type_coverage = min(int(candidate.get("query_coverage_count", 0)), 5) * 0.05
    multimorbidity_bonus = 0.18 if multimorbidity and condition_coverage > 1 else 0.0
    return query_type_coverage + (condition_coverage * 0.12) + multimorbidity_bonus


def compute_primary_alignment_component(
    dominant: str,
    matched_conditions: list[str],
    conditions: list[str],
) -> float:
    if not conditions:
        return 0.0
    primary_condition = conditions[0]
    if primary_condition in matched_conditions and dominant == primary_condition:
        return 0.8
    if primary_condition in matched_conditions:
        return 0.45
    return 0.0


def compute_centrality_component(
    candidate: dict[str, Any],
    matched_conditions: list[str],
    conditions: list[str],
) -> float:
    if not matched_conditions:
        return 0.0
    term_tokens = tokenize_text(candidate.get("term", ""))
    bonuses = []
    for condition in matched_conditions:
        cond_tokens = tokenize_text(condition)
        extra = term_tokens - cond_tokens - {"disorder", "finding"}
        if len(extra) <= 1:
            bonuses.append(0.25 if condition == conditions[0] else 0.14)
    return max(bonuses) if bonuses else 0.0


def compute_modifier_component(
    candidate: dict[str, Any],
    matched_conditions: list[str],
    conditions: list[str],
    query_ctx: dict[str, Any],
) -> float:
    term = str(candidate.get("term", "")).lower()
    modifier_tokens = query_ctx.get("modifier_tokens", set())
    supporting_tokens = query_ctx.get("supporting_tokens", set())
    bonus = 0.0
    primary_condition = conditions[0].lower() if conditions else ""

    if "severe" in modifier_tokens and any(token in term for token in ["severe", "morbid", "extreme"]):
        if primary_condition == "obesity" and (
            "obesity" in term or any(condition.lower() == "obesity" for condition in matched_conditions)
        ):
            bonus += 0.55
        elif "obesity" in term or any(condition.lower() == "obesity" for condition in matched_conditions):
            bonus += 0.3
        else:
            bonus += 0.08

    if {"poorly", "controlled"} & supporting_tokens and any(
        token in term for token in ["poorly controlled", "uncontrolled", "control"]
    ):
        bonus += 0.14

    return bonus


def compute_query_misalignment_penalty(candidate: dict[str, Any], query_ctx: dict[str, Any]) -> float:
    penalty = 0.0
    term = str(candidate.get("term", "")).lower()
    raw_query = query_ctx.get("raw_query", "")

    if is_resolved_candidate(candidate) and not allows_resolved_language(query_ctx):
        penalty += 4.0

    if is_causal_variant(candidate) and not allows_causal_language(query_ctx):
        penalty += 1.6

    if is_pregnancy_variant(candidate) and not contains_any_phrase(raw_query, PREGNANCY_TRIGGER_TERMS):
        penalty += 2.0

    if "localized" in term and "localized" not in raw_query:
        penalty += 0.4

    return penalty


def compute_subtype_clutter_penalty(
    candidate: dict[str, Any],
    matched_conditions: list[str],
    multimorbidity: bool,
    query_ctx: dict[str, Any],
) -> float:
    term = str(candidate.get("term", "")).lower()
    term_tokens = tokenize_text(term)
    base_condition_tokens = set()
    for condition in matched_conditions:
        base_condition_tokens |= tokenize_text(condition)

    extra_tokens = term_tokens - base_condition_tokens - {"disorder", "finding"}
    has_authority = (
        bool(candidate.get("in_qof", False))
        or bool(candidate.get("in_opencodelists", False))
        or float(candidate.get("usage_count_nhs", 0.0)) >= 1000
    )
    clutter_markers = {
        "hyperplastic",
        "hypertrophic",
        "associated",
        "due",
        "history",
        "extreme",
        "past",
        "retinopathy",
        "amenorrhea",
        "lymphedema",
    }

    penalty = 0.0
    if extra_tokens & clutter_markers and not has_authority:
        penalty += 0.45
    elif len(extra_tokens) >= 2 and not has_authority:
        penalty += 0.25

    if multimorbidity and len(matched_conditions) <= 1 and not has_authority:
        penalty += 0.15

    raw_query = query_ctx.get("raw_query", "")
    phrase_penalties = {
        "type 1": 0.65,
        "malignant": 0.55,
        "gestational": 0.55,
        "juvenile": 0.45,
        "secondary": 0.65,
        "renovascular": 0.8,
        "drug-induced": 0.8,
        "brittle": 0.7,
        "goldblatt": 0.8,
    }
    if "type 2" in term and "type 2" not in raw_query:
        penalty += 0.12
    for phrase, value in phrase_penalties.items():
        if phrase in term and phrase not in raw_query:
            penalty += value

    if "morbid" in term and "severe" not in raw_query:
        penalty += 0.18

    return penalty


def compute_fusion_component(candidate: dict[str, Any]) -> float:
    return (
        float(candidate.get("fusion_score", 0.0)) * 0.32
        + float(candidate.get("semantic_score_max", 0.0)) * 0.06
        + float(candidate.get("bm25_score_max", 0.0)) * 0.06
        + float(candidate.get("lexical_overlap_max", 0.0)) * 0.12
    )
