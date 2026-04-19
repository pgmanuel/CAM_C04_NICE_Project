import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from decision_engine import DecisionEngine


def candidate(
    code: str,
    term: str,
    *,
    ce_score: float,
    condition: str = "hypertension",
    semantic_tag: str = "disorder",
) -> dict:
    return {
        "snomed_code": code,
        "term": term,
        "semantic_tag": semantic_tag,
        "in_qof": False,
        "in_opencodelists": False,
        "usage_count_nhs": 1000.0,
        "fusion_score": ce_score,
        "rrf_score": ce_score,
        "semantic_score_max": ce_score,
        "bm25_score_max": ce_score,
        "lexical_overlap_max": 1.0,
        "term_precision_max": 1.0,
        "specificity_score_max": 0.8,
        "query_coverage_count": 1,
        "rerank_score": ce_score,
        "relevance_score": ce_score,
        "condition_relevance": {},
        "ce_score": ce_score,
        "ce_score_by_condition": {
            condition: {
                "ce_condition_score": ce_score,
                "ce_score": ce_score,
            }
        },
        "matched_conditions_from_ce": [condition],
        "dominant_condition_from_ce": condition,
        "retrieval_method": "direct",
        "retrieval_trace": [],
    }


def run_decision(candidates: list[dict], primary_condition: str = "hypertension") -> dict:
    structured_query = {
        "original_query": primary_condition,
        "primary_condition": primary_condition,
        "secondary_conditions": [],
        "modifiers": [],
        "supporting_terms": [],
    }
    return DecisionEngine().assign_final_decisions(
        candidates,
        structured_query=structured_query,
        top_k=10,
    )


def terms(bucket: list[dict]) -> list[str]:
    return [item["term"].lower() for item in bucket]


def test_plain_hypertension_selects_anchor():
    output = run_decision([
        candidate("1", "Hypertension", ce_score=0.9),
    ])

    assert terms(output["include_candidates"]) == ["hypertension"]
    assert output["include_candidates"][0]["candidate_role"] == "core"


def test_stable_hypertension_can_be_anchor_for_stable_query():
    output = run_decision([
        candidate("2", "Stable hypertension", ce_score=0.9, condition="stable hypertension"),
    ], primary_condition="stable hypertension")

    assert terms(output["include_candidates"]) == ["stable hypertension"]
    assert output["include_candidates"][0]["candidate_role"] == "core"


def test_pulmonary_hypertension_is_not_core_for_plain_hypertension():
    output = run_decision([
        candidate("1", "Hypertension", ce_score=0.45),
        candidate("3", "Pulmonary hypertension", ce_score=0.95),
    ])

    assert "hypertension" in terms(output["include_candidates"])
    assert "pulmonary hypertension" not in terms(output["include_candidates"])
    assert "pulmonary hypertension" in terms(output["specific_variants"])


def test_fear_of_hypertension_is_suppressed_not_core():
    output = run_decision([
        candidate("1", "Hypertension", ce_score=0.45),
        candidate("4", "Fear of hypertension", ce_score=0.95),
    ])

    assert "fear of hypertension" not in terms(output["include_candidates"])
    assert "fear of hypertension" in terms(output["suppressed_candidates"])
