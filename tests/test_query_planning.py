import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from query_planning import QueryPlanner


QUERY_WEIGHTS = {
    "primary_condition": 1.0,
    "secondary_condition": 1.0,
    "modifier": 0.5,
    "supporting_term": 0.35,
    "combined": 0.0,
}


def _job_texts(structured_query):
    planner = QueryPlanner(QUERY_WEIGHTS)
    return [job["query_text"] for job in planner.build_search_queries(structured_query)]


def test_modifier_scoping_blocks_unrelated_condition_jobs():
    texts = _job_texts({
        "primary_condition": "type 2 diabetes",
        "secondary_conditions": ["obesity", "hypertension"],
        "modifiers": ["poorly controlled", "morbid"],
        "supporting_terms": [],
    })

    assert "type 2 diabetes" in texts
    assert "poorly controlled type 2 diabetes" in texts
    assert "morbid type 2 diabetes" not in texts

    assert "obesity" in texts
    assert "morbid obesity" in texts
    assert "poorly controlled obesity" not in texts

    assert "hypertension" in texts
    assert "morbid hypertension" not in texts
    assert "poorly controlled hypertension" not in texts


def test_broad_severity_modifier_still_applies_across_conditions():
    texts = _job_texts({
        "primary_condition": "asthma",
        "secondary_conditions": ["hypertension"],
        "modifiers": ["severe"],
        "supporting_terms": [],
    })

    assert "severe asthma" in texts
    assert "severe hypertension" in texts


def test_duplicate_modifier_words_are_not_repeated():
    texts = _job_texts({
        "primary_condition": "morbid obesity",
        "secondary_conditions": [],
        "modifiers": ["morbid"],
        "supporting_terms": [],
    })

    assert texts == ["morbid obesity"]
