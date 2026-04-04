import json
import sys

import pytest
from project_paths import build_project_paths, find_project_root

PROJECT_DIR = find_project_root() / "CAM_Project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from main import run_pipeline

CASES_PATH = build_project_paths(find_project_root()).regression_cases_path


def _load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", _load_cases(), ids=lambda case: case["name"])
def test_regression_properties(case: dict) -> None:
    output = run_pipeline(case["query"], top_k=20)

    include_terms = [item["term"] for item in output["include_candidates"]]
    include_text = " || ".join(include_terms).lower()

    for expected_term in case["include_must_contain"]:
        assert any(expected_term.lower() in term.lower() for term in include_terms)

    for blocked_term in case["include_must_not_contain"]:
        assert blocked_term.lower() not in include_text

    assert len(output["review_candidates"]) <= case["max_review_candidates"]
    assert len(output["specific_variants"]) <= case["max_specific_variants"]
