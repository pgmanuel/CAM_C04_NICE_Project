import json
import sys

from project_paths import build_project_paths, find_project_root

PROJECT_DIR = find_project_root() / "CAM_Project"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from main import run_pipeline

CASES_PATH = build_project_paths(find_project_root()).regression_cases_path


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def run_case(case: dict) -> list[str]:
    errors: list[str] = []
    output = run_pipeline(case["query"], top_k=20)
    include_terms = [item["term"] for item in output["include_candidates"]]
    include_text = " || ".join(include_terms).lower()

    for expected_term in case.get("include_must_contain", []):
        if not any(expected_term.lower() in term.lower() for term in include_terms):
            errors.append(f"missing include anchor: {expected_term}")

    for blocked_term in case.get("include_must_not_contain", []):
        if blocked_term.lower() in include_text:
            errors.append(f"blocked term appeared in include_candidates: {blocked_term}")

    if len(output["review_candidates"]) > case.get("max_review_candidates", 3):
        errors.append("review_candidates exceeded configured cap")

    if len(output["specific_variants"]) > case.get("max_specific_variants", 5):
        errors.append("specific_variants exceeded configured cap")

    return errors


def main() -> int:
    failures = 0
    for case in load_cases():
        errors = run_case(case)
        if errors:
            failures += 1
            print(f"FAIL {case['name']}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"PASS {case['name']}")

    if failures:
        print(f"\n{failures} regression case(s) failed.")
        return 1

    print("\nAll regression cases passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
