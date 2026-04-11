import json
import sys
from pathlib import Path

# Add project root to path so we can import from main project
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from main import run_pipeline

def check_substrings(candidates, expected_substrings):
    terms = [c.get("term", "").lower() for c in candidates]
    missing = []
    for sub in expected_substrings:
        if not any(sub.lower() in term for term in terms):
            missing.append(sub)
    return missing

def main():
    cases_file = project_root / "tests" / "regression_cases.json"
    if not cases_file.exists():
        print(f"Error: {cases_file} not found.")
        sys.exit(1)

    with open(cases_file, "r") as f:
        cases = json.load(f)

    all_passed = True

    for case in cases:
        query = case["query"]
        print(f"\nRunning regression check for query: '{query}'")
        try:
            output = run_pipeline(query, top_k=10)
        except Exception as e:
            print(f"  [FAIL] Pipeline raised an exception: {e}")
            all_passed = False
            continue

        expectations = case.get("expectations", {})
        
        # 1. Required buckets
        for bucket in expectations.get("required_buckets", []):
            if bucket not in output:
                print(f"  [FAIL] Missing required bucket: {bucket}")
                all_passed = False
            else:
                print(f"  [PASS] Found bucket: {bucket}")

        # 2. Must not be empty
        for bucket in expectations.get("must_not_be_empty_buckets", []):
            if bucket in output and len(output[bucket]) == 0:
                print(f"  [FAIL] Bucket '{bucket}' is empty but expected to have items.")
                all_passed = False
            elif bucket in output:
                print(f"  [PASS] Bucket '{bucket}' is appropriately populated ({len(output[bucket])} items).")

        # 3. Substring term matching
        subs = expectations.get("must_include_terms_substring", [])
        if subs:
            # check the main include_candidates Usually
            pool = output.get("include_candidates", [])
            missing = check_substrings(pool, subs)
            if missing:
                print(f"  [FAIL] include_candidates missing expected substrings: {missing}")
                all_passed = False
            else:
                print(f"  [PASS] include_candidates contains expected key concepts.")
                
        # 4. Suppression logic check
        if "suppression_expectations" in expectations:
            if len(output.get("suppressed_candidates", [])) > 0:
                print(f"  [PASS] Suppression logic behaves correctly for this context.")
            else:
                print(f"  [FAIL] Expected items in suppressed_candidates but found 0.")
                all_passed = False

    if all_passed:
        print("\nAll regression checks PASSED.")
        sys.exit(0)
    else:
        print("\nSome regression checks FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
