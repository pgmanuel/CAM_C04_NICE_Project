import sys
import pytest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from main import run_pipeline

@pytest.fixture(scope="session")
def demo_query_output():
    # Run once per test session to avoid repeated heavy model loads
    query = "Obesity, diabetes mellitus, and hypertension"
    return run_pipeline(query, top_k=10)

def test_buckets_exist(demo_query_output):
    """Ensure all critical structural buckets are produced."""
    expected_buckets = [
        "include_candidates", 
        "review_candidates", 
        "specific_variants", 
        "suppressed_candidates"
    ]
    for bucket in expected_buckets:
        assert bucket in demo_query_output, f"Missing {bucket} in pipeline output."

def test_anchor_concepts_present(demo_query_output):
    """Ensure key terms loosely exist in the 'include_candidates' bucket."""
    includes = demo_query_output.get("include_candidates", [])
    terms = [c.get("term", "").lower() for c in includes]
    
    # We do loose substring matching to avoid precise hardcoded String breaks
    assert any("obesity" in t for t in terms), "Obesity concept missing from core includes."
    assert any("diabetes" in t for t in terms), "Diabetes concept missing from core includes."
    assert any("hypertension" in t for t in terms), "Hypertension concept missing from core includes."

def test_suppression_logic_active(demo_query_output):
    """Ensure the suppression bucket catches at least one item."""
    suppressed = demo_query_output.get("suppressed_candidates", [])
    assert len(suppressed) > 0, "Suppression logic failed to catch items in a complex known query."
