from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
from service import generate_report

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


SAMPLE_PAYLOAD: dict = {
    "query": "Type 2 diabetes with hypertension and possible CKD in elderly patient",
    "metadata": {
        "source": "GP clinical record",
        "reviewer": "clinical_coder_v2",
        "timestamp": "2026-03-24T10:00:00Z",
        "confidence_threshold": 0.75,
    },
    "candidates": [
        {
            "MedCodeId": "44054006",
            "term": "Diabetes mellitus type 2",
            "status": "active",
            "score": 0.92,
            "snippets": [
                "Patient has longstanding T2DM managed with metformin.",
                "HbA1c elevated at 58 mmol/mol on most recent review.",
            ],
        },
        {
            "concept_id": "73211009",
            "label": "Essential hypertension",
            "priority": "high",
            "rationale": "Repeated BP readings >140/90 documented over 3 consecutive visits.",
        },
        {
            "id": "CKD-POSSIBLE",
            "name": "Chronic kidney disease (possible)",
            "review_flag": "REVIEW",
            "evidence": (
                "Reviewer noted CKD screening recommended; "
                "no confirmed diagnosis present in this payload."
            ),
        },
    ],
    "notes": (
        "T2DM and hypertension confirmed. CKD not yet diagnosed but screening flagged. "
        "Consider STRATIFIER role for CKD if added to pipeline."
    ),
}


def main() -> None:
    print(f"generating report for {SAMPLE_PAYLOAD['query']}")
    try:
        report = generate_report(SAMPLE_PAYLOAD)
        print(json.dumps(report, indent=2))
    except Exception as exc:
        print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
