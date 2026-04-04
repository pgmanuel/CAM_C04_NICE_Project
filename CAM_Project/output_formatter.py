from typing import Any


BUCKET_ORDER = [
    "include_candidates",
    "review_candidates",
    "specific_variants",
    "suppressed_candidates",
]


class LLMExplanationFormatter:
    def __init__(self, config: Any):
        self.config = config

    @staticmethod
    def _deterministic_rationale(candidate: dict[str, Any]) -> str:
        evidence = candidate["evidence"]
        reasons = []
        role = candidate.get("candidate_role", "")
        if role == "core":
            reasons.append("central query-aligned concept")
        elif role == "variant":
            reasons.append("relevant narrower variant for analyst review")
        elif role == "specific":
            reasons.append("specialised related concept kept as lower-priority depth")
        elif role == "suppress":
            reasons.append("retained for traceability but suppressed from first-pass review")
        if evidence["in_qof"]:
            reasons.append("QOF-supported code")
        if evidence["in_opencodelists"]:
            reasons.append("present in OpenCodelists")
        if evidence["usage_count_nhs"] > 0:
            reasons.append(f"NHS usage count {int(evidence['usage_count_nhs'])}")
        if not reasons:
            reasons.append("retrieved as a relevant candidate but evidence is limited")
        return "; ".join(reasons)

    def _serialize_candidate(
        self,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "presentation_score": candidate.get("presentation_score"),
            "snomed_code": candidate["snomed_code"],
            "term": candidate["term"],
            "confidence_tier": candidate["confidence_tier"],
            "candidate_role": candidate.get("candidate_role"),
            "rationale": self._deterministic_rationale(candidate),
            "evidence": {
                "in_qof": candidate["evidence"]["in_qof"],
                "in_opencodelists": candidate["evidence"]["in_opencodelists"],
                "usage_count_nhs": candidate["evidence"]["usage_count_nhs"],
            },
        }

    def format_candidates(
        self,
        user_query: str,
        candidate_groups: dict[str, list[dict[str, Any]]],
    ) -> tuple[dict[str, list[dict[str, Any]]], str, dict[str, Any]]:
        del user_query
        grouped_output: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in BUCKET_ORDER}
        for bucket in BUCKET_ORDER:
            for candidate in candidate_groups.get(bucket, []):
                grouped_output[bucket].append(self._serialize_candidate(candidate))

        debug_info = {
            "formatter_enabled": False,
            "schema_validation_error": "formatter disabled by design",
        }
        return grouped_output, "deterministic_formatter_disabled", debug_info
