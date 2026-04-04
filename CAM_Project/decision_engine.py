from collections import defaultdict
from typing import Any

from pipeline_policy import (
    DEFAULT_CANDIDATE_POOL_LIMIT,
    DEFAULT_INCLUDE_CANDIDATES_CAP,
    DEFAULT_REVIEW_CANDIDATES_CAP,
    DEFAULT_SPECIFIC_VARIANTS_CAP,
)
from scoring_rules import (
    allows_causal_language,
    candidate_role,
    compute_authority_score,
    compute_centrality_component,
    compute_coverage_component,
    compute_fusion_component,
    compute_modifier_component,
    compute_primary_alignment_component,
    compute_query_misalignment_penalty,
    compute_subtype_clutter_penalty,
    condition_matches,
    dominant_condition,
    has_primary_modifier_alignment,
    is_causal_variant,
    is_generic_anchor,
    is_narrow_subtype_variant,
    is_pregnancy_variant,
    is_preferred_obesity_refinement,
    major_conditions,
    query_context,
    serialize_candidate_output,
)


def assign_confidence(candidate: dict[str, Any]) -> str:
    in_qof = bool(candidate.get("in_qof", False))
    in_opencodelists = bool(candidate.get("in_opencodelists", False))
    usage_count_nhs = float(candidate.get("usage_count_nhs", 0.0))

    if in_qof:
        return "HIGH"
    if in_opencodelists and usage_count_nhs >= 29097:
        return "HIGH"
    if in_opencodelists or usage_count_nhs >= 6730:
        return "MEDIUM"
    return "REVIEW"


class DecisionEngine:
    def assign_final_decisions(
        self,
        fused_candidates: list[dict[str, Any]],
        structured_query: dict[str, Any],
        top_k: int,
    ) -> dict[str, list[dict[str, Any]]]:
        conditions = major_conditions(structured_query)
        query_ctx = query_context(structured_query)
        multimorbidity = len(conditions) > 1
        decisions: list[dict[str, Any]] = []

        for candidate in fused_candidates:
            decision = dict(candidate)
            decision["confidence_tier"] = assign_confidence(candidate)
            matched = condition_matches(candidate, conditions)
            dominant = dominant_condition(candidate, matched, conditions)
            authority_component = compute_authority_score(candidate)
            coverage_component = compute_coverage_component(candidate, matched, multimorbidity)
            fusion_component = compute_fusion_component(candidate)
            primary_alignment_component = compute_primary_alignment_component(dominant, matched, conditions)
            centrality_component = compute_centrality_component(candidate, matched, conditions)
            modifier_component = compute_modifier_component(candidate, matched, conditions, query_ctx)
            subtype_penalty = compute_subtype_clutter_penalty(candidate, matched, multimorbidity, query_ctx)
            query_misalignment_penalty = compute_query_misalignment_penalty(candidate, query_ctx)

            decision["matched_conditions"] = matched
            decision["dominant_condition"] = dominant
            decision["ranking_components"] = {
                "authority_component": round(authority_component, 6),
                "fusion_component": round(fusion_component, 6),
                "coverage_component": round(coverage_component, 6),
                "primary_alignment_component": round(primary_alignment_component, 6),
                "centrality_component": round(centrality_component, 6),
                "modifier_component": round(modifier_component, 6),
                "diversity_component": 0.0,
                "family_policy_component": 0.0,
                "subtype_clutter_penalty": round(subtype_penalty, 6),
                "query_misalignment_penalty": round(query_misalignment_penalty, 6),
            }
            decision["candidate_role"] = candidate_role(
                candidate,
                matched,
                conditions,
                query_ctx,
                decision["ranking_components"],
            )
            decision["base_presentation_score"] = round(
                authority_component
                + fusion_component
                + coverage_component
                + primary_alignment_component
                + centrality_component
                + modifier_component
                - subtype_penalty
                - query_misalignment_penalty,
                6,
            )
            decision["presentation_score"] = decision["base_presentation_score"]
            decision["authority_bucket"] = (
                0
                if bool(candidate.get("in_qof", False))
                else 1
                if bool(candidate.get("in_opencodelists", False))
                or float(candidate.get("usage_count_nhs", 0.0)) >= 6730
                else 2
            )
            decisions.append(decision)

        suppressed_decisions = [row for row in decisions if row.get("candidate_role") == "suppress"]
        eligible_decisions = [row for row in decisions if row.get("candidate_role") != "suppress"]

        ranked = sorted(
            eligible_decisions,
            key=lambda row: (
                -row["base_presentation_score"],
                -row["ranking_components"]["primary_alignment_component"],
                -row["ranking_components"]["centrality_component"],
                -row["presentation_score"],
                -row["ranking_components"]["authority_component"],
                row["snomed_code"],
            ),
        )

        selected: list[dict[str, Any]] = []
        selected_codes: set[str] = set()
        dominant_condition_counts: dict[str, int] = defaultdict(int)
        covered_conditions: set[str] = set()
        generic_anchor_conditions: set[str] = set()

        primary_condition = conditions[0] if conditions else ""

        def best_for_condition(condition: str) -> dict[str, Any] | None:
            branch_candidates = [
                row
                for row in ranked
                if row["snomed_code"] not in selected_codes
                and condition in row.get("matched_conditions", [])
                and not is_pregnancy_variant(row)
                and (allows_causal_language(query_ctx) or not is_causal_variant(row))
            ]
            if not branch_candidates:
                return None
            if condition == primary_condition:
                modifier_aligned = [
                    row
                    for row in branch_candidates
                    if has_primary_modifier_alignment(
                        row,
                        row.get("matched_conditions", []),
                        conditions,
                        query_ctx,
                    )
                ]
                if modifier_aligned:
                    return modifier_aligned[0]
                preferred_obesity = [row for row in branch_candidates if is_preferred_obesity_refinement(row)]
                if preferred_obesity:
                    return preferred_obesity[0]
            return branch_candidates[0] if branch_candidates else None

        if primary_condition:
            chosen = best_for_condition(primary_condition)
            if chosen is not None:
                chosen["ranking_components"]["family_policy_component"] = round(
                    chosen["ranking_components"]["family_policy_component"] + 0.65,
                    6,
                )
                chosen["presentation_score"] = round(chosen["base_presentation_score"] + 0.65, 6)
                selected.append(chosen)
                selected_codes.add(chosen["snomed_code"])
                dominant_condition_counts[chosen["dominant_condition"]] += 1
                covered_conditions.update(chosen.get("matched_conditions", []))
                if is_generic_anchor(chosen, primary_condition):
                    generic_anchor_conditions.add(primary_condition)

        if multimorbidity:
            for condition in conditions[1:]:
                chosen = best_for_condition(condition)
                if chosen is not None:
                    chosen["ranking_components"]["family_policy_component"] = round(
                        chosen["ranking_components"]["family_policy_component"] + 0.35,
                        6,
                    )
                    chosen["presentation_score"] = round(chosen["base_presentation_score"] + 0.35, 6)
                    selected.append(chosen)
                    selected_codes.add(chosen["snomed_code"])
                    dominant_condition_counts[chosen["dominant_condition"]] += 1
                    covered_conditions.update(chosen.get("matched_conditions", []))
                    if is_generic_anchor(chosen, condition):
                        generic_anchor_conditions.add(condition)

        max_per_condition = 2 if multimorbidity else top_k
        for row in ranked:
            if len(selected) >= top_k:
                break
            if row["snomed_code"] in selected_codes:
                continue
            if is_pregnancy_variant(row):
                continue
            if is_causal_variant(row) and not allows_causal_language(query_ctx):
                continue
            if is_narrow_subtype_variant(row, query_ctx) and any(
                condition in generic_anchor_conditions for condition in row.get("matched_conditions", [])
            ):
                continue

            dominant = row["dominant_condition"]
            if multimorbidity and dominant_condition_counts[dominant] >= max_per_condition:
                continue

            diversity_component = 0.0
            if multimorbidity and row.get("matched_conditions"):
                uncovered = [
                    condition
                    for condition in row["matched_conditions"]
                    if condition not in covered_conditions
                ]
                if uncovered:
                    diversity_component = 0.25

            family_policy_component = 0.0
            if primary_condition and row["dominant_condition"] == primary_condition:
                family_policy_component += 0.15

            row["ranking_components"]["diversity_component"] = round(diversity_component, 6)
            row["ranking_components"]["family_policy_component"] = round(
                row["ranking_components"]["family_policy_component"] + family_policy_component,
                6,
            )
            row["presentation_score"] = round(
                row["base_presentation_score"] + diversity_component + family_policy_component,
                6,
            )
            selected.append(row)
            selected_codes.add(row["snomed_code"])
            dominant_condition_counts[dominant] += 1
            covered_conditions.update(row.get("matched_conditions", []))
            for condition in row.get("matched_conditions", []):
                if is_generic_anchor(row, condition):
                    generic_anchor_conditions.add(condition)

        cleaned_selected: list[dict[str, Any]] = []
        cleaned_codes: set[str] = set()
        for row in selected:
            if is_narrow_subtype_variant(row, query_ctx) and any(
                condition in generic_anchor_conditions for condition in row.get("matched_conditions", [])
            ):
                continue
            cleaned_selected.append(row)
            cleaned_codes.add(row["snomed_code"])

        for row in ranked:
            if len(cleaned_selected) >= top_k:
                break
            if row["snomed_code"] in cleaned_codes:
                continue
            if is_pregnancy_variant(row):
                continue
            if is_causal_variant(row) and not allows_causal_language(query_ctx):
                continue
            if is_narrow_subtype_variant(row, query_ctx) and any(
                condition in generic_anchor_conditions for condition in row.get("matched_conditions", [])
            ):
                continue
            cleaned_selected.append(row)
            cleaned_codes.add(row["snomed_code"])

        candidate_pool_limit = min(top_k, DEFAULT_CANDIDATE_POOL_LIMIT)
        final_candidates = cleaned_selected[:candidate_pool_limit]
        grouped_output: dict[str, list[dict[str, Any]]] = {
            "include_candidates": [],
            "review_candidates": [],
            "specific_variants": [],
            "suppressed_candidates": [],
        }

        for index, row in enumerate(final_candidates, start=1):
            serialized = serialize_candidate_output(row, index=index)
            role = row.get("candidate_role")
            if role == "core":
                if len(grouped_output["include_candidates"]) < DEFAULT_INCLUDE_CANDIDATES_CAP:
                    grouped_output["include_candidates"].append(serialized)
                elif len(grouped_output["review_candidates"]) < DEFAULT_REVIEW_CANDIDATES_CAP:
                    grouped_output["review_candidates"].append(serialized)
            elif role == "specific":
                if len(grouped_output["specific_variants"]) < DEFAULT_SPECIFIC_VARIANTS_CAP:
                    grouped_output["specific_variants"].append(serialized)
            elif len(grouped_output["review_candidates"]) < DEFAULT_REVIEW_CANDIDATES_CAP:
                grouped_output["review_candidates"].append(serialized)

        for row in sorted(
            suppressed_decisions,
            key=lambda item: (-item["base_presentation_score"], item["snomed_code"]),
        )[:10]:
            grouped_output["suppressed_candidates"].append(serialize_candidate_output(row))

        return grouped_output
