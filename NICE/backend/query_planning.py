"""query_planning.py — query decomposition and per-condition search job planning.

Source: Playground.ipynb, Section 3 (cell mLiCkec6XBDr).
The planner keeps the notebook's per-condition job model, with one repo-side
refinement: modifiers are only attached to compatible conditions.

Key changes from original:
1. No combined jobs by default (include_combined_jobs=False).
2. Per-condition query_profile — not global.
3. query_weight decoupled from retrieval: stored as metadata only.
   HybridRetriever sorts by adjusted_retrieval_score, not weighted_retrieval_score.
   Field name: "query_weight" (was "weight").
4. Conservative modifier scoping prevents unrelated jobs such as
   "morbid hypertension" and "poorly controlled obesity".
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "by", "for", "from",
    "in", "of", "on", "or", "the", "to", "with",
}


def normalize_query(raw_query: str) -> str:
    return " ".join(str(raw_query).strip().split())


def tokenize_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token and token not in STOPWORDS and len(token) > 1
    }


# ── QueryDecomposer ───────────────────────────────────────────────────────────
# Unchanged from original — the decomposition logic is sound.
# The problems were downstream in how the planner used its output.

class QueryDecomposer:
    def __init__(self, llm_model: str):
        self.llm_model = llm_model
        self._llm = None

    def _get_llm(self):
        if self._llm is False:
            return None
        if self._llm is None:
            try:
                from langchain_ollama import ChatOllama
                self._llm = ChatOllama(model=self.llm_model, temperature=0, format="json")
            except Exception as exc:
                logger.warning("Could not initialise decomposition LLM: %s", exc)
                self._llm = False
        return self._llm if self._llm is not False else None

    @staticmethod
    def _clean_list(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        return list(dict.fromkeys(normalize_query(v) for v in values if normalize_query(v)))

    @staticmethod
    def _split_condition_candidates(text: str) -> list[str]:
        normalized = normalize_query(text)
        if not normalized:
            return []
        protected = normalized
        protected = re.sub(r"\bpoorly controlled\b", "poorly_controlled", protected, flags=re.IGNORECASE)
        protected = re.sub(r"\btype 1\b", "type_1", protected, flags=re.IGNORECASE)
        protected = re.sub(r"\btype 2\b", "type_2", protected, flags=re.IGNORECASE)
        parts = re.split(r"\s*,\s*|\s+\band\b\s+", protected, flags=re.IGNORECASE)
        cleaned = []
        for part in parts:
            restored = (
                part.replace("poorly_controlled", "poorly controlled")
                    .replace("type_1", "type 1")
                    .replace("type_2", "type 2")
            )
            restored = normalize_query(restored)
            if restored:
                cleaned.append(restored)
        return list(dict.fromkeys(cleaned))

    @classmethod
    def _extract_leading_modifiers(cls, segment: str) -> tuple[list[str], str]:
        tokens = normalize_query(segment).split()
        if not tokens:
            return [], ""
        known_modifiers = {
            "severe", "mild", "moderate", "morbid", "central",
            "poorly", "controlled", "uncontrolled",
        }
        modifier_tokens: list[str] = []
        while tokens and tokens[0].lower() in known_modifiers:
            modifier_tokens.append(tokens.pop(0))
        mods = cls._clean_list([" ".join(modifier_tokens)]) if modifier_tokens else []
        return mods, " ".join(tokens)

    def _fallback_decompose(self, query: str) -> dict[str, Any]:
        normalized_query = normalize_query(query)
        modifiers: list[str] = []
        supporting_terms: list[str] = []
        segments = re.split(r"\bwith\b", normalized_query, maxsplit=1, flags=re.IGNORECASE)
        primary_segment = normalize_query(segments[0]) if segments else normalized_query
        tail_segment = normalize_query(segments[1]) if len(segments) > 1 else ""
        head_conditions = self._split_condition_candidates(primary_segment)
        tail_conditions = self._split_condition_candidates(tail_segment) if tail_segment else []
        primary_condition = head_conditions[0] if head_conditions else primary_segment or normalized_query
        secondary_conditions = head_conditions[1:] + tail_conditions
        extracted_modifiers, stripped_primary = self._extract_leading_modifiers(primary_condition)
        if stripped_primary:
            primary_condition = stripped_primary
            modifiers.extend(extracted_modifiers)
        normalized_secondary: list[str] = []
        for secondary in secondary_conditions:
            ext_mods, stripped = self._extract_leading_modifiers(secondary)
            if stripped:
                normalized_secondary.append(stripped)
                supporting_terms.extend(ext_mods)
            elif secondary:
                normalized_secondary.append(secondary)
        return {
            "original_query": normalized_query,
            "primary_condition": primary_condition or normalized_query,
            "secondary_conditions": self._clean_list(normalized_secondary),
            "modifiers": self._clean_list(modifiers),
            "supporting_terms": self._clean_list(supporting_terms),
        }

    def _normalize_schema(self, parsed: dict[str, Any], fallback: dict[str, Any]) -> dict[str, Any]:
        primary = normalize_query(parsed.get("primary_condition", "")) or fallback["primary_condition"]
        normalized = {
            "original_query": fallback["original_query"],
            "primary_condition": primary,
            "secondary_conditions": self._clean_list(parsed.get("secondary_conditions", [])),
            "modifiers": self._clean_list(parsed.get("modifiers", [])),
            "supporting_terms": self._clean_list(parsed.get("supporting_terms", [])),
        }
        if not normalized["secondary_conditions"] and parsed.get("comorbidities"):
            normalized["secondary_conditions"] = self._clean_list(parsed.get("comorbidities", []))
        if (
            normalized["primary_condition"].lower() == fallback["original_query"].lower()
            and not normalized["secondary_conditions"]
            and fallback["secondary_conditions"]
        ):
            return fallback
        return normalized

    def decompose(self, query: str) -> dict[str, Any]:
        fallback = self._fallback_decompose(query)
        llm = self._get_llm()
        if llm is None:
            return fallback
        schema_lines = [
            "{",
            '  "primary_condition": "string",',
            '  "secondary_conditions": ["string"],',
            '  "modifiers": ["string"],',
            '  "supporting_terms": ["string"]',
            "}",
        ]
        prompt = "\n".join([
            "You are a clinical query decomposition assistant.",
            "Return ONLY valid JSON using this exact schema:",
            "\n".join(schema_lines),
            "Guidance: no markdown, no commentary. Keep the schema lean.",
            'User query: "' + query + '"',
        ])
        try:
            response = llm.invoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
            parsed = json.loads(raw_text.strip().replace("```json", "").replace("```", ""))
            return self._normalize_schema(parsed, fallback)
        except Exception as exc:
            logger.warning("Decomposition failed; using fallback. %s", exc)
            return fallback


# ── QueryPlanner ──────────────────────────────────────────────────────────────

class QueryPlanner:
    def __init__(
        self,
        query_type_weights: dict[str, float],
        include_combined_jobs: bool = False,
    ):
        self.query_type_weights = query_type_weights
        # Combined jobs (e.g. "obesity with hypertension") are OFF by default.
        # They cause BM25 and semantic search to surface compound SNOMED concepts
        # that happen to token-match both conditions but are clinically irrelevant
        # as broad anchors for either. Enable only if cross-condition enrichment
        # is explicitly needed and downstream scoring can handle the noise.
        self.include_combined_jobs = include_combined_jobs

    @staticmethod
    def _build_condition_profile(condition_text: str) -> dict[str, Any]:
        """Build a query profile scoped to a single condition's tokens only.

        Previously this was built from all conditions combined, meaning
        "hypertension"'s retrieval job carried obesity tokens in its profile.
        That distorted tag preferences and blocked/tolerated sets for every job.
        """
        tokens = tokenize_text(condition_text)

        infection_terms  = {"infection", "infective", "organism", "bacteria", "bacterial",
                            "virus", "viral", "fungal", "microbiology"}
        medication_terms = {"drug", "medication", "medicine", "tablet", "capsule", "insulin"}
        procedure_terms  = {"procedure", "operation", "surgery", "therapy", "treatment", "screening"}

        preferred_tags = {"disorder", "finding"}
        tolerated_tags = {"situation", "observable entity"}
        blocked_tags   = {
            "organism", "substance", "qualifier value", "occupation",
            "physical object", "environment", "specimen", "attribute", "assessment scale",
        }

        if tokens & infection_terms:
            preferred_tags.add("organism")
            blocked_tags.discard("organism")
        if tokens & medication_terms:
            tolerated_tags.update({"clinical drug", "medicinal product", "substance"})
            blocked_tags.difference_update({"clinical drug", "medicinal product", "substance"})
        if tokens & procedure_terms:
            tolerated_tags.update({"procedure", "regime/therapy"})

        return {
            "tokens": sorted(tokens),
            "preferred_tags": sorted(preferred_tags),
            "tolerated_tags": sorted(tolerated_tags),
            "blocked_tags": sorted(blocked_tags),
        }

    @staticmethod
    def _modifier_applies_to_condition(modifier: str, condition_text: str) -> bool:
        """Return whether a modifier should generate a query for this condition.

        The decomposer exposes modifiers globally, so without this check every
        modifier gets crossed with every condition. That creates noisy jobs such
        as "morbid hypertension" in a query about morbid obesity plus
        hypertension.
        """
        modifier_tokens = tokenize_text(modifier)
        condition_tokens = tokenize_text(condition_text)
        if not modifier_tokens or not condition_tokens:
            return False

        # Avoid generating duplicate variants when the condition already carries
        # the same modifier words.
        if modifier_tokens <= condition_tokens:
            return False

        obesity_terms = {"obesity", "obese", "overweight", "adiposity"}
        diabetes_terms = {"diabetes", "diabetic", "mellitus"}

        obesity_specific_modifiers = {"morbid", "central"}
        control_modifiers = {"poorly", "controlled", "uncontrolled"}
        broad_severity_modifiers = {"severe", "mild", "moderate"}

        if modifier_tokens & obesity_specific_modifiers:
            return bool(condition_tokens & obesity_terms)
        if modifier_tokens & control_modifiers:
            return bool(condition_tokens & diabetes_terms)
        if modifier_tokens & broad_severity_modifiers:
            return True

        # Unknown modifiers are kept for compatibility with future decomposer
        # output rather than silently reducing recall.
        return True

    def build_search_queries(self, structured_query: dict[str, Any]) -> list[dict[str, Any]]:
        primary              = normalize_query(structured_query.get("primary_condition", ""))
        secondary_conditions = structured_query.get("secondary_conditions", [])
        modifiers            = structured_query.get("modifiers", [])
        supporting_terms     = structured_query.get("supporting_terms", [])

        jobs: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        def add_job(
            query_text: str,
            query_type: str,
            clinical_focus: str,
            condition_profile: dict[str, Any],
        ) -> None:
            normalized_text = normalize_query(query_text)
            if not normalized_text:
                return
            key = (normalized_text.lower(), query_type)
            if key in seen:
                return
            seen.add(key)
            jobs.append({
                "query_text":     normalized_text,
                "query_type":     query_type,
                "clinical_focus": clinical_focus,
                # query_weight is metadata for fusion; it must NOT affect retrieval
                # sort order. HybridRetriever sorts by adjusted_retrieval_score only.
                "query_weight":   self.query_type_weights.get(query_type, 1.0),
                "query_terms":    sorted(tokenize_text(normalized_text)),
                # Per-condition profile — scoped only to this condition's tokens.
                "query_profile":  condition_profile,
            })

        # ── Primary condition ──────────────────────────────────────────────────
        if primary:
            primary_profile = self._build_condition_profile(primary)
            add_job(primary, "primary_condition", primary, primary_profile)

            # Modifier variants: "morbid obesity", "poorly controlled diabetes".
            # Modifiers are scoped to clinically compatible conditions.
            for modifier in modifiers:
                if self._modifier_applies_to_condition(modifier, primary):
                    add_job(f"{modifier} {primary}", "modifier", primary, primary_profile)

        # ── Secondary conditions — each fully isolated ─────────────────────────
        # Each secondary condition gets its own bare query with its own profile.
        # This is the key fix: "hypertension" retrieves against SNOMED as a clean
        # single-condition query, without obesity tokens in the query text or profile.
        for secondary in secondary_conditions:
            secondary_profile = self._build_condition_profile(secondary)
            add_job(secondary, "secondary_condition", secondary, secondary_profile)

            for modifier in modifiers:
                if self._modifier_applies_to_condition(modifier, secondary):
                    add_job(f"{modifier} {secondary}", "modifier", secondary, secondary_profile)

        # ── Supporting terms ───────────────────────────────────────────────────
        if primary and supporting_terms:
            primary_profile = self._build_condition_profile(primary)
            for term in supporting_terms:
                add_job(term, "supporting_term", term, primary_profile)

        # ── Combined cross-condition jobs (disabled by default) ────────────────
        # Only enable if you have specific evidence that combined queries improve
        # recall for your use case AND downstream scoring penalises compound hits.
        if self.include_combined_jobs and primary:
            primary_profile = self._build_condition_profile(primary)
            for secondary in secondary_conditions:
                add_job(
                    f"{primary} with {secondary}",
                    "combined",
                    secondary,
                    primary_profile,
                )

        return jobs
