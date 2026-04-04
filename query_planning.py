import json
import logging
import re
from typing import Any

from langchain_ollama import ChatOllama


logger = logging.getLogger("IntegratedPipeline")


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}


def normalize_query(raw_query: str) -> str:
    return " ".join(str(raw_query).strip().split())


def tokenize_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", str(text).lower())
        if token and token not in STOPWORDS and len(token) > 1
    }


class QueryDecomposer:
    def __init__(self, llm_model: str):
        self.llm_model = llm_model
        self._llm: ChatOllama | None | bool = None

    def _get_llm(self) -> ChatOllama | None:
        if self._llm is False:
            return None
        if self._llm is None:
            try:
                self._llm = ChatOllama(model=self.llm_model, temperature=0, format="json")
            except Exception as exc:
                logger.warning("Could not initialize decomposition LLM: %s", exc)
                self._llm = False
        return self._llm if self._llm is not False else None

    @staticmethod
    def _clean_list(values: Any) -> list[str]:
        if not isinstance(values, list):
            return []
        cleaned = []
        for value in values:
            text = normalize_query(value)
            if text:
                cleaned.append(text)
        return list(dict.fromkeys(cleaned))

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
        cleaned_parts = []
        for part in parts:
            restored = (
                part.replace("poorly_controlled", "poorly controlled")
                .replace("type_1", "type 1")
                .replace("type_2", "type 2")
            )
            restored = normalize_query(restored)
            if restored:
                cleaned_parts.append(restored)
        return list(dict.fromkeys(cleaned_parts))

    @classmethod
    def _extract_leading_modifiers(cls, segment: str) -> tuple[list[str], str]:
        tokens = normalize_query(segment).split()
        if not tokens:
            return [], ""

        known_modifiers = {
            "severe",
            "mild",
            "moderate",
            "morbid",
            "central",
            "poorly",
            "controlled",
            "uncontrolled",
        }
        modifier_tokens: list[str] = []
        while tokens and tokens[0].lower() in known_modifiers:
            modifier_tokens.append(tokens.pop(0))
        return cls._clean_list([" ".join(modifier_tokens)]) if modifier_tokens else [], " ".join(tokens)

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
            extracted_secondary_modifiers, stripped_secondary = self._extract_leading_modifiers(secondary)
            if stripped_secondary:
                normalized_secondary.append(stripped_secondary)
                supporting_terms.extend(extracted_secondary_modifiers)
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

        prompt = f"""
You are a clinical query decomposition assistant.

Return ONLY valid JSON using this exact schema:
{{
  "primary_condition": "string",
  "secondary_conditions": ["string"],
  "modifiers": ["string"],
  "supporting_terms": ["string"]
}}

Guidance:
- Keep the schema lean.
- Put extra medically useful terms into supporting_terms.
- Do not include markdown or commentary.

User query: "{query}"
""".strip()

        try:
            response = llm.invoke(prompt)
            raw_text = response.content if hasattr(response, "content") else str(response)
            parsed = json.loads(raw_text.strip().replace("```json", "").replace("```", ""))
            return self._normalize_schema(parsed, fallback)
        except Exception as exc:
            logger.warning("Decomposition failed; using fallback schema. %s", exc)
            return fallback


class QueryPlanner:
    def __init__(self, query_type_weights: dict[str, float]):
        self.query_type_weights = query_type_weights

    @staticmethod
    def _build_query_profile(structured_query: dict[str, Any]) -> dict[str, Any]:
        combined_text = " ".join(
            [
                structured_query.get("primary_condition", ""),
                *structured_query.get("secondary_conditions", []),
                *structured_query.get("modifiers", []),
                *structured_query.get("supporting_terms", []),
            ]
        )
        tokens = tokenize_text(combined_text)

        infection_terms = {
            "infection",
            "infective",
            "organism",
            "bacteria",
            "bacterial",
            "virus",
            "viral",
            "fungal",
            "microbiology",
        }
        medication_terms = {
            "drug",
            "medication",
            "medicine",
            "tablet",
            "capsule",
            "insulin",
        }
        procedure_terms = {
            "procedure",
            "operation",
            "surgery",
            "therapy",
            "treatment",
            "screening",
        }

        wants_organism = bool(tokens & infection_terms)
        wants_medication = bool(tokens & medication_terms)
        wants_procedure = bool(tokens & procedure_terms)

        preferred_tags = {"disorder", "finding"}
        tolerated_tags = {"situation", "observable entity"}
        blocked_tags = {
            "organism",
            "substance",
            "qualifier value",
            "occupation",
            "physical object",
            "environment",
            "specimen",
            "attribute",
            "assessment scale",
        }

        if wants_organism:
            preferred_tags.add("organism")
            blocked_tags.discard("organism")
        if wants_medication:
            tolerated_tags.update({"clinical drug", "medicinal product", "substance"})
            blocked_tags.difference_update({"clinical drug", "medicinal product", "substance"})
        if wants_procedure:
            tolerated_tags.update({"procedure", "regime/therapy"})

        return {
            "tokens": sorted(tokens),
            "preferred_tags": sorted(preferred_tags),
            "tolerated_tags": sorted(tolerated_tags),
            "blocked_tags": sorted(blocked_tags),
        }

    def build_search_queries(self, structured_query: dict[str, Any]) -> list[dict[str, Any]]:
        primary = normalize_query(structured_query.get("primary_condition", ""))
        secondary_conditions = structured_query.get("secondary_conditions", [])
        modifiers = structured_query.get("modifiers", [])
        supporting_terms = structured_query.get("supporting_terms", [])
        query_profile = self._build_query_profile(structured_query)
        jobs: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()

        def add_job(query_text: str, query_type: str, clinical_focus: str) -> None:
            normalized_text = normalize_query(query_text)
            if not normalized_text:
                return
            key = (normalized_text.lower(), query_type)
            if key in seen:
                return
            seen.add(key)
            jobs.append(
                {
                    "query_text": normalized_text,
                    "query_type": query_type,
                    "clinical_focus": clinical_focus,
                    "weight": self.query_type_weights[query_type],
                    "query_terms": sorted(tokenize_text(normalized_text)),
                    "query_profile": query_profile,
                }
            )

        add_job(primary, "primary_condition", primary)

        for secondary in secondary_conditions:
            add_job(secondary, "secondary_condition", secondary)
            if primary:
                add_job(f"{primary} with {secondary}", "combined", secondary)

        for modifier in modifiers:
            if primary:
                add_job(f"{modifier} {primary}", "modifier", primary)
            for secondary in secondary_conditions:
                add_job(f"{modifier} {secondary}", "modifier", secondary)

        for term in supporting_terms:
            add_job(term, "supporting_term", term)

        return jobs
