"""retrieval_engine.py — hybrid semantic + BM25 retrieval, CE-first, no leakage.

Source: Playground.ipynb, Section 3 (cell t-BbAUVpX9uf).
Logic is unchanged from the notebook.

Key changes from original:
1. _lexical_overlap checks candidate TERM only (not text_for_embedding).
2. _term_precision added as precision-oriented signal.
3. specificity_score is NOT computed here — it is produced only by HierarchyEnricher.
4. Retrieval sort decoupled from query_weight (sorts by adjusted_retrieval_score).
5. Evidence bonus (+0.05) removed from adj_score.
6. BM25 indexed on term only (text_for_bm25), not enriched metadata.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

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


# ── DataLoader ────────────────────────────────────────────────────────────────
# Unchanged except: num_parents and num_children are now explicitly loaded and
# normalised so HierarchyEnricher can use them downstream.
# text_for_bm25 is now term-only (no enriched metadata).

class DataLoader:
    _shared_df: pd.DataFrame | None                       = None
    _shared_df_by_code: pd.DataFrame | None               = None
    _shared_bm25: BM25Okapi | None                        = None
    _shared_collection                                     = None
    _shared_embedding_model: SentenceTransformer | None   = None

    def __init__(self, config: Any):
        self.config = config

    @staticmethod
    def _safe_text(value: Any) -> str:
        return "" if pd.isna(value) else str(value).strip()

    @staticmethod
    def _column_or_default(df: pd.DataFrame, column: str, default: Any) -> pd.Series:
        if column in df.columns:
            return df[column]
        return pd.Series([default] * len(df), index=df.index)

    def get_dataframe(self) -> pd.DataFrame | None:
        if DataLoader._shared_df is not None:
            return DataLoader._shared_df

        try:
            df = pd.read_csv(self.config.snomed_path)
        except Exception as exc:
            logger.warning("Could not load SNOMED CSV from %s: %s", self.config.snomed_path, exc)
            return None

        if "snomed_code" not in df.columns or "term" not in df.columns:
            logger.warning("SNOMED CSV is missing required columns.")
            return None

        # ── Core cleanup ──────────────────────────────────────────────────────
        df = df.dropna(subset=["snomed_code", "term"]).copy()

        df["snomed_code"] = df["snomed_code"].astype(str).str.strip()
        df["term"]        = df["term"].astype(str).str.strip()

        df["semantic_tag"]     = self._column_or_default(df, "semantic_tag", "").fillna("").astype(str)
        df["in_qof"]           = self._column_or_default(df, "in_qof", False).fillna(False).astype(bool)
        df["in_opencodelists"] = self._column_or_default(df, "in_opencodelists", False).fillna(False).astype(bool)

        df["log_usage_nhs"] = pd.to_numeric(
            self._column_or_default(df, "log_usage_nhs", 0.0),
            errors="coerce"
        ).fillna(0.0)

        if "usage_count_nhs" not in df.columns:
            df["usage_count_nhs"] = np.exp(df["log_usage_nhs"]) - 1

        df["usage_count_nhs"] = pd.to_numeric(
            df["usage_count_nhs"],
            errors="coerce"
        ).fillna(0.0)

        # ── Keep metadata (NOT used in retrieval anymore) ─────────────────────
        df["opencodelist_clinical_areas"] = self._column_or_default(
            df, "opencodelist_clinical_areas", ""
        ).fillna("").astype(str)

        df["qof_cluster_description"] = self._column_or_default(
            df, "qof_cluster_description", ""
        ).fillna("").astype(str)

        # ── Hierarchy fields (used downstream ONLY) ───────────────────────────
        df["num_parents"] = pd.to_numeric(
            self._column_or_default(df, "num_parents", 0),
            errors="coerce"
        ).fillna(0).astype(int)

        df["num_children"] = pd.to_numeric(
            self._column_or_default(df, "num_children", 0),
            errors="coerce"
        ).fillna(0).astype(int)

        # ── Deduplicate ───────────────────────────────────────────────────────
        df = df.drop_duplicates(subset=["snomed_code"]).copy()

        # ── CLEAN TEXT FIELDS (THIS IS THE FIX) ──────────────────────────────

        # Used for semantic retrieval (clean, clinical only)
        df["text_for_embedding"] = (
            df["term"].apply(self._safe_text) + " | " +
            df["semantic_tag"].apply(self._safe_text)
        ).str.strip()

        # Used for BM25 (STRICT term-level only — no enriched metadata)
        df["text_for_bm25"] = (
            df["term"].apply(self._safe_text)
        ).str.strip()

        # ── Cache ─────────────────────────────────────────────────────────────
        DataLoader._shared_df = df
        DataLoader._shared_df_by_code = df.set_index("snomed_code", drop=False)

        logger.info("Loaded %s SNOMED records.", len(df))

        return DataLoader._shared_df

    def get_dataframe_by_code(self) -> pd.DataFrame | None:
        if DataLoader._shared_df_by_code is None:
            self.get_dataframe()
        return DataLoader._shared_df_by_code

    def get_bm25(self) -> BM25Okapi | None:
        if DataLoader._shared_bm25 is not None:
            return DataLoader._shared_bm25
        df = self.get_dataframe()
        if df is None:
            return None
        # BM25 indexed on term-only text (text_for_bm25), NOT text_for_embedding
        tokenized_corpus = [
            str(t).lower().split()
            for t in df["text_for_bm25"].fillna("").tolist()
        ]
        DataLoader._shared_bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Initialised BM25 over %s records (term-only).", len(tokenized_corpus))
        return DataLoader._shared_bm25

    def get_collection(self):
        if DataLoader._shared_collection is not None:
            return DataLoader._shared_collection
        try:
            chroma_dir = Path(self.config.chroma_persist_dir)
            if chroma_dir.exists() and any(chroma_dir.iterdir()):
                logger.info("Loading existing Chroma DB from %s.", chroma_dir)
            else:
                logger.info("Chroma DB not found at %s — a fresh instance will be created.", chroma_dir)
            client = chromadb.PersistentClient(path=str(chroma_dir))
            DataLoader._shared_collection = client.get_or_create_collection(
                name=self.config.chroma_collection_name
            )
            return DataLoader._shared_collection
        except Exception as exc:
            logger.warning(
                "Could not load/create Chroma collection from %s: %s",
                self.config.chroma_persist_dir, exc,
            )
            return None

    def get_embedding_model(self) -> SentenceTransformer | None:
        if DataLoader._shared_embedding_model is not None:
            return DataLoader._shared_embedding_model
        try:
            embeddings_dir   = Path(self.config.embeddings_dir)
            model_cache_name = "models--" + self.config.embedding_model_name.replace("/", "--")
            is_direct_model  = embeddings_dir.exists() and (
                (embeddings_dir / "modules.json").exists() or
                (embeddings_dir / "config.json").exists()
            )
            is_cached_model  = embeddings_dir.exists() and (
                embeddings_dir / model_cache_name
            ).exists()
            if is_direct_model:
                logger.info("Loading direct local embedding model from %s.", embeddings_dir)
                DataLoader._shared_embedding_model = SentenceTransformer(str(embeddings_dir))
            elif is_cached_model:
                logger.info("Found cached embedding model — reusing.")
                DataLoader._shared_embedding_model = SentenceTransformer(
                    self.config.embedding_model_name, cache_folder=str(embeddings_dir)
                )
            else:
                logger.info("Downloading/initialising embedding model %s...", self.config.embedding_model_name)
                DataLoader._shared_embedding_model = SentenceTransformer(
                    self.config.embedding_model_name, cache_folder=str(embeddings_dir)
                )
            return DataLoader._shared_embedding_model
        except Exception as exc:
            logger.warning("Could not load embedding model: %s", exc)
            return None


# ── HybridRetriever (clean, CE-first, no leakage) ─────────────────────────────

class HybridRetriever:
    def __init__(self, config: Any, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader

    @staticmethod
    def _semantic_tag_weight(semantic_tag: str, query_profile: dict[str, Any]) -> float:
        tag = str(semantic_tag).strip().lower()
        preferred = set(query_profile.get("preferred_tags", []))
        tolerated = set(query_profile.get("tolerated_tags", []))
        blocked   = set(query_profile.get("blocked_tags", []))

        if tag in blocked:
            return 0.0
        if tag in preferred:
            return 1.15
        if tag in tolerated:
            return 0.75

        if tag in {"body structure", "procedure", "event", "morphologic abnormality"}:
            return 0.4

        return 0.55

    @staticmethod
    def _lexical_overlap(query_terms: set[str], candidate_term: str) -> float:
        """Recall-oriented: how many query tokens appear in the candidate term?
        Checks term-only text — NOT text_for_embedding."""
        if not query_terms:
            return 0.0
        term_tokens = tokenize_text(candidate_term)
        if not term_tokens:
            return 0.0
        return len(query_terms & term_tokens) / len(query_terms)

    @staticmethod
    def _term_precision(query_terms: set[str], candidate_term: str) -> float:
        """Precision-oriented: what fraction of the candidate's tokens are query tokens?
        Penalises compound concepts without any disease-specific rules."""
        if not query_terms:
            return 0.0
        term_tokens = tokenize_text(candidate_term)
        if not term_tokens:
            return 0.0
        return len(query_terms & term_tokens) / len(term_tokens)

    @staticmethod
    def _should_suppress(term: str, query_terms: set[str]) -> bool:
        lowered = term.lower()

        HISTORY_TERMS   = {"resolved", "remission", "history", "past", "follow", "inactive"}
        PREGNANCY_TERMS = {"pregnancy", "childbirth", "puerperium", "eclampsia"}

        if "resolved" in lowered and not (query_terms & HISTORY_TERMS):
            return True

        if any(t in lowered for t in PREGNANCY_TERMS) and not (query_terms & PREGNANCY_TERMS):
            return True

        return False

    def retrieve(self, job: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        df_by_code = self.data_loader.get_dataframe_by_code()
        df         = self.data_loader.get_dataframe()
        bm25       = self.data_loader.get_bm25()
        collection = self.data_loader.get_collection()
        embedder   = self.data_loader.get_embedding_model()

        if any(r is None for r in (df_by_code, df, bm25, collection, embedder)):
            return []

        query_text    = job["query_text"]
        query_type    = job["query_type"]
        query_weight  = float(job["query_weight"])   # renamed from "weight" — metadata only
        clinical_focus = job["clinical_focus"]
        query_terms   = set(job.get("query_terms", []))
        query_profile = job.get("query_profile", {})

        # ── Semantic retrieval ─────────────────────────────────────────────────
        semantic_scores: dict[str, float] = {}
        try:
            qe = embedder.encode(query_text).tolist()
            sr = collection.query(query_embeddings=[qe], n_results=top_k, include=["distances"])

            for idx, code in enumerate(sr.get("ids", [[]])[0]):
                d = float(sr["distances"][0][idx])
                semantic_scores[str(code)] = 1.0 / (1.0 + d)
        except Exception:
            pass

        # ── BM25 retrieval (TERM ONLY) ─────────────────────────────────────────
        bm25_scores: dict[str, float] = {}
        try:
            query_tokens = list(tokenize_text(query_text))
            raw_scores   = bm25.get_scores(query_tokens)

            top_idx   = np.argsort(raw_scores)[::-1][:top_k]
            max_score = float(raw_scores[top_idx].max()) if len(top_idx) else 0.0

            for row_idx in top_idx:
                code = str(df.iloc[row_idx]["snomed_code"])
                bm25_scores[code] = raw_scores[row_idx] / max_score if max_score > 0 else 0.0
        except Exception:
            pass

        # ── Candidate scoring (STRICT relevance only) ──────────────────────────
        candidates = []

        for snomed_code in set(semantic_scores) | set(bm25_scores):

            if snomed_code not in df_by_code.index:
                continue

            row = df_by_code.loc[snomed_code]

            term         = str(row["term"])
            semantic_tag = str(row.get("semantic_tag", ""))

            if self._should_suppress(term, query_terms):
                continue

            tag_weight = self._semantic_tag_weight(semantic_tag, query_profile)
            if tag_weight == 0.0:
                continue

            sem_score  = semantic_scores.get(snomed_code, 0.0)
            bm25_score = bm25_scores.get(snomed_code, 0.0)

            # ── Base retrieval signal (NO additive mixing) ─────────────────────
            base_retrieval = max(sem_score, bm25_score)

            lex_overlap = self._lexical_overlap(query_terms, term)
            term_prec   = self._term_precision(query_terms, term)

            # ── HARD relevance gate ─────────────────────────────────────────────
            if base_retrieval < 0.15 and lex_overlap == 0.0:
                continue

            # ── Multiplicative scoring (strict filtering) ──────────────────────
            # specificity_score is NOT used here — it is metadata from HierarchyEnricher
            adj_score = (
                base_retrieval *
                tag_weight *
                (0.5 + 0.5 * term_prec) *
                (0.5 + 0.5 * lex_overlap)
            )

            if adj_score <= 0.0:
                continue

            candidates.append({
                "snomed_code": snomed_code,
                "term": term,
                "semantic_tag": semantic_tag,
                "in_qof": bool(row["in_qof"]),
                "in_opencodelists": bool(row["in_opencodelists"]),
                "usage_count_nhs": float(row["usage_count_nhs"]),

                "query_text":    query_text,
                "query_type":    query_type,
                "clinical_focus": clinical_focus,
                "query_weight":  query_weight,

                "semantic_score":             float(sem_score),
                "bm25_score":                 float(bm25_score),
                "lexical_overlap":            float(lex_overlap),
                "term_precision":             float(term_prec),
                "semantic_tag_weight":        float(tag_weight),

                "retrieval_score":            float(base_retrieval),
                "adjusted_retrieval_score":   float(adj_score),
                "weighted_retrieval_score":   float(adj_score * query_weight),
            })

        # ── Sort by PURE retrieval quality (decoupled from query_weight) ──────
        candidates.sort(
            key=lambda c: (
                c["adjusted_retrieval_score"],
                c["term_precision"],
                c["lexical_overlap"],
                c["semantic_score"],
            ),
            reverse=True,
        )

        return candidates[:top_k]
