import logging
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import pandas as pd
from pipeline_policy import (
    RETRIEVAL_HISTORY_EXCEPTION_TERMS,
    RETRIEVAL_PREGNANCY_EXCEPTION_TERMS,
    RETRIEVAL_PREGNANCY_MARKERS,
)
from query_planning import tokenize_text
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


logger = logging.getLogger("IntegratedPipeline")


class DataLoader:
    _shared_df: pd.DataFrame | None = None
    _shared_df_by_code: pd.DataFrame | None = None
    _shared_bm25: BM25Okapi | None = None
    _shared_collection = None
    _shared_embedding_model: SentenceTransformer | None = None

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

        df = df.dropna(subset=["snomed_code", "term"]).copy()
        df["snomed_code"] = df["snomed_code"].astype(str).str.strip()
        df["term"] = df["term"].astype(str).str.strip()
        df["semantic_tag"] = self._column_or_default(df, "semantic_tag", "").fillna("").astype(str)
        df["in_qof"] = self._column_or_default(df, "in_qof", False).fillna(False).astype(bool)
        df["in_opencodelists"] = (
            self._column_or_default(df, "in_opencodelists", False).fillna(False).astype(bool)
        )
        df["log_usage_nhs"] = pd.to_numeric(
            self._column_or_default(df, "log_usage_nhs", 0.0),
            errors="coerce",
        ).fillna(0.0)
        if "usage_count_nhs" not in df.columns:
            df["usage_count_nhs"] = np.exp(df["log_usage_nhs"]) - 1
        df["usage_count_nhs"] = pd.to_numeric(df["usage_count_nhs"], errors="coerce").fillna(0.0)
        df["opencodelist_clinical_areas"] = self._column_or_default(
            df,
            "opencodelist_clinical_areas",
            "",
        ).fillna("").astype(str)
        df["qof_cluster_description"] = self._column_or_default(
            df,
            "qof_cluster_description",
            "",
        ).fillna("").astype(str)
        df = df.drop_duplicates(subset=["snomed_code"]).copy()
        df["text_for_embedding"] = (
            df["term"].apply(self._safe_text)
            + " | "
            + df["semantic_tag"].apply(self._safe_text)
            + " | "
            + df["opencodelist_clinical_areas"].apply(self._safe_text)
            + " | "
            + df["qof_cluster_description"].apply(self._safe_text)
        ).str.strip()

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

        tokenized_corpus = [
            str(text).lower().split() for text in df["text_for_embedding"].fillna("").tolist()
        ]
        DataLoader._shared_bm25 = BM25Okapi(tokenized_corpus)
        logger.info("Initialized BM25 over %s records.", len(tokenized_corpus))
        return DataLoader._shared_bm25

    def get_collection(self):
        if DataLoader._shared_collection is not None:
            return DataLoader._shared_collection

        try:
            chroma_dir = Path(self.config.chroma_persist_dir)
            if chroma_dir.exists() and any(chroma_dir.iterdir()):
                logger.info("Loading existing Chroma DB from '%s'.", chroma_dir)
            else:
                logger.info("Chroma DB not found or empty at '%s'. A fresh instance will be created.", chroma_dir)

            client = chromadb.PersistentClient(path=str(chroma_dir))
            DataLoader._shared_collection = client.get_or_create_collection(name=self.config.chroma_collection_name)
            return DataLoader._shared_collection
        except Exception as exc:
            logger.warning("Could not load/create Chroma collection from %s: %s", self.config.chroma_persist_dir, exc)
            return None

    def get_embedding_model(self) -> SentenceTransformer | None:
        if DataLoader._shared_embedding_model is not None:
            return DataLoader._shared_embedding_model

        try:
            embeddings_dir = Path(self.config.embeddings_dir)
            
            is_direct_model = embeddings_dir.exists() and (
                (embeddings_dir / "modules.json").exists() or 
                (embeddings_dir / "config.json").exists()
            )
            
            cache_model_dir_name = f"models--{self.config.embedding_model_name.replace('/', '--')}"
            is_cached_model = embeddings_dir.exists() and (embeddings_dir / cache_model_dir_name).exists()
            
            if is_direct_model:
                logger.info("Loading existing direct local embedding model from '%s'.", embeddings_dir)
                DataLoader._shared_embedding_model = SentenceTransformer(str(embeddings_dir))
            elif is_cached_model:
                logger.info("Found cached embedding model '%s' in '%s'. Reusing...", self.config.embedding_model_name, embeddings_dir)
                DataLoader._shared_embedding_model = SentenceTransformer(
                    self.config.embedding_model_name, 
                    cache_folder=str(embeddings_dir)
                )
            else:
                logger.info("No usable local resource or cache found at '%s'. Downloading/initialising '%s'...", embeddings_dir, self.config.embedding_model_name)
                DataLoader._shared_embedding_model = SentenceTransformer(
                    self.config.embedding_model_name, 
                    cache_folder=str(embeddings_dir)
                )

            return DataLoader._shared_embedding_model
        except Exception as exc:
            logger.warning("Could not load embedding model '%s': %s", self.config.embedding_model_name, exc)
            return None


class HybridRetriever:
    def __init__(self, config: Any, data_loader: DataLoader):
        self.config = config
        self.data_loader = data_loader

    @staticmethod
    def _semantic_tag_weight(semantic_tag: str, query_profile: dict[str, Any]) -> float:
        tag = str(semantic_tag).strip().lower()
        preferred = set(query_profile.get("preferred_tags", []))
        tolerated = set(query_profile.get("tolerated_tags", []))
        blocked = set(query_profile.get("blocked_tags", []))

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
    def _lexical_overlap(query_terms: set[str], candidate_text: str) -> float:
        if not query_terms:
            return 0.0
        candidate_terms = tokenize_text(candidate_text)
        if not candidate_terms:
            return 0.0
        overlap = len(query_terms & candidate_terms)
        return overlap / max(1, len(query_terms))

    @staticmethod
    def _should_suppress_candidate_term(term: str, query_terms: set[str]) -> bool:
        lowered = str(term).lower()
        if "resolved" in lowered and not (query_terms & RETRIEVAL_HISTORY_EXCEPTION_TERMS):
            return True

        if any(marker in lowered for marker in RETRIEVAL_PREGNANCY_MARKERS) and not (
            query_terms & RETRIEVAL_PREGNANCY_EXCEPTION_TERMS
        ):
            return True

        return False

    def retrieve(self, job: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        df_by_code = self.data_loader.get_dataframe_by_code()
        df = self.data_loader.get_dataframe()
        bm25 = self.data_loader.get_bm25()
        collection = self.data_loader.get_collection()
        embedder = self.data_loader.get_embedding_model()

        if any(resource is None for resource in (df_by_code, df, bm25, collection, embedder)):
            logger.warning("Skipping retrieval because one or more resources are unavailable.")
            return []

        query_text = job["query_text"]
        query_type = job["query_type"]
        query_weight = float(job["weight"])
        clinical_focus = job["clinical_focus"]
        query_terms = set(job.get("query_terms", []))
        query_profile = job.get("query_profile", {})

        semantic_scores: dict[str, float] = {}
        try:
            query_embedding = embedder.encode(query_text).tolist()
            semantic_results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["distances"],
            )
            for idx, snomed_code in enumerate(semantic_results.get("ids", [[]])[0]):
                distance = float(semantic_results["distances"][0][idx])
                semantic_scores[str(snomed_code)] = 1.0 / (1.0 + distance)
        except Exception as exc:
            logger.warning("Semantic retrieval failed for '%s': %s", query_text, exc)

        bm25_scores: dict[str, float] = {}
        try:
            raw_scores = bm25.get_scores(query_text.lower().split())
            top_idx = np.argsort(raw_scores)[::-1][:top_k]
            top_scores = raw_scores[top_idx] if len(top_idx) else np.array([])
            max_score = float(top_scores.max()) if len(top_scores) else 0.0
            for row_idx in top_idx:
                snomed_code = str(df.iloc[row_idx]["snomed_code"])
                bm25_scores[snomed_code] = (
                    float(raw_scores[row_idx]) / max_score if max_score > 0 else 0.0
                )
        except Exception as exc:
            logger.warning("BM25 retrieval failed for '%s': %s", query_text, exc)

        all_codes = set(semantic_scores).union(set(bm25_scores))
        candidates: list[dict[str, Any]] = []
        for snomed_code in all_codes:
            if snomed_code not in df_by_code.index:
                continue
            row = df_by_code.loc[snomed_code]
            semantic_tag = str(row.get("semantic_tag", ""))
            term = str(row["term"])
            if self._should_suppress_candidate_term(term, query_terms):
                continue
            semantic_score = semantic_scores.get(snomed_code, 0.0)
            bm25_score = bm25_scores.get(snomed_code, 0.0)
            retrieval_score = (semantic_score * 0.7) + (bm25_score * 0.3)
            lexical_overlap = self._lexical_overlap(
                query_terms,
                f"{term} {row.get('text_for_embedding', '')}",
            )
            tag_weight = self._semantic_tag_weight(semantic_tag, query_profile)
            has_evidence = bool(row["in_qof"]) or bool(row["in_opencodelists"]) or float(row["usage_count_nhs"]) > 0

            if tag_weight == 0.0:
                continue
            if lexical_overlap == 0.0 and bm25_score == 0.0 and not has_evidence:
                continue

            adjusted_retrieval_score = (
                retrieval_score * tag_weight
                + (lexical_overlap * 0.25)
                + (0.05 if has_evidence else 0.0)
            )
            if adjusted_retrieval_score <= 0.0:
                continue

            candidates.append(
                {
                    "snomed_code": snomed_code,
                    "term": str(row["term"]),
                    "semantic_tag": semantic_tag,
                    "in_qof": bool(row["in_qof"]),
                    "in_opencodelists": bool(row["in_opencodelists"]),
                    "usage_count_nhs": float(row["usage_count_nhs"]),
                    "query_text": query_text,
                    "query_type": query_type,
                    "clinical_focus": clinical_focus,
                    "query_weight": query_weight,
                    "semantic_score": float(semantic_score),
                    "bm25_score": float(bm25_score),
                    "lexical_overlap": float(lexical_overlap),
                    "semantic_tag_weight": float(tag_weight),
                    "retrieval_score": float(retrieval_score),
                    "adjusted_retrieval_score": float(adjusted_retrieval_score),
                    "weighted_retrieval_score": float(adjusted_retrieval_score * query_weight),
                }
            )

        candidates.sort(
            key=lambda candidate: (
                candidate["weighted_retrieval_score"],
                candidate["lexical_overlap"],
                candidate["semantic_score"],
                candidate["bm25_score"],
            ),
            reverse=True,
        )
        return candidates[:top_k]
