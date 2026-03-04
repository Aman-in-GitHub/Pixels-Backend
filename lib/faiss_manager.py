import pickle
import time
from pathlib import Path
from typing import Any, TypeAlias

import faiss
import numpy as np
from loguru import logger

from lib.constants import (
    FAISS_DEFAULT_REBUILD_BATCH_SIZE,
    FAISS_EMBEDDING_DIMENSION,
    MATCHING_THRESHOLD,
)
from lib.db import fetch_embedded_image_ids, fetch_embedding_rows

RecordData: TypeAlias = dict[str, Any]


class FAISSManager:
    def __init__(
        self, index_path="face_embeddings.index", metadata_path="face_metadata.pkl"
    ):
        self.index_path = Path(index_path)

        self.metadata_path = Path(metadata_path)
        self.version_path = self.metadata_path.with_suffix(
            f"{self.metadata_path.suffix}.version"
        )

        self.index: Any | None = None

        self.metadata: dict[int, RecordData] = {}
        self.loaded_version: str | None = self._read_index_version()

        self.embedding_dimension = FAISS_EMBEDDING_DIMENSION

        self._load_or_create_index()

    def _load_or_create_index(self) -> None:
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))

                with open(self.metadata_path, "rb") as f:
                    loaded_metadata = pickle.load(f)

                if isinstance(loaded_metadata, dict):
                    sanitized_metadata: dict[int, RecordData] = {}
                    for key, value in loaded_metadata.items():
                        if not isinstance(value, dict):
                            continue
                        try:
                            sanitized_key = int(key)
                        except (TypeError, ValueError):
                            continue
                        sanitized_metadata[sanitized_key] = value

                    self.metadata = sanitized_metadata
                else:
                    logger.warning(
                        "Invalid metadata format detected; resetting metadata"
                    )
                    self.metadata = {}

            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self) -> None:
        self.index = faiss.IndexFlatIP(self.embedding_dimension)

        self.metadata = {}

    def _read_index_version(self) -> str | None:
        try:
            if not self.version_path.exists():
                return None
            version = self.version_path.read_text(encoding="utf-8").strip()
            return version or None
        except Exception:
            return None

    def _write_index_version(self) -> None:
        version = str(time.time_ns())
        self.version_path.write_text(version, encoding="utf-8")
        self.loaded_version = version

    def _reload_if_index_updated(self) -> None:
        current_version = self._read_index_version()
        if not current_version or current_version == self.loaded_version:
            return

        logger.info(
            f"Detected updated FAISS files at {self.index_path}, reloading index..."
        )
        self._load_or_create_index()
        self.loaded_version = current_version

    def _normalize_embeddings(self, embeddings: list[Any]) -> np.ndarray:
        embeddings_array = np.array(embeddings, dtype=np.float32)

        faiss.normalize_L2(embeddings_array)

        return embeddings_array

    def _require_index(self) -> Any:
        if self.index is None:
            raise RuntimeError("FAISS index is not initialized")
        return self.index

    async def _get_existing_ids(self) -> set[Any]:
        local_ids = {
            meta.get("id") for meta in self.metadata.values() if meta.get("id")
        }

        try:
            db_ids = await fetch_embedded_image_ids()

            return local_ids | db_ids

        except Exception as e:
            logger.error(f"Error getting existing IDs: {e}")
            return local_ids

    async def add_embeddings(
        self, embeddings: list[Any], records: list[RecordData]
    ) -> None:
        if not embeddings or not records:
            logger.warning("No embeddings or records provided")
            return

        try:
            existing_ids = await self._get_existing_ids()

            new_embeddings, new_records = [], []

            for embedding, record in zip(embeddings, records):
                if record.get("id") not in existing_ids:
                    new_embeddings.append(embedding)

                    new_records.append(record)

            if not new_embeddings:
                logger.info("No new embeddings to add (all duplicates)")
                return

            normalized_embeddings = self._normalize_embeddings(new_embeddings)

            index = self._require_index()

            start_idx = index.ntotal

            index.add(normalized_embeddings)

            for i, record in enumerate(new_records):
                self.metadata[start_idx + i] = record

            self.save_index()

            logger.info(f"Added {len(new_embeddings)} new embeddings")

        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")

    def search_matching_images(
        self,
        query_embedding: list[float],
        k: int = 100,
        threshold: float = MATCHING_THRESHOLD,
    ) -> list[RecordData]:
        self._reload_if_index_updated()

        index = self.index
        if index is None or index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        try:
            query_normalized = self._normalize_embeddings([query_embedding])

            scores, indices = index.search(query_normalized, min(k, index.ntotal))

            results: list[RecordData] = []

            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= threshold:
                    metadata_item = self.metadata.get(int(idx), {})
                    result = metadata_item.copy()

                    result["match_score"] = float(score)

                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []

    def save_index(self) -> None:
        try:
            faiss.write_index(self._require_index(), str(self.index_path))

            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)

            self._write_index_version()

        except Exception as e:
            logger.error(f"Error saving index: {e}")

    async def rebuild_images_index_from_db(
        self, batch_size: int = FAISS_DEFAULT_REBUILD_BATCH_SIZE
    ) -> None:
        logger.info("Rebuilding FAISS images index from database...")

        try:
            self._create_new_index()
            index = self._require_index()

            processed = 0

            offset = 0

            while True:
                rows = await fetch_embedding_rows(
                    table_name="embedded_images",
                    limit=batch_size,
                    offset=offset,
                )

                if not rows:
                    break

                embeddings, records = [], []

                for record in rows:
                    embedding = record.get("embedding")
                    if isinstance(embedding, (list, tuple, np.ndarray)):
                        embeddings.append(embedding)
                        records.append(record)

                if embeddings:
                    normalized = self._normalize_embeddings(embeddings)

                    start_idx = index.ntotal

                    index.add(normalized)

                    for i, record in enumerate(records):
                        self.metadata[start_idx + i] = record

                    processed += len(embeddings)

                offset += batch_size

                if len(rows) < batch_size:
                    break

            self.save_index()

            logger.info(f"Rebuilt images index with {processed} embeddings")

        except Exception as e:
            logger.error(f"Error rebuilding images index: {e}")

    async def rebuild_video_index_from_db(
        self, batch_size: int = FAISS_DEFAULT_REBUILD_BATCH_SIZE
    ) -> None:
        logger.info("Rebuilding FAISS video index from database...")

        try:
            self._create_new_index()
            index = self._require_index()

            processed = 0

            offset = 0

            while True:
                rows = await fetch_embedding_rows(
                    table_name="embedded_videos",
                    limit=batch_size,
                    offset=offset,
                )

                if not rows:
                    break

                embeddings, records = [], []

                for record in rows:
                    embedding = record.get("embedding")
                    if isinstance(embedding, (list, tuple, np.ndarray)):
                        embeddings.append(embedding)
                        records.append(record)

                if embeddings:
                    normalized = self._normalize_embeddings(embeddings)

                    start_idx = index.ntotal

                    index.add(normalized)

                    for i, record in enumerate(records):
                        self.metadata[start_idx + i] = record

                    processed += len(embeddings)

                offset += batch_size

                if len(rows) < batch_size:
                    break

            self.save_index()

            logger.info(f"Rebuilt video index with {processed} embeddings")

        except Exception as e:
            logger.error(f"Error rebuilding video index: {e}")

    def get_stats(self) -> dict[str, Any]:
        self._reload_if_index_updated()
        index = self.index
        return {
            "total_embeddings": index.ntotal if index is not None else 0,
            "embedding_dimension": self.embedding_dimension,
            "index_type": type(index).__name__ if index is not None else None,
        }


images_faiss_manager = FAISSManager(
    index_path="face_embeddings_images.index", metadata_path="face_metadata_images.pkl"
)

videos_faiss_manager = FAISSManager(
    index_path="face_embeddings_videos.index", metadata_path="face_metadata_videos.pkl"
)
