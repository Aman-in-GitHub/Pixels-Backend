import pickle
from pathlib import Path

import faiss
import numpy as np
from loguru import logger

from lib.supabase_client import supabase


class FAISSManager:
    def __init__(
        self, index_path="face_embeddings.index", metadata_path="face_metadata.pkl"
    ):
        self.index_path = Path(index_path)

        self.metadata_path = Path(metadata_path)

        self.index = None

        self.metadata = {}

        self.embedding_dimension = 512

        self._load_or_create_index()

    def _load_or_create_index(self):
        if self.index_path.exists():
            try:
                self.index = faiss.read_index(str(self.index_path))

                with open(self.metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)

            except Exception as e:
                logger.error(f"Error loading index: {e}")
                self._create_new_index()
        else:
            self._create_new_index()

    def _create_new_index(self):
        self.index = faiss.IndexFlatIP(self.embedding_dimension)

        self.metadata = {}

    def _normalize_embeddings(self, embeddings):
        embeddings_array = np.array(embeddings, dtype=np.float32)

        faiss.normalize_L2(embeddings_array)

        return embeddings_array

    def _get_existing_ids(self):
        local_ids = {
            meta.get("id") for meta in self.metadata.values() if meta.get("id")
        }

        try:
            response = supabase.table("embedded_images").select("id").execute()

            db_ids = (
                {record["id"] for record in response.data} if response.data else set()
            )

            return local_ids | db_ids

        except Exception as e:
            logger.error(f"Error getting existing IDs: {e}")
            return local_ids

    def add_embeddings(self, embeddings, records):
        if not embeddings or not records:
            logger.warning("No embeddings or records provided")
            return

        try:
            existing_ids = self._get_existing_ids()

            new_embeddings, new_records = [], []

            for embedding, record in zip(embeddings, records):
                if record.get("id") not in existing_ids:
                    new_embeddings.append(embedding)

                    new_records.append(record)

            if not new_embeddings:
                logger.info("No new embeddings to add (all duplicates)")
                return

            normalized_embeddings = self._normalize_embeddings(new_embeddings)

            start_idx = self.index.ntotal

            self.index.add(normalized_embeddings)

            for i, record in enumerate(new_records):
                self.metadata[start_idx + i] = record

            self.save_index()

            logger.info(f"Added {len(new_embeddings)} new embeddings")

        except Exception as e:
            logger.error(f"Error adding embeddings: {e}")

    def search_matching_images(self, query_embedding, k=100, threshold=0.3):
        if not self.index or self.index.ntotal == 0:
            logger.warning("FAISS index is empty")
            return []

        try:
            query_normalized = self._normalize_embeddings([query_embedding])

            scores, indices = self.index.search(
                query_normalized, min(k, self.index.ntotal)
            )

            results = []

            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= threshold:
                    result = self.metadata.get(idx, {}).copy()

                    result["match_score"] = float(score)

                    results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error searching index: {e}")
            return []

    def save_index(self):
        try:
            faiss.write_index(self.index, str(self.index_path))

            with open(self.metadata_path, "wb") as f:
                pickle.dump(self.metadata, f)

        except Exception as e:
            logger.error(f"Error saving index: {e}")

    def rebuild_images_index_from_db(self, batch_size=100000):
        logger.info("Rebuilding FAISS images index from database...")

        try:
            self._create_new_index()

            processed = 0

            offset = 0

            while True:
                response = (
                    supabase.table("embedded_images")
                    .select("*")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                if not response.data:
                    break

                embeddings, records = [], []

                for record in response.data:
                    if record.get("embedding"):
                        embeddings.append(record["embedding"])

                        records.append(record)

                if embeddings:
                    normalized = self._normalize_embeddings(embeddings)

                    start_idx = self.index.ntotal

                    self.index.add(normalized)

                    for i, record in enumerate(records):
                        self.metadata[start_idx + i] = record

                    processed += len(embeddings)

                offset += batch_size

                if len(response.data) < batch_size:
                    break

            self.save_index()

            logger.info(f"Rebuilt images index with {processed} embeddings")

        except Exception as e:
            logger.error(f"Error rebuilding images index: {e}")

    def rebuild_video_index_from_db(self, batch_size=100000):
        logger.info("Rebuilding FAISS video index from database...")

        try:
            self._create_new_index()

            processed = 0

            offset = 0

            while True:
                response = (
                    supabase.table("embedded_videos")
                    .select("*")
                    .range(offset, offset + batch_size - 1)
                    .execute()
                )

                if not response.data:
                    break

                embeddings, records = [], []

                for record in response.data:
                    if record.get("embedding"):
                        embeddings.append(record["embedding"])

                        records.append(record)

                if embeddings:
                    normalized = self._normalize_embeddings(embeddings)

                    start_idx = self.index.ntotal

                    self.index.add(normalized)

                    for i, record in enumerate(records):
                        self.metadata[start_idx + i] = record

                    processed += len(embeddings)

                offset += batch_size

                if len(response.data) < batch_size:
                    break

            self.save_index()

            logger.info(f"Rebuilt video index with {processed} embeddings")

        except Exception as e:
            logger.error(f"Error rebuilding video index: {e}")

    def get_stats(self):
        return {
            "total_embeddings": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedding_dimension,
            "index_type": type(self.index).__name__ if self.index else None,
        }


images_faiss_manager = FAISSManager(
    index_path="face_embeddings_images.index", metadata_path="face_metadata_images.pkl"
)

videos_faiss_manager = FAISSManager(
    index_path="face_embeddings_videos.index", metadata_path="face_metadata_videos.pkl"
)
