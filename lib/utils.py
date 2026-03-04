import io
from typing import Any, Iterable

import aiohttp
import cv2
import numpy as np
from fastapi import UploadFile
from loguru import logger
from PIL import Image

from lib.constants import (
    ALLOWED_IMAGE_TYPES,
    DEFAULT_HTTP_MAX_FIELD_SIZE,
    DEFAULT_HTTP_MAX_LINE_SIZE,
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    DEFAULT_HTTP_USER_AGENT,
    MAX_FILE_SIZE,
    MIN_CONFIDENCE,
    MIN_FILE_SIZE,
    REMOTE_MIN_IMAGE_MAGIC_BYTES,
)


def create_http_session(limit_per_host: int, limit: int) -> aiohttp.ClientSession:
    connector = aiohttp.TCPConnector(limit_per_host=limit_per_host, limit=limit)
    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=DEFAULT_HTTP_TIMEOUT_SECONDS),
        headers={"User-Agent": DEFAULT_HTTP_USER_AGENT},
        connector=connector,
        max_line_size=DEFAULT_HTTP_MAX_LINE_SIZE,
        max_field_size=DEFAULT_HTTP_MAX_FIELD_SIZE,
    )


def is_allowed_image_upload(file: UploadFile | None) -> bool:
    return bool(file and file.content_type in ALLOWED_IMAGE_TYPES)


def is_valid_file_size(
    file_data: bytes, min_size: int = MIN_FILE_SIZE, max_size: int = MAX_FILE_SIZE
) -> bool:
    return min_size <= len(file_data) <= max_size


def decode_image_bytes_to_opencv(image_data: bytes) -> np.ndarray | None:
    try:
        pil_img = Image.open(io.BytesIO(image_data))
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        return None


def extract_single_face_from_image_data(
    image_data: bytes,
    analyzer: Any,
    min_confidence: float = MIN_CONFIDENCE,
) -> tuple[Any | None, np.ndarray | None, str | None]:
    try:
        opencv_img = decode_image_bytes_to_opencv(image_data)
        if opencv_img is None:
            return None, None, "Face processing failed"

        faces = analyzer.get(opencv_img)

        if len(faces) == 0:
            return None, None, "No faces detected"

        if len(faces) > 1:
            return None, None, "Multiple faces detected"

        face = faces[0]

        if float(face.det_score) < min_confidence:
            return None, None, "Low confidence score"

        return face, opencv_img, None

    except Exception as e:
        logger.error(f"Face extraction failed: {e}")
        return None, None, "Face processing failed"


def build_face_embeddings_from_faces(
    faces: list[Any], min_confidence: float = MIN_CONFIDENCE
) -> list[dict[str, Any]]:
    face_embeddings: list[dict[str, Any]] = []

    for face_index, face in enumerate(faces):
        if not hasattr(face, "embedding") or face.embedding is None:
            continue

        if float(face.det_score) < min_confidence:
            continue

        face_embeddings.append(
            {
                "face_id": face_index,
                "embedding": face.embedding.tolist(),
                "bbox": face.bbox.tolist() if hasattr(face, "bbox") else [],
            }
        )

    return face_embeddings


def compute_cosine_similarity(embedding1: Any, embedding2: Any) -> float:
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)

    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return float(similarity)


def remove_keys(data: dict[str, Any], keys: Iterable[str]) -> dict[str, Any]:
    keys_set = set(keys)
    return {k: v for k, v in data.items() if k not in keys_set}


def unique_preserve_order(items: Iterable[Any]) -> list[Any]:
    return list(dict.fromkeys(items))


def content_type_matches_any(content_type: str, allowed_types: set[str]) -> bool:
    content_type_lower = content_type.lower()
    return any(allowed_type in content_type_lower for allowed_type in allowed_types)


def is_valid_image_magic_bytes(image_data: bytes) -> bool:
    if len(image_data) < REMOTE_MIN_IMAGE_MAGIC_BYTES:
        return False

    return (
        image_data[:2] == b"\xff\xd8"
        or image_data[:8] == b"\x89PNG\r\n\x1a\n"
        or image_data[:6] in (b"GIF87a", b"GIF89a")
        or (image_data[:4] == b"RIFF" and image_data[8:12] == b"WEBP")
    )
