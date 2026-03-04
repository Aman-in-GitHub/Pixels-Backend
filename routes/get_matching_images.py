import asyncio
import atexit
import base64
import io
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, status
from loguru import logger
from NudeNetv2 import NudeClassifier
from PIL import Image

from lib.constants import (
    DEFAULT_ASYNC_SEMAPHORE_LIMIT,
    DEFAULT_THREAD_POOL_WORKERS,
    IMAGE_MATCH_EXCLUDED_FIELDS,
    MATCHING_THRESHOLD,
    MAX_FILE_SIZE,
    MAX_MATCHING_RESULTS,
)
from lib.face_analyzer import face_analyzer
from lib.faiss_manager import images_faiss_manager
from lib.utils import (
    create_http_session,
    error_response,
    extract_single_face_from_image_data,
    is_allowed_image_upload,
    is_valid_file_size,
    remove_keys,
    success_response,
)

router = APIRouter()

http_session = None

semaphore = asyncio.Semaphore(DEFAULT_ASYNC_SEMAPHORE_LIMIT)

nude_classifier = NudeClassifier()

executor = ThreadPoolExecutor(max_workers=DEFAULT_THREAD_POOL_WORKERS)


def cleanup_resources():
    executor.shutdown(wait=True)


atexit.register(cleanup_resources)


async def get_http_session():
    global http_session

    if not http_session or http_session.closed:
        http_session = create_http_session(limit_per_host=20, limit=120)

    return http_session


def calculate_nsfw_score(opencv_image):
    try:
        result = nude_classifier.classify(opencv_image)

        return float(result.get("unsafe", 0.0))

    except Exception:
        return 0.0


def process_image_with_bbox(image_data, bbox=None):
    try:
        img_array = np.frombuffer(image_data, np.uint8)

        opencv_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if opencv_img is None:
            return None

        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)

            cv2.rectangle(opencv_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        rgb_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(rgb_img)

        nsfw_score = calculate_nsfw_score(opencv_img)

        buffer = io.BytesIO()

        pil_img.save(buffer, format="JPEG", quality=80)

        img_b64 = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_b64}", nsfw_score

    except Exception as e:
        logger.warning(f"Image processing failed: {e}")
        return None


async def download_and_process_image(url, bbox):
    async with semaphore:
        try:
            session = await get_http_session()

            async with session.get(url) as response:
                content_length = response.content_length
                if response.status != 200 or (
                    content_length is not None and content_length > MAX_FILE_SIZE
                ):
                    return None

                image_data = await response.read()

            loop = asyncio.get_running_loop()

            result = await loop.run_in_executor(
                executor, process_image_with_bbox, image_data, bbox
            )

            return result

        except Exception:
            logger.warning(f"Failed to process {url}")
            return None


async def process_single_match(match):
    clean_match = remove_keys(match, IMAGE_MATCH_EXCLUDED_FIELDS)

    bbox = match.get("bbox")

    embedded_url = match.get("embedded_image")

    if embedded_url and bbox and "images" in match:
        img_info = next(
            (img for img in match["images"] if img["src"] == embedded_url), None
        )

        if img_info:
            clean_match["raw_image"] = {
                "src": img_info["src"],
                "width": img_info["width"],
                "height": img_info["height"],
                "alt": img_info["alt"],
            }

            result = await download_and_process_image(embedded_url, bbox)

            if result:
                clean_match["bounded_image"], clean_match["nsfw_score"] = result
            else:
                clean_match["nsfw_score"] = 0.0
        else:
            clean_match["nsfw_score"] = 0.0
    else:
        clean_match["nsfw_score"] = 0.0

    return clean_match


def process_input_image(image_data):
    face, _, error_msg = extract_single_face_from_image_data(image_data, face_analyzer)
    if not face:
        return None, error_msg

    try:
        result = process_image_with_bbox(image_data, face.bbox)

        if not result:
            return None, "Failed to process input image"

        bounded_img, _ = result

        return (face.embedding.tolist(), bounded_img), None

    except Exception as e:
        logger.error(f"Input processing failed: {e}")
        return None, "Face processing failed"


@router.post("/get-matching-images")
async def get_matching_images(file: UploadFile = File(...)):
    if not is_allowed_image_upload(file):
        return error_response(
            "Invalid file type", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )

    try:
        image_data = await file.read()

        if not is_valid_file_size(image_data):
            return error_response(
                "Invalid file size", status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
            )

        loop = asyncio.get_running_loop()

        input_result, error_msg = await loop.run_in_executor(
            executor, process_input_image, image_data
        )

        if not input_result:
            return error_response(
                error_msg or "Face processing failed",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        embedding, bounded_input = input_result

        matches = images_faiss_manager.search_matching_images(
            embedding, k=MAX_MATCHING_RESULTS, threshold=MATCHING_THRESHOLD
        )

        image_matches = [match for match in matches if "embedded_image" in match]

        if not image_matches:
            return error_response("No matching images found", status.HTTP_404_NOT_FOUND)

        processed_matches = await asyncio.gather(
            *[process_single_match(match) for match in image_matches],
            return_exceptions=True,
        )

        successful_matches = [
            match
            for match in processed_matches
            if isinstance(match, dict) and match.get("bounded_image") is not None
        ]

        logger.info(f"Found {len(successful_matches)} matching images")

        return success_response(
            message=f"Found {len(successful_matches)} matching images",
            data={
                "bounded_input_image": bounded_input,
                "matching_images": successful_matches,
                "total_matches": len(successful_matches),
            },
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Error getting matching photos: {e}")
        return error_response(
            "Failed to get matching photos",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
