import io

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from loguru import logger
from PIL import Image

from lib.constants import (
    ALLOWED_IMAGE_TYPES,
    MATCHING_THRESHOLD,
    MAX_FILE_SIZE,
    MAX_MATCHING_RESULTS,
    MIN_CONFIDENCE,
    MIN_FILE_SIZE,
)
from lib.face_analyzer import face_analyzer
from lib.faiss_manager import videos_faiss_manager

router = APIRouter()


def process_input_image(image_data):
    try:
        pil_img = Image.open(io.BytesIO(image_data))

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        faces = face_analyzer.get(opencv_img)

        if len(faces) == 0:
            return None, "No faces detected"
        if len(faces) > 1:
            return None, "Multiple faces detected"

        face = faces[0]

        if face.det_score < MIN_CONFIDENCE:
            return None, "Low confidence score"

        return face.embedding.tolist(), None

    except Exception as e:
        logger.error(f"Input processing failed: {e}")
        return None, "Face processing failed"


@router.post("/get-matching-videos")
async def get_matching_videos(file: UploadFile = File(...)):
    if not file or file.content_type not in ALLOWED_IMAGE_TYPES:
        return {"success": False, "message": "Invalid file"}

    try:
        image_data = await file.read()

        if not (MIN_FILE_SIZE <= len(image_data) <= MAX_FILE_SIZE):
            return {"success": False, "message": "Invalid file size"}

        embedding, error_msg = process_input_image(image_data)

        if not embedding:
            return {"success": False, "message": error_msg}

        matches = videos_faiss_manager.search_matching_images(
            embedding, k=MAX_MATCHING_RESULTS, threshold=MATCHING_THRESHOLD
        )

        video_matches = [match for match in matches if "embedded_video" in match]

        if not video_matches:
            return {"success": False, "message": "No matching videos found"}

        video_groups = {}

        for match in video_matches:
            video_id = match["embedded_video"]

            if video_id not in video_groups:
                video_groups[video_id] = {
                    k: v
                    for k, v in match.items()
                    if k not in ["embedding", "bbox", "confidence", "timestamp"]
                }
                video_groups[video_id]["timestamps"] = []

            if match["timestamp"] not in video_groups[video_id]["timestamps"]:
                video_groups[video_id]["timestamps"].append(match["timestamp"])

        for video_id in video_groups:
            video_groups[video_id]["timestamps"].sort()

            video_groups[video_id]["occurrences"] = len(
                video_groups[video_id]["timestamps"]
            )

        cleaned_matches = list(video_groups.values())

        logger.info(f"Found {len(cleaned_matches)} matching videos")

        return {
            "success": True,
            "matching_videos": cleaned_matches,
            "total_matches": len(cleaned_matches),
            "message": f"Found {len(cleaned_matches)} matching videos",
        }

    except Exception as e:
        logger.error(f"Error getting matching videos: {e}")
        return {"success": False, "message": "Failed to get matching videos"}
