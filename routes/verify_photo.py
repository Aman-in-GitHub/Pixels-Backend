import io

import cv2
import inspireface as isf
import numpy as np
from fastapi import APIRouter, File, UploadFile
from loguru import logger
from PIL import Image

from lib.constants import (
    ALLOWED_IMAGE_TYPES,
    FACE_QUALITY_THRESHOLD,
    MASK_THRESHOLD,
    MATCHING_THRESHOLD,
    MAX_FILE_SIZE,
    MIN_CONFIDENCE,
    MIN_FILE_SIZE,
    WINK_THRESHOLD,
)
from lib.face_analyzer import face_analyzer

isf.launch(resource_path="./models/Megatron")


router = APIRouter()


def analyze_face_feature(opencv_img, feature_option, error_prefix):
    try:
        session = isf.InspireFaceSession(
            feature_option, isf.HF_DETECT_MODE_ALWAYS_DETECT
        )

        session.set_detection_confidence_threshold(0.3)

        faces = session.face_detection(opencv_img)

        if len(faces) == 0:
            return None

        face = faces[0]

        extensions = session.face_pipeline(opencv_img, [face], feature_option)

        if len(extensions) > 0:
            return extensions[0]

        return None

    except Exception as e:
        logger.error(f"{error_prefix} failed: {e}")
        return None


def check_mask_detection(opencv_img):
    result = analyze_face_feature(
        opencv_img, isf.HF_ENABLE_MASK_DETECT, "Mask detection check"
    )

    if result is not None:
        return result.mask_confidence

    return None


def check_face_quality(opencv_img):
    result = analyze_face_feature(
        opencv_img, isf.HF_ENABLE_QUALITY, "Face quality check"
    )

    if result is not None:
        return result.quality_confidence

    return None


def check_winking(opencv_img):
    result = analyze_face_feature(
        opencv_img, isf.HF_ENABLE_INTERACTION, "Winking check"
    )

    if result is not None:
        left_eye_open = result.left_eye_status_confidence > WINK_THRESHOLD
        right_eye_open = result.right_eye_status_confidence > WINK_THRESHOLD
        return left_eye_open, right_eye_open

    return None, None


def compute_cosine_similarity(embedding1, embedding2):
    norm1 = np.linalg.norm(embedding1)

    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

    return float(similarity)


def process_input_image(image_data):
    try:
        pil_img = Image.open(io.BytesIO(image_data))

        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        opencv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        faces = face_analyzer.get(opencv_img)

        if len(faces) == 0:
            return None, None, "No faces detected"

        if len(faces) > 1:
            return None, None, "Multiple faces detected"

        face = faces[0]

        if face.det_score < MIN_CONFIDENCE:
            return None, None, "Low confidence score"

        return face, opencv_img, None

    except Exception as e:
        logger.error(f"Input processing failed: {e}")
        return None, None, "Face processing failed"


@router.post("/verify-photo")
async def verify_photo(
    original_photo: UploadFile = File(...), verification_photo: UploadFile = File(...)
):
    if not original_photo or original_photo.content_type not in ALLOWED_IMAGE_TYPES:
        return {
            "success": False,
            "message": "Invalid type",
        }

    if (
        not verification_photo
        or verification_photo.content_type not in ALLOWED_IMAGE_TYPES
    ):
        return {
            "success": False,
            "message": "Invalid type",
        }

    try:
        original_data = await original_photo.read()

        if not (MIN_FILE_SIZE <= len(original_data) <= MAX_FILE_SIZE):
            return {
                "success": False,
                "message": "Invalid size",
            }

        original_face, _, error_msg = process_input_image(original_data)

        if not original_face:
            return {
                "success": False,
                "message": f"Selected photo: {error_msg}",
            }

        verification_data = await verification_photo.read()

        if not (MIN_FILE_SIZE <= len(verification_data) <= MAX_FILE_SIZE):
            return {
                "success": False,
                "message": "Invalid size",
            }

        verification_face, verification_opencv_img, error_msg = process_input_image(
            verification_data
        )

        if not verification_face:
            return {
                "success": False,
                "message": f"Verification photo: {error_msg}",
            }

        original_embedding = original_face.embedding

        verification_embedding = verification_face.embedding

        similarity = compute_cosine_similarity(
            original_embedding, verification_embedding
        )

        logger.info(f"Photo verification similarity: {similarity}")

        is_verified = similarity >= MATCHING_THRESHOLD

        if not is_verified:
            return {
                "success": False,
                "message": "Verification failed: Faces do not match",
            }

        mask_score = check_mask_detection(verification_opencv_img)

        if mask_score is None:
            return {
                "success": False,
                "message": "Verification failed: Unable to check mask detection",
            }

        logger.info(f"Mask confidence: {mask_score}")

        wearing_mask = mask_score > MASK_THRESHOLD

        if wearing_mask:
            return {
                "success": False,
                "message": "Verification failed: Please remove your mask",
            }

        quality_score = check_face_quality(verification_opencv_img)

        if quality_score is None:
            return {
                "success": False,
                "message": "Verification failed: Unable to check face quality",
            }

        logger.info(f"Face quality score: {quality_score}")

        if quality_score < FACE_QUALITY_THRESHOLD:
            return {
                "success": False,
                "message": "Verification failed: Poor face quality",
            }

        left_eye_open, right_eye_open = check_winking(verification_opencv_img)

        if left_eye_open is None or right_eye_open is None:
            return {
                "success": False,
                "message": "Verification failed: Unable to check eye status",
            }

        is_winking = (left_eye_open and not right_eye_open) or (
            not left_eye_open and right_eye_open
        )

        logger.info(f"Winking detected: {is_winking}")

        if not is_winking:
            return {
                "success": False,
                "message": "Verification failed: Please wink with one eye",
            }

        return {
            "success": True,
            "message": "Verification successful",
        }

    except Exception as e:
        logger.error(f"Error verifying photo: {e}")
        return {
            "success": False,
            "message": str(e),
        }
