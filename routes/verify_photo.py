import inspireface as isf
from fastapi import APIRouter, File, UploadFile, status
from loguru import logger

from lib.constants import (
    FACE_QUALITY_THRESHOLD,
    MASK_THRESHOLD,
    MATCHING_THRESHOLD,
    MIN_CONFIDENCE,
    WINK_THRESHOLD,
)
from lib.face_analyzer import face_analyzer
from lib.utils import (
    compute_cosine_similarity,
    error_response,
    extract_single_face_from_image_data,
    is_allowed_image_upload,
    is_valid_file_size,
    success_response,
)

isf.launch(resource_path="./models/Megatron")


router = APIRouter()


def analyze_face_feature(opencv_img, feature_option, error_prefix):
    try:
        session = isf.InspireFaceSession(
            feature_option, isf.HF_DETECT_MODE_ALWAYS_DETECT
        )

        session.set_detection_confidence_threshold(MIN_CONFIDENCE)

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


@router.post("/verify-photo")
async def verify_photo(
    original_photo: UploadFile = File(...), verification_photo: UploadFile = File(...)
):
    if not is_allowed_image_upload(original_photo):
        return error_response(
            "Invalid original file type", status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
        )

    if not is_allowed_image_upload(verification_photo):
        return error_response(
            "Invalid verification file type",
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        )

    try:
        original_data = await original_photo.read()

        if not is_valid_file_size(original_data):
            return error_response(
                "Invalid original file size",
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        original_face, _, error_msg = extract_single_face_from_image_data(
            original_data, face_analyzer
        )

        if not original_face:
            return error_response(
                f"Selected photo: {error_msg}",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        verification_data = await verification_photo.read()

        if not is_valid_file_size(verification_data):
            return error_response(
                "Invalid verification file size",
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            )

        verification_face, verification_opencv_img, error_msg = (
            extract_single_face_from_image_data(verification_data, face_analyzer)
        )

        if not verification_face:
            return error_response(
                f"Verification photo: {error_msg}",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        original_embedding = original_face.embedding

        verification_embedding = verification_face.embedding

        similarity = compute_cosine_similarity(
            original_embedding, verification_embedding
        )

        logger.info(f"Photo verification similarity: {similarity}")

        is_verified = similarity >= MATCHING_THRESHOLD

        if not is_verified:
            return error_response(
                "Verification failed: Faces do not match",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        mask_score = check_mask_detection(verification_opencv_img)

        if mask_score is None:
            return error_response(
                "Verification failed: Unable to check mask detection",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        logger.info(f"Mask confidence: {mask_score}")

        wearing_mask = mask_score > MASK_THRESHOLD

        if wearing_mask:
            return error_response(
                "Verification failed: Please remove your mask",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        quality_score = check_face_quality(verification_opencv_img)

        if quality_score is None:
            return error_response(
                "Verification failed: Unable to check face quality",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        logger.info(f"Face quality score: {quality_score}")

        if quality_score < FACE_QUALITY_THRESHOLD:
            return error_response(
                "Verification failed: Poor face quality",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        left_eye_open, right_eye_open = check_winking(verification_opencv_img)

        if left_eye_open is None or right_eye_open is None:
            return error_response(
                "Verification failed: Unable to check eye status",
                status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        is_winking = (left_eye_open and not right_eye_open) or (
            not left_eye_open and right_eye_open
        )

        logger.info(f"Winking detected: {is_winking}")

        if not is_winking:
            return error_response(
                "Verification failed: Please wink with one eye",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        return success_response(
            "Verification successful", status_code=status.HTTP_200_OK
        )

    except Exception as e:
        logger.error(f"Error verifying photo: {e}")
        return error_response(
            "Failed to verify photo",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
