from fastapi import APIRouter, File, UploadFile, status
from loguru import logger

from lib.constants import (
    MATCHING_THRESHOLD,
    MAX_MATCHING_RESULTS,
    VIDEO_MATCH_EXCLUDED_FIELDS,
)
from lib.face_analyzer import face_analyzer
from lib.faiss_manager import videos_faiss_manager
from lib.utils import (
    error_response,
    extract_single_face_from_image_data,
    is_allowed_image_upload,
    is_valid_file_size,
    remove_keys,
    success_response,
)

router = APIRouter()


@router.post("/get-matching-videos")
async def get_matching_videos(file: UploadFile = File(...)):
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

        face, _, error_msg = extract_single_face_from_image_data(
            image_data, face_analyzer
        )
        embedding = face.embedding.tolist() if face else None

        if not embedding:
            return error_response(
                error_msg or "Face processing failed",
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            )

        matches = videos_faiss_manager.search_matching_images(
            embedding, k=MAX_MATCHING_RESULTS, threshold=MATCHING_THRESHOLD
        )

        video_matches = [match for match in matches if "embedded_video" in match]

        if not video_matches:
            return error_response("No matching videos found", status.HTTP_404_NOT_FOUND)

        video_groups = {}

        for match in video_matches:
            video_id = match["embedded_video"]

            if video_id not in video_groups:
                video_groups[video_id] = remove_keys(match, VIDEO_MATCH_EXCLUDED_FIELDS)
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

        return success_response(
            message=f"Found {len(cleaned_matches)} matching videos",
            data={
                "matching_videos": cleaned_matches,
                "total_matches": len(cleaned_matches),
            },
            status_code=status.HTTP_200_OK,
        )

    except Exception as e:
        logger.error(f"Error getting matching videos: {e}")
        return error_response(
            "Failed to get matching videos",
            status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
