from fastapi import APIRouter, File, UploadFile
from loguru import logger

from lib.constants import (
    MATCHING_THRESHOLD,
    MAX_MATCHING_RESULTS,
    VIDEO_MATCH_EXCLUDED_FIELDS,
)
from lib.face_analyzer import face_analyzer
from lib.faiss_manager import videos_faiss_manager
from lib.utils import (
    extract_single_face_from_image_data,
    is_allowed_image_upload,
    is_valid_file_size,
    remove_keys,
)

router = APIRouter()


@router.post("/get-matching-videos")
async def get_matching_videos(file: UploadFile = File(...)):
    if not is_allowed_image_upload(file):
        return {"success": False, "message": "Invalid file"}

    try:
        image_data = await file.read()

        if not is_valid_file_size(image_data):
            return {"success": False, "message": "Invalid file size"}

        face, _, error_msg = extract_single_face_from_image_data(
            image_data, face_analyzer
        )
        embedding = face.embedding.tolist() if face else None

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

        return {
            "success": True,
            "matching_videos": cleaned_matches,
            "total_matches": len(cleaned_matches),
            "message": f"Found {len(cleaned_matches)} matching videos",
        }

    except Exception as e:
        logger.error(f"Error getting matching videos: {e}")
        return {"success": False, "message": "Failed to get matching videos"}
