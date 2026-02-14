import os

import cv2
import imagehash
from loguru import logger
from PIL import Image

from lib.constants import MAX_VIDEO_DURATION, MIN_CONFIDENCE
from lib.face_analyzer import face_analyzer
from lib.faiss_manager import videos_faiss_manager
from lib.supabase_client import supabase


def compute_frame_hash(frame):
    try:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(frame_rgb)

        return imagehash.phash(pil_image)

    except Exception as e:
        logger.error(f"Error computing frame hash: {e}")
        return None


def is_duplicate_frame(current_hash, previous_hash, threshold=3):
    if previous_hash is None or current_hash is None:
        return False

    try:
        hash_diff = current_hash - previous_hash

        is_duplicate = hash_diff <= threshold

        if is_duplicate:
            logger.debug(f"Duplicate frame detected (hash diff: {hash_diff})")

        return is_duplicate

    except Exception as e:
        logger.error(f"Error checking frame duplicate: {e}")
        return False


def is_frame_worth_processing(frame, previous_frame, motion_threshold=10):
    if previous_frame is None:
        return True

    try:
        diff = cv2.absdiff(frame, previous_frame)

        motion_level = cv2.mean(diff)[0]

        is_worth = motion_level > motion_threshold

        if not is_worth:
            logger.debug(f"Skipping low motion frame (motion: {motion_level:.1f})")

        return is_worth

    except Exception as e:
        logger.error(f"Error checking frame quality: {e}")
        return True


def create_embedding_from_videos(
    video_path,
    scraped_record,
    frame_interval=30,
):
    if frame_interval <= 0:
        logger.error(
            f"Invalid frame_interval: {frame_interval}. Must be greater than 0"
        )
        return []

    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return []

    cap = None

    all_embeddings = []

    try:
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = cap.get(cv2.CAP_PROP_FPS)

        if fps > 0:
            samples_per_second = fps / frame_interval

            max_frames = int(MAX_VIDEO_DURATION * samples_per_second)
        else:
            max_frames = 360

        duration = total_frames / fps if fps > 0 else 0

        logger.info(
            f"Processing video: {video_path} | Frames: {total_frames} | FPS: {fps:.2f} | Duration: {duration:.2f}s"
        )

        frame_count = 0

        processed_frames = 0

        previous_hash = None

        previous_frame = None

        while cap.isOpened() and processed_frames < max_frames:
            ret, frame = cap.read()

            if not ret:
                break

            if frame_count % frame_interval != 0:
                frame_count += 1
                continue

            current_hash = compute_frame_hash(frame)

            if is_duplicate_frame(current_hash, previous_hash):
                frame_count += 1

                continue

            if not is_frame_worth_processing(frame, previous_frame):
                frame_count += 1

                previous_frame = frame

                previous_hash = current_hash

                continue

            timestamp = frame_count / fps if fps > 0 else 0

            face_embeddings = extract_face_embeddings_from_frame(frame)

            if face_embeddings:
                for face_data in face_embeddings:
                    embedding_record = create_video_embedding_record(
                        scraped_record=scraped_record,
                        video_path=video_path,
                        face_data=face_data,
                        frame_number=frame_count,
                        timestamp=timestamp,
                    )

                    if embedding_record:
                        all_embeddings.append(embedding_record)

                logger.debug(
                    f"Frame {frame_count}: Found {len(face_embeddings)} faces at {timestamp:.2f}s"
                )

            previous_hash = current_hash

            previous_frame = frame

            frame_count += 1

            processed_frames += 1

        logger.info(
            f"Video processing complete: {len(all_embeddings)} total face embeddings from {processed_frames} frames"
        )

    except Exception as e:
        logger.error(f"Error processing video {video_path}: {e}")

    finally:
        if cap is not None:
            cap.release()

    return all_embeddings


def extract_face_embeddings_from_frame(frame):
    try:
        faces = face_analyzer.get(frame)

        if len(faces) == 0:
            return []

        face_embeddings = []

        for face_index, face in enumerate(faces):
            if not hasattr(face, "embedding") or face.embedding is None:
                continue

            if float(face.det_score) < MIN_CONFIDENCE:
                continue

            face_embeddings.append(
                {
                    "face_id": face_index,
                    "embedding": face.embedding.tolist(),
                    "bbox": face.bbox.tolist() if hasattr(face, "bbox") else [],
                }
            )

        return face_embeddings

    except Exception as e:
        logger.error(f"Error extracting face embeddings from frame: {e}")
        return []


def create_video_embedding_record(
    scraped_record,
    video_path,
    face_data,
    frame_number,
    timestamp,
):
    try:
        return {
            "url": scraped_record["url"],
            "hostname": scraped_record["hostname"],
            "domain": scraped_record["domain"],
            "title": scraped_record.get("title"),
            "favicon": scraped_record.get("favicon"),
            "screenshot": scraped_record.get("screenshot", ""),
            "embedding": face_data["embedding"],
            "bbox": face_data["bbox"],
            "embedded_video": video_path,
            "frame_number": frame_number,
            "timestamp": timestamp,
        }

    except Exception as e:
        logger.error(f"Error creating video embedding record: {e}")
        return None


def add_video_embeddings_to_faiss(video_embeddings_data):
    if not video_embeddings_data:
        logger.warning("No video embeddings data provided")
        return False

    try:
        records_list = []

        embeddings_list = []

        for item in video_embeddings_data:
            if "embedding" in item:
                embeddings_list.append(item["embedding"])

                records_list.append(item)

        if not embeddings_list:
            logger.warning("No valid embeddings found in video data")
            return False

        if records_list:
            chunk_size = 100

            for i in range(0, len(records_list), chunk_size):
                chunk = records_list[i : i + chunk_size]

                try:
                    supabase.table("embedded_videos").insert(chunk).execute()

                    logger.info(f"Inserted {len(chunk)} video embeddings to database")

                except Exception as e:
                    logger.error(f"Error inserting chunk: {e}")

        videos_faiss_manager.rebuild_video_index_from_db()

        logger.info(
            f"Processing complete: {len(records_list)} video embeddings inserted"
        )
        return True

    except Exception as e:
        logger.error(f"Error adding video embeddings to FAISS: {e}")
        return False


def process_videos_from_folder(videos_folder_path, frame_interval=30):
    if not os.path.exists(videos_folder_path):
        logger.error(f"Videos folder not found: {videos_folder_path}")
        return False

    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"]

    video_files = []

    for file in os.listdir(videos_folder_path):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(videos_folder_path, file))

    if not video_files:
        logger.warning(f"No video files found in {videos_folder_path}")
        return False

    logger.info(f"Found {len(video_files)} videos to process")

    total_embeddings = 0

    for idx, video_path in enumerate(video_files, 1):
        logger.info(
            f"Processing video {idx}/{len(video_files)}: {os.path.basename(video_path)}"
        )

        scraped_record = {
            "url": "https://angiedevfolio.pages.dev",
            "hostname": "angiedevfolio.pages.dev",
            "domain": "pages.dev",
            "title": "Angie Portfolio",
            "favicon": "https://angiedevfolio.pages.dev/favicon.svg",
            "screenshot": "http://127.0.0.1:54321/storage/v1/object/public/pixels-screenshots/685c1e7c2fd37d6c7d5b41505e0f29104ac03ed5f8bc20099a663708eaf097bb.png",
        }

        embeddings = create_embedding_from_videos(
            video_path=video_path,
            scraped_record=scraped_record,
            frame_interval=frame_interval,
        )

        if embeddings:
            success = add_video_embeddings_to_faiss(embeddings)

            if success:
                total_embeddings += len(embeddings)
                logger.info(
                    f"✓ Added {len(embeddings)} embeddings from {os.path.basename(video_path)}"
                )
            else:
                logger.error(
                    f"✗ Failed to add embeddings from {os.path.basename(video_path)}"
                )
        else:
            logger.warning(f"⊘ No faces found in {os.path.basename(video_path)}")

    logger.info(f"Processing complete! Total embeddings: {total_embeddings}")
    return True


if __name__ == "__main__":
    import sys

    videos_folder = sys.argv[1] if len(sys.argv) > 1 else "videos"

    frame_interval = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    logger.info("Starting video embedding process...")

    logger.info(f"Videos folder: {videos_folder}")

    logger.info(f"Frame interval: {frame_interval}")

    process_videos_from_folder(videos_folder, frame_interval)
