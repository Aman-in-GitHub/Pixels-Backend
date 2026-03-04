import asyncio
import atexit
import concurrent.futures
import io
import json
import os
import pathlib
import shutil
import tempfile
import time

import aiohttp
import cv2
import numpy as np
from loguru import logger
from PIL import Image

from lib.constants import (
    CRON_INTERVAL,
    MAX_FILE_SIZE,
    MIN_CONFIDENCE,
    MIN_FILE_SIZE,
    ROWS_LIMIT,
)
from lib.face_analyzer import face_analyzer
from lib.faiss_manager import images_faiss_manager
from lib.supabase_client import supabase

http_session = None

semaphore = asyncio.Semaphore(6)

executor = concurrent.futures.ThreadPoolExecutor(max_workers=6)

IMG2DATASET_PROCESS_COUNT = max(1, min(16, os.cpu_count() or 4))

IMG2DATASET_THREAD_COUNT = 64


def cleanup_resources():
    executor.shutdown(wait=True)


atexit.register(cleanup_resources)


async def get_http_session():
    global http_session

    if http_session is None or http_session.closed:
        connector = aiohttp.TCPConnector(
            limit_per_host=100,
            limit=600,
        )

        http_session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            },
            connector=connector,
            max_line_size=32768,
            max_field_size=32768,
        )

    return http_session


def extract_face_embeddings(image_data):
    try:
        pil_image = Image.open(io.BytesIO(image_data))

        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        faces = face_analyzer.get(opencv_image)

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
        logger.error(f"Error extracting face embeddings: {e}")
        return []


def extract_image_urls(record):
    images = record.get("images", [])

    if not images:
        return []

    image_urls = []

    for image_data in images:
        image_url = image_data.get("src")

        if image_url:
            if image_url.startswith("/"):
                image_url = f"https://{record['hostname']}{image_url}"
            elif not image_url.startswith("http"):
                image_url = f"https://{record['hostname']}/{image_url}"
            image_urls.append(image_url)

    return image_urls


def resolve_downloaded_image_path(metadata_path, key):
    parent_dir = metadata_path.parent
    image_candidates = []

    if key:
        image_candidates.extend(parent_dir.glob(f"{key}.*"))

    image_candidates.extend(parent_dir.glob(f"{metadata_path.stem}.*"))

    for candidate in image_candidates:
        if candidate.suffix.lower() not in {".json", ".txt"} and candidate.is_file():
            return candidate

    return None


def collect_downloaded_images(output_folder):
    downloaded_images = {}

    for metadata_path in pathlib.Path(output_folder).rglob("*.json"):
        if metadata_path.name.endswith("_stats.json"):
            continue

        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning(f"Failed to read img2dataset metadata {metadata_path}: {e}")
            continue

        if metadata.get("status") != "success":
            continue

        source_url = metadata.get("url")

        if not source_url or source_url in downloaded_images:
            continue

        image_path = resolve_downloaded_image_path(metadata_path, metadata.get("key"))

        if not image_path:
            continue

        try:
            image_data = image_path.read_bytes()
        except Exception as e:
            logger.warning(f"Failed to read downloaded image {image_path}: {e}")
            continue

        if MIN_FILE_SIZE <= len(image_data) <= MAX_FILE_SIZE:
            downloaded_images[source_url] = image_data

    return downloaded_images


async def download_images_with_img2dataset(image_urls):
    if not image_urls:
        return {}

    tmp_dir = tempfile.mkdtemp(prefix="img2dataset_cron_")

    try:
        url_list_path = pathlib.Path(tmp_dir) / "urls.txt"
        output_folder = pathlib.Path(tmp_dir) / "output"

        output_folder.mkdir(parents=True, exist_ok=True)
        url_list_path.write_text("\n".join(image_urls), encoding="utf-8")

        process = await asyncio.create_subprocess_exec(
            "img2dataset",
            f"--url_list={url_list_path}",
            "--input_format=txt",
            f"--output_folder={output_folder}",
            "--output_format=files",
            f"--processes_count={IMG2DATASET_PROCESS_COUNT}",
            f"--thread_count={IMG2DATASET_THREAD_COUNT}",
            "--number_sample_per_shard=10000",
            "--resize_mode=no",
            "--enable_wandb=False",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr_data = await process.communicate()

        if process.returncode != 0:
            stderr_text = stderr_data.decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"img2dataset failed with code {process.returncode}: {stderr_text}"
            )

        downloaded_images = collect_downloaded_images(output_folder)

        logger.info(
            f"img2dataset downloaded {len(downloaded_images)}/{len(image_urls)} images"
        )

        return downloaded_images

    except FileNotFoundError:
        raise RuntimeError(
            "img2dataset binary not found. Install img2dataset to use cron downloader."
        )
    except Exception as e:
        raise RuntimeError(f"img2dataset download failed: {e}") from e
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def create_embedding_record(scraped_record, image_url, face_data):
    try:
        return {
            "scraped_images_id": scraped_record["id"],
            "url": scraped_record["url"],
            "hostname": scraped_record["hostname"],
            "domain": scraped_record["domain"],
            "title": scraped_record.get("title"),
            "favicon": scraped_record.get("favicon"),
            "images": scraped_record.get("images", []),
            "embedding": face_data["embedding"],
            "bbox": face_data["bbox"],
            "embedded_image": image_url,
        }

    except Exception as e:
        logger.error(f"Error creating embedding record: {e}")
        return None


async def process_single_record(record, downloaded_images):
    try:
        image_urls = extract_image_urls(record)

        if not image_urls:
            return []

        loop = asyncio.get_running_loop()

        candidate_urls = []
        extraction_tasks = []

        for url in image_urls:
            image_data = downloaded_images.get(url)

            if not image_data:
                continue

            candidate_urls.append(url)
            extraction_tasks.append(
                loop.run_in_executor(executor, extract_face_embeddings, image_data)
            )

        if not extraction_tasks:
            return []

        image_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)

        all_embeddings = []

        for url, result in zip(candidate_urls, image_results):
            if isinstance(result, Exception):
                logger.warning(f"Error processing {url}: {result}")
                continue

            if result:
                logger.info(f"Found {len(result)} faces in {url}")

                for face_data in result:
                    embedding_record = create_embedding_record(record, url, face_data)

                    if embedding_record:
                        all_embeddings.append(embedding_record)

        return all_embeddings

    except Exception as e:
        logger.error(f"Error processing record {record.get('id')}: {e}")
        return []


async def process_scraped_images():
    try:
        supabase_response = (
            supabase.table("scraped_images")
            .select("*")
            .eq("is_processed", False)
            .limit(ROWS_LIMIT)
            .execute()
        )

        scraped_records = supabase_response.data

        if not scraped_records:
            logger.info("No unprocessed images found")
            return 0

        logger.info(f"Processing {len(scraped_records)} records...")

        batch_size = 25

        total_processed = 0

        all_embeddings = []

        processed_record_ids = []

        for i in range(0, len(scraped_records), batch_size):
            batch_records = scraped_records[i : i + batch_size]

            batch_urls = []
            for record in batch_records:
                batch_urls.extend(extract_image_urls(record))

            unique_batch_urls = list(dict.fromkeys(batch_urls))

            downloaded_images = await download_images_with_img2dataset(
                unique_batch_urls
            )

            batch_results = await asyncio.gather(
                *[
                    process_single_record(record, downloaded_images)
                    for record in batch_records
                ],
                return_exceptions=True,
            )

            for record, result in zip(batch_records, batch_results):
                total_processed += 1

                if isinstance(result, Exception):
                    logger.error(f"Error processing record {record['id']}: {result}")
                    continue

                processed_record_ids.append(record["id"])

                if result:
                    all_embeddings.extend(result)

                    logger.info(
                        f"Processed record {record['id']} | {len(result)} embeddings | ({total_processed}/{len(scraped_records)})"
                    )
                else:
                    logger.info(
                        f"Processed record {record['id']} | 0 embeddings | ({total_processed}/{len(scraped_records)})"
                    )

        if all_embeddings:
            chunk_size = 25

            for i in range(0, len(all_embeddings), chunk_size):
                chunk = all_embeddings[i : i + chunk_size]

                try:
                    supabase.table("embedded_images").insert(chunk).execute()

                    logger.info(f"Inserted {len(chunk)} embeddings to database")

                except Exception as e:
                    logger.error(f"Error inserting chunk: {e}")

            images_faiss_manager.rebuild_images_index_from_db()

        if processed_record_ids:
            chunk_size = 25

            for i in range(0, len(processed_record_ids), chunk_size):
                id_chunk = processed_record_ids[i : i + chunk_size]

                try:
                    supabase.table("scraped_images").update({"is_processed": True}).in_(
                        "id", id_chunk
                    ).execute()

                    logger.info(f"Marked {len(id_chunk)} records as processed")

                except Exception as e:
                    logger.error(f"Error updating chunk of IDs: {e}")

            logger.info(
                f"Total marked as processed: {len(processed_record_ids)} records"
            )

        logger.info(
            f"Processing complete: {len(all_embeddings)} embeddings from {len(processed_record_ids)} records"
        )

        return len(scraped_records)

    except Exception as e:
        logger.error(f"Error in process_scraped_images: {e}")
        return 0


processing_lock = asyncio.Lock()


async def process_scraped_images_for_embeddings():
    while True:
        try:
            if processing_lock.locked():
                logger.warning(
                    "Previous cron job still running, skipping this iteration"
                )

                await asyncio.sleep(CRON_INTERVAL)

                continue

            async with processing_lock:
                logger.info("Starting cron cycle...")

                cycle_start_time = time.perf_counter()

                total_batches = 0

                while True:
                    logger.info(f"Starting batch {total_batches + 1}...")

                    start_time = time.perf_counter()

                    records_processed = await process_scraped_images()

                    end_time = time.perf_counter()

                    duration = end_time - start_time

                    logger.info(
                        f"Batch {total_batches + 1} completed in {duration:.2f} seconds"
                    )

                    total_batches += 1

                    if records_processed == 0:
                        logger.info("No more records to process, ending cycle")
                        break

                    logger.info("Waiting 5 seconds before next batch...")

                    await asyncio.sleep(5)

                cycle_end_time = time.perf_counter()

                cycle_duration = cycle_end_time - cycle_start_time

                logger.info(
                    f"Cycle complete: {total_batches} batches in {cycle_duration:.2f} seconds"
                )

            logger.info(f"Waiting {CRON_INTERVAL} seconds before next cycle...")
            await asyncio.sleep(CRON_INTERVAL)

        except asyncio.CancelledError:
            logger.warning("Background task cancelled")
            break

        except Exception as e:
            logger.error(f"Error in background task: {e}")
            await asyncio.sleep(CRON_INTERVAL)
