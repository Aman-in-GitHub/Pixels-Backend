import asyncio
import atexit
import concurrent.futures
import io
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


async def download_and_process_image(image_url, session):
    async with semaphore:
        try:
            async with session.get(image_url) as response:
                if response.status != 200:
                    return None

                image_data = await response.read()

                if not (MIN_FILE_SIZE <= len(image_data) <= MAX_FILE_SIZE):
                    return None

            loop = asyncio.get_running_loop()

            face_embeddings = await loop.run_in_executor(
                executor, extract_face_embeddings, image_data
            )

            logger.info(f"Found {len(face_embeddings)} faces in {image_url}")

            if face_embeddings:
                return face_embeddings

            return None

        except Exception:
            logger.warning(f"Failed to process {image_url}")
            return None


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
            "screenshot": scraped_record.get("screenshot", ""),
            "embedding": face_data["embedding"],
            "bbox": face_data["bbox"],
            "embedded_image": image_url,
        }

    except Exception as e:
        logger.error(f"Error creating embedding record: {e}")
        return None


async def process_single_record(record, session):
    try:
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

        if not image_urls:
            return []

        image_results = await asyncio.gather(
            *[download_and_process_image(url, session) for url in image_urls],
            return_exceptions=True,
        )

        all_embeddings = []

        for url, result in zip(image_urls, image_results):
            if isinstance(result, Exception):
                logger.warning(f"Error processing {url}: {result}")
                continue

            if result:
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

        session = await get_http_session()

        batch_size = 25

        total_processed = 0

        all_embeddings = []

        processed_record_ids = []

        for i in range(0, len(scraped_records), batch_size):
            batch_records = scraped_records[i : i + batch_size]

            batch_results = await asyncio.gather(
                *[process_single_record(record, session) for record in batch_records],
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
