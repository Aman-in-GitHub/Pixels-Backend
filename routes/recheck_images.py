import asyncio
from collections import deque
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from fastapi import APIRouter
from loguru import logger
from pydantic import BaseModel

from lib.constants import (
    DEFAULT_HTTP_TIMEOUT_SECONDS,
    MATCHING_THRESHOLD,
    RECHECK_DEFAULT_MAX_DEPTH,
    RECHECK_DEFAULT_MAX_PAGES,
    REMOTE_ALLOWED_IMAGE_TYPES,
    REMOTE_MIN_IMAGE_MAGIC_BYTES,
    SCRAPE_SKIP_IMAGE_EXTENSIONS,
)
from lib.cron import (
    executor,
    extract_face_embeddings,
    get_http_session,
    semaphore,
)
from lib.utils import (
    compute_cosine_similarity,
    content_type_matches_any,
    is_valid_image_magic_bytes,
)

router = APIRouter()


class RecheckImageRequest(BaseModel):
    id: str
    image_url: str
    website_url: str
    max_pages: int = RECHECK_DEFAULT_MAX_PAGES
    max_depth: int = RECHECK_DEFAULT_MAX_DEPTH


def extract_single_face_embedding(image_data):
    try:
        face_embeddings = extract_face_embeddings(image_data)

        if not face_embeddings or len(face_embeddings) == 0:
            return None, "No faces detected"

        return face_embeddings[0]["embedding"], None

    except Exception as e:
        logger.error(f"Error extracting face embedding: {e}")
        return None, str(e)


def compare_embeddings(embedding1, embedding2, threshold=MATCHING_THRESHOLD):
    try:
        similarity = compute_cosine_similarity(embedding1, embedding2)

        return similarity >= threshold, similarity

    except Exception as e:
        logger.error(f"Error comparing embeddings: {e}")
        return False, 0.0


async def download_image(url, session):
    async with semaphore:
        try:
            async with session.get(
                url, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS
            ) as response:
                if response.status != 200:
                    return None

                content_type = response.headers.get("Content-Type", "")
                if not content_type_matches_any(
                    content_type, REMOTE_ALLOWED_IMAGE_TYPES
                ):
                    logger.warning(
                        f"Skipping {url}: invalid content type {content_type}"
                    )
                    return None

                image_data = await response.read()

                if len(image_data) < REMOTE_MIN_IMAGE_MAGIC_BYTES:
                    logger.warning(
                        f"Skipping {url}: file too small ({len(image_data)} bytes)"
                    )
                    return None

                if not is_valid_image_magic_bytes(image_data):
                    logger.warning(
                        f"Skipping {url}: not a valid image file (magic bytes check failed)"
                    )
                    return None

                return image_data

        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return None


def normalize_url(url, base_url):
    if not url:
        return None

    if url.startswith("/"):
        parsed = urlparse(base_url)
        url = f"{parsed.scheme}://{parsed.netloc}{url}"
    elif not url.startswith("http"):
        url = urljoin(base_url, url)

    url = url.split("#")[0]

    return url


def is_same_domain(url, base_url):
    parsed_url = urlparse(url)
    parsed_base = urlparse(base_url)
    return parsed_url.netloc == parsed_base.netloc


async def crawl_website_pages(
    base_url,
    session,
    max_pages=RECHECK_DEFAULT_MAX_PAGES,
    max_depth=RECHECK_DEFAULT_MAX_DEPTH,
):
    visited_urls = set()
    pages_to_visit = deque([(base_url, 0)])
    discovered_pages = []

    logger.info(
        f"Starting website crawl from {base_url} (max_pages={max_pages}, max_depth={max_depth})"
    )

    while pages_to_visit and len(discovered_pages) < max_pages:
        current_url, depth = pages_to_visit.popleft()

        if current_url in visited_urls or depth > max_depth:
            continue

        visited_urls.add(current_url)
        discovered_pages.append(current_url)

        try:
            async with session.get(
                current_url, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS
            ) as response:
                if response.status != 200:
                    continue

                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type:
                    continue

                html_content = await response.text()

            soup = BeautifulSoup(html_content, "html.parser")

            if depth < max_depth:
                for link in soup.find_all("a", href=True):
                    href = link.get("href")
                    normalized_url = normalize_url(href, current_url)

                    if not normalized_url:
                        continue

                    if (
                        is_same_domain(normalized_url, base_url)
                        and normalized_url not in visited_urls
                    ):
                        pages_to_visit.append((normalized_url, depth + 1))

            logger.info(
                f"Crawled {current_url} (depth {depth}), found {len(pages_to_visit)} new URLs to visit"
            )

        except Exception as e:
            logger.warning(f"Error crawling {current_url}: {e}")
            continue

    logger.info(f"Crawl complete: discovered {len(discovered_pages)} pages")
    return discovered_pages


async def scrape_page_images(page_url, session):
    """Extract all image URLs from a single page."""
    try:
        async with session.get(
            page_url, timeout=DEFAULT_HTTP_TIMEOUT_SECONDS
        ) as response:
            if response.status != 200:
                logger.warning(f"Failed to fetch page: {response.status}")
                return []

            html_content = await response.text()

        soup = BeautifulSoup(html_content, "html.parser")

        image_urls = []

        for img in soup.find_all("img"):
            src = img.get("src") or img.get("data-src")
            if not src:
                continue

            normalized_url = normalize_url(src, page_url)
            if not normalized_url:
                continue

            url_lower = normalized_url.lower()
            if any(url_lower.endswith(ext) for ext in SCRAPE_SKIP_IMAGE_EXTENSIONS):
                continue

            if normalized_url.startswith("data:"):
                continue

            image_urls.append(normalized_url)

        return image_urls

    except Exception as e:
        logger.warning(f"Error scraping images from {page_url}: {e}")
        return []


async def scrape_all_website_images(page_urls, session):
    all_image_urls = set()

    logger.info(f"Scraping images from {len(page_urls)} pages")

    scrape_tasks = [scrape_page_images(url, session) for url in page_urls]
    results = await asyncio.gather(*scrape_tasks, return_exceptions=True)

    for page_url, result in zip(page_urls, results):
        if isinstance(result, list):
            all_image_urls.update(result)
            logger.info(f"Found {len(result)} images on {page_url}")
        elif isinstance(result, Exception):
            logger.warning(f"Failed to scrape {page_url}: {result}")

    logger.info(f"Total unique images found across all pages: {len(all_image_urls)}")
    return list(all_image_urls)


async def check_image_on_website(target_embedding, image_url, session):
    try:
        image_data = await download_image(image_url, session)

        if not image_data:
            logger.debug(f"Skipped {image_url}: failed to download or invalid image")
            return None

        loop = asyncio.get_running_loop()

        try:
            face_embeddings = await loop.run_in_executor(
                executor, extract_face_embeddings, image_data
            )
        except Exception as e:
            logger.debug(f"Failed to extract faces from {image_url}: {e}")
            return None

        if not face_embeddings or len(face_embeddings) == 0:
            logger.debug(f"No faces found in {image_url}")
            return None

        for face_data in face_embeddings:
            embedding = face_data["embedding"]
            is_match, similarity = compare_embeddings(target_embedding, embedding)

            if is_match:
                logger.info(
                    f"MATCH FOUND in {image_url} with similarity {similarity:.4f}"
                )
                return {
                    "image_url": image_url,
                    "similarity": similarity,
                }

        return None

    except Exception as e:
        logger.warning(f"Error checking image {image_url}: {e}")
        return None


@router.post("/recheck-image")
async def recheck_image(request: RecheckImageRequest):
    try:
        logger.info(f"Rechecking image {request.id} from {request.website_url}")

        session = await get_http_session()

        logger.info(f"Downloading original image: {request.image_url}")

        original_image_data = await download_image(request.image_url, session)

        if not original_image_data:
            return {
                "success": False,
                "message": "Failed to download original image",
            }

        loop = asyncio.get_running_loop()

        target_embedding, error = await loop.run_in_executor(
            executor, extract_single_face_embedding, original_image_data
        )

        if error or not target_embedding:
            return {
                "success": False,
                "message": f"Failed to extract face from original image: {error}",
            }

        logger.info("Successfully extracted face embedding from original image")

        logger.info(f"Crawling website to discover all pages: {request.website_url}")
        discovered_pages = await crawl_website_pages(
            request.website_url,
            session,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
        )

        logger.info(f"Scraping images from {len(discovered_pages)} discovered pages")
        website_image_urls = await scrape_all_website_images(discovered_pages, session)

        if not website_image_urls:
            return {
                "success": True,
                "still_exists": False,
                "message": "No images found on website",
                "pages_crawled": len(discovered_pages),
                "total_images_checked": 0,
                "matches_found": [],
            }

        logger.info(
            f"Found {len(website_image_urls)} unique images to check across {len(discovered_pages)} pages"
        )

        check_tasks = [
            check_image_on_website(target_embedding, url, session)
            for url in website_image_urls
        ]

        results = await asyncio.gather(*check_tasks, return_exceptions=True)

        matches = [
            result
            for result in results
            if isinstance(result, dict) and result is not None
        ]

        errors = [result for result in results if isinstance(result, Exception)]
        none_results = [result for result in results if result is None]

        matches.sort(key=lambda x: x["similarity"], reverse=True)

        still_exists = len(matches) > 0

        logger.info(
            f"Recheck complete: {len(matches)} matches found out of {len(website_image_urls)} images "
            f"across {len(discovered_pages)} pages (errors: {len(errors)}, no faces/invalid: {len(none_results)})"
        )

        return {
            "success": True,
            "matches_found": matches,
            "message": (
                f"Found {len(matches)} matching image(s) across {len(discovered_pages)} pages"
                if still_exists
                else "No matching images found - image has been removed"
            ),
        }

    except Exception as e:
        logger.error(f"Error in recheck_image: {e}")
        return {
            "success": False,
            "message": f"Failed to recheck image: {str(e)}",
        }
