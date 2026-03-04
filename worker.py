import asyncio
import sys

from loguru import logger

from lib.constants import APP_LOG_LEVEL
from lib.cron import process_scraped_images_for_embeddings
from lib.db import close_db_pool, init_db_pool

logger.remove()
logger.add(sys.stdout, level=APP_LOG_LEVEL)


async def run_worker() -> None:
    logger.info("Starting embeddings worker...")
    await init_db_pool()

    try:
        await process_scraped_images_for_embeddings()
    finally:
        await close_db_pool()
        logger.info("Embeddings worker stopped")


def main() -> None:
    try:
        asyncio.run(run_worker())
    except KeyboardInterrupt:
        logger.warning("Embeddings worker interrupted")


if __name__ == "__main__":
    main()
