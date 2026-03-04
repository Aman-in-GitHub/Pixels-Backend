import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from loguru import logger

from lib.constants import (
    APP_LOG_LEVEL,
    CORS_ALLOW_ALL,
    ROOT_HELLO_MESSAGE,
    STATIC_VIDEOS_DIR,
    STATIC_VIDEOS_ROUTE,
)
from lib.db import close_db_pool, init_db_pool
from lib.faiss_manager import images_faiss_manager, videos_faiss_manager
from lib.utils import success_response
from routes.get_matching_images import router as get_matching_images_router
from routes.get_matching_videos import router as get_matching_videos_router
from routes.get_whois import router as get_whois_router
from routes.recheck_images import router as recheck_images_router
from routes.verify_photo import router as verify_photo_router

logger.remove()

logger.add(sys.stdout, level=APP_LOG_LEVEL)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await init_db_pool()

        images_stats = images_faiss_manager.get_stats()
        logger.info(f"Images FAISS index loaded: {images_stats}")

        if images_stats["total_embeddings"] == 0:
            await images_faiss_manager.rebuild_images_index_from_db()

        videos_stats = videos_faiss_manager.get_stats()
        logger.info(f"Videos FAISS index loaded: {videos_stats}")

        if videos_stats["total_embeddings"] == 0:
            await videos_faiss_manager.rebuild_video_index_from_db()

    except Exception as e:
        logger.error(f"Error during startup: {e}")

    yield

    await close_db_pool()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ALL,
    allow_credentials=True,
    allow_methods=CORS_ALLOW_ALL,
    allow_headers=CORS_ALLOW_ALL,
)

app.include_router(get_matching_images_router)
app.include_router(get_matching_videos_router)
app.include_router(get_whois_router)
app.include_router(verify_photo_router)
app.include_router(recheck_images_router)


app.mount(STATIC_VIDEOS_ROUTE, StaticFiles(directory=STATIC_VIDEOS_DIR), name="videos")


@app.get("/")
async def root():
    return success_response(ROOT_HELLO_MESSAGE, data={"service": "pixels-backend"})


@app.get("/favicon.ico")
async def favicon():
    svg_content = """
    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">
        <text x="50%" y="50%" font-size="24" text-anchor="middle" dy=".35em">💫</text>
    </svg>
    """

    return Response(content=svg_content, media_type="image/svg+xml")
