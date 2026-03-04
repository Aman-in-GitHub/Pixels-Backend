ROWS_LIMIT = 50
BATCH_SIZE = 50
CRON_INTERVAL = 10
CRON_BATCH_DELAY_SECONDS = 3
APP_LOG_LEVEL = "INFO"

MIN_CONFIDENCE = 0.3
MATCHING_THRESHOLD = 0.3
MASK_THRESHOLD = 0.5
WINK_THRESHOLD = 0.5
FACE_QUALITY_THRESHOLD = 0.5

MAX_MATCHING_RESULTS = 100
MAX_VIDEO_DURATION = 180

MIN_FILE_SIZE = 1 * 1024
MAX_FILE_SIZE = 100 * 1024 * 1024

DEFAULT_ASYNC_SEMAPHORE_LIMIT = 6
DEFAULT_THREAD_POOL_WORKERS = 6

DEFAULT_HTTP_TIMEOUT_SECONDS = 10
DEFAULT_HTTP_MAX_LINE_SIZE = 32768
DEFAULT_HTTP_MAX_FIELD_SIZE = 32768
DEFAULT_HTTP_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
    "image/tiff",
    "image/svg+xml",
    "image/heic",
    "image/heif",
    "image/avif",
}

REMOTE_ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/gif",
}
REMOTE_MIN_IMAGE_MAGIC_BYTES = 12

IMAGE_MATCH_EXCLUDED_FIELDS = {
    "embedding",
    "bbox",
    "confidence",
    "images",
    "embedded_image",
}
VIDEO_MATCH_EXCLUDED_FIELDS = {"embedding", "bbox", "confidence", "timestamp"}

SCRAPE_SKIP_IMAGE_EXTENSIONS = {
    ".svg",
    ".ico",
    ".webm",
    ".mp4",
    ".pdf",
    ".css",
    ".js",
}

RECHECK_DEFAULT_MAX_PAGES = 50
RECHECK_DEFAULT_MAX_DEPTH = 3

DB_WRITE_CHUNK_SIZE = 25
VIDEO_DB_WRITE_CHUNK_SIZE = 100

FAISS_EMBEDDING_DIMENSION = 512
FAISS_DEFAULT_REBUILD_BATCH_SIZE = 100000

DEFAULT_VIDEO_FRAME_INTERVAL = 30
DEFAULT_VIDEO_SAMPLES_FALLBACK = 360
DUPLICATE_FRAME_HASH_THRESHOLD = 3
FRAME_MOTION_THRESHOLD = 10
VIDEO_FILE_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm")

CORS_ALLOW_ALL = ["*"]
STATIC_VIDEOS_ROUTE = "/videos"
STATIC_VIDEOS_DIR = "videos"
ROOT_HELLO_MESSAGE = "Hello World!"
