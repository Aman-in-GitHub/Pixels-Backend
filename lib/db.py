import asyncio
import json
import os
from typing import Any, Iterable
from uuid import UUID

import asyncpg
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

_pool: asyncpg.Pool | None = None
_pool_lock = asyncio.Lock()

_ALLOWED_EMBED_TABLES = {"embedded_images", "embedded_videos"}
_udt_cache: dict[tuple[str, str], str | None] = {}


def _json_default(value: Any) -> Any:
    if isinstance(value, UUID):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=_json_default)


def _get_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL is required")
    return database_url


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning(f"Invalid {name} value {value!r}; using default {default}")
        return default


async def _init_connection(connection: asyncpg.Connection) -> None:
    await connection.set_type_codec(
        "json",
        schema="pg_catalog",
        encoder=_json_dumps,
        decoder=json.loads,
        format="text",
    )
    await connection.set_type_codec(
        "jsonb",
        schema="pg_catalog",
        encoder=_json_dumps,
        decoder=json.loads,
        format="text",
    )


def _decode_json_if_string(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    trimmed = value.strip()
    if not trimmed or trimmed[0] not in "[{":
        return value
    try:
        return json.loads(trimmed)
    except json.JSONDecodeError:
        return value


def normalize_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("images", "embedding", "bbox"):
        if key in record:
            record[key] = _decode_json_if_string(record[key])
    return record


def _rows_to_dicts(rows: list[asyncpg.Record]) -> list[dict[str, Any]]:
    return [normalize_record(dict(row)) for row in rows]


async def init_db_pool() -> asyncpg.Pool:
    global _pool

    if _pool is not None:
        return _pool

    async with _pool_lock:
        if _pool is not None:
            return _pool

        _pool = await asyncpg.create_pool(
            dsn=_get_database_url(),
            min_size=_env_int("DB_POOL_MIN_SIZE", 1),
            max_size=_env_int("DB_POOL_MAX_SIZE", 10),
            command_timeout=_env_int("DB_COMMAND_TIMEOUT", 30),
            max_inactive_connection_lifetime=_env_int(
                "DB_MAX_INACTIVE_CONNECTION_LIFETIME", 300
            ),
            init=_init_connection,
        )
        logger.info("Initialized asyncpg connection pool")
        return _pool


async def get_db_pool() -> asyncpg.Pool:
    if _pool is None:
        return await init_db_pool()
    return _pool


async def close_db_pool() -> None:
    global _pool
    if _pool is None:
        return
    await _pool.close()
    _pool = None
    _udt_cache.clear()
    logger.info("Closed asyncpg connection pool")


async def _get_column_udt(table_name: str, column_name: str) -> str | None:
    cache_key = (table_name, column_name)
    if cache_key in _udt_cache:
        return _udt_cache[cache_key]

    pool = await get_db_pool()
    async with pool.acquire() as connection:
        udt_name = await connection.fetchval(
            """
            SELECT udt_name
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = $1
              AND column_name = $2
            LIMIT 1
            """,
            table_name,
            column_name,
        )

    _udt_cache[cache_key] = udt_name
    return udt_name


def _embedding_to_vector_literal(embedding: Any) -> str | None:
    if embedding is None:
        return None
    if isinstance(embedding, str):
        return embedding

    try:
        return "[" + ",".join(str(float(value)) for value in embedding) + "]"
    except Exception:
        return str(embedding)


async def fetch_unprocessed_scraped_images(limit: int) -> list[dict[str, Any]]:
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            """
            SELECT *
            FROM scraped_images
            WHERE is_processed = FALSE
            ORDER BY id
            LIMIT $1
            """,
            limit,
        )
    return _rows_to_dicts(rows)


async def insert_embedded_images(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0

    embedding_udt = await _get_column_udt("embedded_images", "embedding")
    embedding_is_vector = embedding_udt == "vector"

    rows = [
        (
            record.get("scraped_images_id"),
            record.get("url"),
            record.get("hostname"),
            record.get("domain"),
            record.get("title"),
            record.get("favicon"),
            record.get("images", []),
            _embedding_to_vector_literal(record.get("embedding"))
            if embedding_is_vector
            else record.get("embedding"),
            record.get("bbox", []),
            record.get("embedded_image"),
        )
        for record in records
    ]

    query = """
        INSERT INTO embedded_images (
            scraped_images_id,
            url,
            hostname,
            domain,
            title,
            favicon,
            images,
            embedding,
            bbox,
            embedded_image
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    """
    if embedding_is_vector:
        query = """
            INSERT INTO embedded_images (
                scraped_images_id,
                url,
                hostname,
                domain,
                title,
                favicon,
                images,
                embedding,
                bbox,
                embedded_image
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector, $9, $10)
        """

    pool = await get_db_pool()
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.executemany(query, rows)
    return len(rows)


async def insert_embedded_videos(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0

    embedding_udt = await _get_column_udt("embedded_videos", "embedding")
    embedding_is_vector = embedding_udt == "vector"

    rows = [
        (
            record.get("url"),
            record.get("hostname"),
            record.get("domain"),
            record.get("title"),
            record.get("favicon"),
            _embedding_to_vector_literal(record.get("embedding"))
            if embedding_is_vector
            else record.get("embedding"),
            record.get("bbox", []),
            record.get("embedded_video"),
            record.get("frame_number"),
            record.get("timestamp"),
        )
        for record in records
    ]

    query = """
        INSERT INTO embedded_videos (
            url,
            hostname,
            domain,
            title,
            favicon,
            embedding,
            bbox,
            embedded_video,
            frame_number,
            timestamp
        )
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
    """
    if embedding_is_vector:
        query = """
            INSERT INTO embedded_videos (
                url,
                hostname,
                domain,
                title,
                favicon,
                embedding,
                bbox,
                embedded_video,
                frame_number,
                timestamp
            )
            VALUES ($1, $2, $3, $4, $5, $6::vector, $7, $8, $9, $10)
        """

    pool = await get_db_pool()
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.executemany(query, rows)
    return len(rows)


async def mark_scraped_images_processed(ids: Iterable[Any]) -> int:
    rows = [(record_id,) for record_id in ids]
    if not rows:
        return 0

    pool = await get_db_pool()
    async with pool.acquire() as connection:
        async with connection.transaction():
            await connection.executemany(
                """
                UPDATE scraped_images
                SET is_processed = TRUE
                WHERE id = $1
                """,
                rows,
            )
    return len(rows)


async def fetch_embedded_image_ids() -> set[Any]:
    pool = await get_db_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch("SELECT id FROM embedded_images")
    return {row["id"] for row in rows}


async def fetch_embedding_rows(
    table_name: str, limit: int, offset: int
) -> list[dict[str, Any]]:
    if table_name not in _ALLOWED_EMBED_TABLES:
        raise ValueError(f"Unsupported table: {table_name}")

    pool = await get_db_pool()
    async with pool.acquire() as connection:
        rows = await connection.fetch(
            f"SELECT * FROM {table_name} LIMIT $1 OFFSET $2",
            limit,
            offset,
        )
    return _rows_to_dicts(rows)
