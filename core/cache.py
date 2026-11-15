"""校正预处理缓存。

缓存在反相前按照片内容命中，不包含模型、思考强度等模型参数。缓存内容保存在
配置文件所在目录的 cache/preprocess/ 下，用一个 SQLite 数据库代替
“每张照片一个 JPEG + JSON 索引”：
  - 单文件、无需每次重写索引，原子写入且多进程安全；
  - 记录最近使用时间，超过容量上限时自动淘汰最久未用的条目；
  - 缓存是可重建数据，默认上限 1 GB（NCC_CACHE_MAX_MB 可调）。
"""

import hashlib
import os
import sqlite3
import time
from contextlib import closing
from pathlib import Path
from typing import Optional

import numpy as np


CACHE_VERSION = "preprocess-v3"
DB_FILENAME = f"{CACHE_VERSION}.db"
DEFAULT_MAX_CACHE_MB = 1024
LEGACY_MANIFEST = "index.json"
LEGACY_JPEG_GLOB = "preprocess-v*.jpg"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS preprocess (
    photo_key    TEXT PRIMARY KEY,
    data         BLOB NOT NULL,
    bytes        INTEGER NOT NULL,
    created_at   INTEGER NOT NULL,   -- time.time_ns()
    last_used_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_preprocess_last_used
    ON preprocess(last_used_at);
"""


def cache_enabled() -> bool:
    """允许通过环境变量临时关闭缓存（测试/排障用）。"""
    value = os.environ.get("NCC_DISABLE_CACHE", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def cache_root() -> Path:
    """返回缓存目录；默认跟随配置文件所在目录。"""
    override = os.environ.get("NCC_CACHE_DIR", "").strip()
    if override:
        return Path(override).expanduser()

    config_path = os.environ.get("NCC_CONFIG_PATH", "").strip()
    if config_path:
        config_dir = Path(config_path).expanduser().parent
    else:
        config_dir = Path.home() / ".negative-corrector"
    return config_dir / "cache" / "preprocess"


def build_photo_cache_key(image: np.ndarray) -> str:
    """仅根据照片像素内容生成缓存键。"""
    contiguous = np.ascontiguousarray(image)
    hasher = hashlib.sha256()
    hasher.update(CACHE_VERSION.encode("ascii"))
    hasher.update(b"\0")
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(b"\0")
    hasher.update(str(contiguous.shape).encode("ascii"))
    hasher.update(b"\0")
    # 直接用缓冲区喂给 hasher，避免 tobytes() 为大体量 16-bit 扫描再复制一份内存。
    hasher.update(memoryview(contiguous).cast("B"))
    return hasher.hexdigest()


def _db_path() -> Path:
    return cache_root() / DB_FILENAME


def _max_cache_bytes() -> int:
    raw = os.environ.get("NCC_CACHE_MAX_MB", "").strip()
    try:
        megabytes = int(raw) if raw else DEFAULT_MAX_CACHE_MB
    except ValueError:
        megabytes = DEFAULT_MAX_CACHE_MB
    return max(megabytes, 1) * 1024 * 1024


def _connect() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh_database = not path.exists()

    conn = sqlite3.connect(path, timeout=5.0)
    try:
        # auto_vacuum 必须在建表和切换 WAL 之前设置才会生效；
        # FULL 让淘汰旧条目后数据库文件真正缩小，而不只是留下空闲页。
        conn.execute("PRAGMA auto_vacuum=FULL")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
    except Exception:
        conn.close()
        raise

    if fresh_database:
        _cleanup_legacy()
    return conn


def _cleanup_legacy() -> None:
    """首次创建新版数据库时清理旧版缓存文件（均为可重建数据）。"""
    root = cache_root()
    candidates = list(root.glob(LEGACY_JPEG_GLOB))
    candidates.append(root / LEGACY_MANIFEST)

    removed = 0
    freed = 0
    for path in candidates:
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        removed += 1
        freed += size

    if removed:
        print(f"[Cache] 已清理旧版缓存 {removed} 个文件"
              f"（{freed / 1048576:.1f} MB）")


def load_preprocessed_jpeg(photo_key: str) -> Optional[bytes]:
    """从缓存数据库读取预处理预览 JPEG；不存在或损坏时返回 None。"""
    if not cache_enabled():
        return None

    try:
        with closing(_connect()) as conn:
            row = conn.execute(
                "SELECT data FROM preprocess WHERE photo_key = ?",
                (photo_key,),
            ).fetchone()
            if row is None:
                return None

            data = row[0]
            if not data or len(data) < 2 or data[:2] != b"\xff\xd8":
                with conn:
                    conn.execute(
                        "DELETE FROM preprocess WHERE photo_key = ?",
                        (photo_key,),
                    )
                return None

            with conn:
                conn.execute(
                    "UPDATE preprocess SET last_used_at = ? "
                    "WHERE photo_key = ?",
                    (time.time_ns(), photo_key),
                )
            return data
    except (sqlite3.Error, OSError) as exc:
        print(f"[Cache] 读取缓存失败: {exc}")
        return None


def store_preprocessed_jpeg(photo_key: str, jpeg_bytes: bytes) -> bool:
    """写入预处理预览 JPEG；写入失败不影响校正流程。"""
    if not cache_enabled() or not jpeg_bytes:
        return False

    now = time.time_ns()
    pruned = 0
    try:
        with closing(_connect()) as conn:
            with conn:
                conn.execute(
                    """
                    INSERT INTO preprocess
                        (photo_key, data, bytes, created_at, last_used_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(photo_key) DO UPDATE SET
                        data = excluded.data,
                        bytes = excluded.bytes,
                        last_used_at = excluded.last_used_at
                    """,
                    (photo_key, jpeg_bytes, len(jpeg_bytes), now, now),
                )
                pruned = _prune_entries(conn)
                # auto_vacuum=FULL 会在提交时回收被删除条目占用的页，
                # 因此这里不再需要额外 VACUUM。

        if pruned:
            print(f"[Cache] 缓存超出容量上限，已淘汰 {pruned} 条最久未用记录")
        return True
    except (sqlite3.Error, OSError) as exc:
        print(f"[Cache] 写入缓存失败: {exc}")
        return False


def _prune_entries(conn: sqlite3.Connection) -> int:
    """按最久未用顺序淘汰条目，使缓存回落到上限的 90%。"""
    limit = _max_cache_bytes()
    total = conn.execute(
        "SELECT COALESCE(SUM(bytes), 0) FROM preprocess"
    ).fetchone()[0]
    if total <= limit:
        return 0

    target = int(limit * 0.9)
    rows = conn.execute(
        "SELECT photo_key, bytes FROM preprocess "
        "ORDER BY last_used_at ASC, created_at ASC, rowid ASC"
    ).fetchall()

    victims = []
    for photo_key, size in rows:
        if total <= target:
            break
        victims.append((photo_key,))
        total -= size

    if victims:
        conn.executemany(
            "DELETE FROM preprocess WHERE photo_key = ?", victims)
    return len(victims)
