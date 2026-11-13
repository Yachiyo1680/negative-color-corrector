"""校正预处理缓存。

缓存在反相前按照片内容命中，不包含模型、思考强度等模型参数。缓存文件放在
配置文件所在目录的 cache/preprocess/ 下，用于复用与模型无关的预处理结果。
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


CACHE_VERSION = "preprocess-v2"
MANIFEST_FILENAME = "index.json"


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
    hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


def _cache_path(photo_key: str) -> Path:
    return cache_root() / f"{CACHE_VERSION}-{photo_key}.jpg"


def _manifest_path() -> Path:
    return cache_root() / MANIFEST_FILENAME


def _load_manifest() -> dict:
    """读取照片 hash 索引；损坏时按空索引处理。"""
    try:
        data = json.loads(_manifest_path().read_text(encoding="utf-8"))
        if data.get("version") == CACHE_VERSION:
            entries = data.get("entries")
            if isinstance(entries, dict):
                return entries
    except (OSError, ValueError, TypeError):
        pass
    return {}


def _store_manifest(entries: dict) -> bool:
    """原子写入照片 hash 索引。"""
    path = _manifest_path()
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    payload = {
        "version": CACHE_VERSION,
        "entries": entries,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
        return True
    except OSError as exc:
        print(f"[Cache] 写入缓存索引失败: {exc}")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def load_preprocessed_jpeg(photo_key: str) -> Optional[bytes]:
    """读取预处理预览 JPEG；不存在或损坏时返回 None。"""
    manifest_entry = _load_manifest().get(photo_key)
    path = _cache_path(photo_key)
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if len(data) < 2 or data[:2] != b"\xff\xd8":
        return None
    # 索引存在但缺少当前 hash 时仍可回退到文件名查找，兼容旧缓存。
    if manifest_entry is not None:
        expected_file = manifest_entry.get("file")
        if expected_file and expected_file != path.name:
            return None
    return data


def store_preprocessed_jpeg(photo_key: str, jpeg_bytes: bytes) -> bool:
    """原子写入预处理预览 JPEG；写入失败不影响校正流程。"""
    if not jpeg_bytes:
        return False

    path = _cache_path(photo_key)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_bytes(jpeg_bytes)
        os.replace(tmp_path, path)
        entries = _load_manifest()
        entries[photo_key] = {
            "original_hash": photo_key,
            "file": path.name,
            "bytes": len(jpeg_bytes),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        _store_manifest(entries)
        return True
    except OSError as exc:
        print(f"[Cache] 写入预处理缓存失败: {exc}")
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        return False
