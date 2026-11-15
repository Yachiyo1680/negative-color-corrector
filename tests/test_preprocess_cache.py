import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import mask_analyzer
from core.cache import (
    DB_FILENAME,
    build_photo_cache_key,
    load_preprocessed_jpeg,
    store_preprocessed_jpeg,
)
from core.engine import CorrectionConfig, Engine
from core.mask_analyzer import MaskResult


def _db_entry_count(cache_dir: str) -> int:
    conn = sqlite3.connect(Path(cache_dir) / DB_FILENAME)
    try:
        return conn.execute("SELECT COUNT(*) FROM preprocess").fetchone()[0]
    finally:
        conn.close()


class PreprocessCacheTests(unittest.TestCase):
    def test_cache_key_depends_only_on_photo_pixels(self):
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        same_image = image.copy()
        different_image = image.copy()
        different_image[0, 0, 0] = 1

        self.assertEqual(
            build_photo_cache_key(image),
            build_photo_cache_key(same_image),
        )
        self.assertNotEqual(
            build_photo_cache_key(image),
            build_photo_cache_key(different_image),
        )

    def test_cache_round_trip_uses_single_sqlite_database(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"NCC_CACHE_DIR": tmp}):
            key = "test-photo"
            jpeg = b"\xff\xd8fake-jpeg"

            self.assertTrue(store_preprocessed_jpeg(key, jpeg))
            self.assertEqual(load_preprocessed_jpeg(key), jpeg)

            self.assertTrue((Path(tmp) / DB_FILENAME).exists())
            self.assertEqual(_db_entry_count(tmp), 1)
            # 不再产生每张照片一个的 JPEG 文件，也没有 JSON 索引。
            self.assertEqual(list(Path(tmp).glob("*.jpg")), [])
            self.assertFalse((Path(tmp) / "index.json").exists())

    def test_corrupt_entry_is_removed(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"NCC_CACHE_DIR": tmp}):
            self.assertTrue(store_preprocessed_jpeg("broken", b"not-a-jpeg"))

            self.assertIsNone(load_preprocessed_jpeg("broken"))
            self.assertEqual(_db_entry_count(tmp), 0)

    def test_cache_evicts_least_recently_used_when_over_limit(self):
        blob = b"\xff\xd8" + b"x" * (600 * 1024 - 2)

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {
                 "NCC_CACHE_DIR": tmp,
                 "NCC_CACHE_MAX_MB": "1",
             }):
            self.assertTrue(store_preprocessed_jpeg("old", blob))
            self.assertTrue(store_preprocessed_jpeg("middle", blob))
            self.assertTrue(store_preprocessed_jpeg("new", blob))

            # 1 MB 上限、三个 600 KB 条目 -> 只保留最新的一个。
            self.assertIsNone(load_preprocessed_jpeg("old"))
            self.assertIsNone(load_preprocessed_jpeg("middle"))
            self.assertEqual(load_preprocessed_jpeg("new"), blob)
            self.assertEqual(_db_entry_count(tmp), 1)

    def test_first_store_cleans_legacy_cache_files(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"NCC_CACHE_DIR": tmp}):
            legacy_jpeg = Path(tmp) / "preprocess-v2-abc.jpg"
            legacy_manifest = Path(tmp) / "index.json"
            legacy_jpeg.write_bytes(b"old")
            legacy_manifest.write_text(
                json.dumps({"version": "preprocess-v2"}),
                encoding="utf-8",
            )

            self.assertTrue(store_preprocessed_jpeg("k", b"\xff\xd8x"))

            self.assertFalse(legacy_jpeg.exists())
            self.assertFalse(legacy_manifest.exists())
            self.assertEqual(_db_entry_count(tmp), 1)

    def test_same_photo_reuses_preprocess_across_model_parameters(self):
        image = np.full((40, 40, 3), 40000.0, dtype=np.float32)
        regions = json.dumps({
            "regions": [{
                "description": "neutral gray wall",
                "center": [0.5, 0.5],
                "rgb": [100, 110, 105],
            }]
        })

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {
                 "NCC_CACHE_DIR": tmp,
                 "NCC_DISABLE_CACHE": "",
             }), \
             patch(
                 "core.cast_detector._call_vlm_api",
                 return_value=regions,
             ), \
             patch(
                 "core.mask_analyzer._prepare_vlm_preview",
                 wraps=mask_analyzer._prepare_vlm_preview,
             ) as prepare:
            first = mask_analyzer._vlm_find_neutral_gray(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "model-a",
                "timeout": 1,
                "reasoning_effort": "off",
            })

            self.assertIsNotNone(first)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(_db_entry_count(tmp), 1)

            with patch(
                "core.mask_analyzer._prepare_vlm_preview",
                side_effect=AssertionError("preprocess should hit cache"),
            ):
                second = mask_analyzer._vlm_find_neutral_gray(image, {
                    "api_base": "https://example.invalid/v1",
                    "api_key": "test-key",
                    "model": "model-b",
                    "timeout": 1,
                    "reasoning_effort": "high",
                })

        self.assertIsNotNone(second)

    def test_engine_checks_cache_before_inversion(self):
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
        events = []

        def fake_load(_key):
            events.append("cache")
            return b"\xff\xd8fake-jpeg"

        def fake_invert(img, max_val=255.0):
            events.append("invert")
            return img

        mask_result = MaskResult(
            method="edge",
            ref_r=128.0,
            ref_g=128.0,
            ref_b=128.0,
            scale_r=1.0,
            scale_g=1.0,
            scale_b=1.0,
            confidence=1.0,
        )
        config = CorrectionConfig(
            film_type="negative",
            vlm_api_key="test-key",
            detector_mode="heuristic",
            max_iterations=0,
        )

        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {
                 "NCC_CACHE_DIR": tmp,
                 "NCC_DISABLE_CACHE": "",
             }), \
             patch(
                 "core.engine.load_preprocessed_jpeg",
                 side_effect=fake_load,
             ), \
             patch("core.engine.invert", side_effect=fake_invert), \
             patch(
                 "core.engine.analyze_mask",
                 return_value=mask_result,
             ):
            Engine(config).correct(image)

        self.assertEqual(events, ["cache", "invert"])


if __name__ == "__main__":
    unittest.main()
