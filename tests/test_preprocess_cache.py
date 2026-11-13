import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core import mask_analyzer
from core.cache import (
    CACHE_VERSION,
    build_photo_cache_key,
    load_preprocessed_jpeg,
    store_preprocessed_jpeg,
)
from core.engine import CorrectionConfig, Engine
from core.mask_analyzer import MaskResult


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

    def test_jpeg_cache_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp, \
             patch.dict(os.environ, {"NCC_CACHE_DIR": tmp}):
            key = "test-photo"
            jpeg = b"\xff\xd8fake-jpeg"

            self.assertTrue(store_preprocessed_jpeg(key, jpeg))
            self.assertEqual(load_preprocessed_jpeg(key), jpeg)
            self.assertEqual(len(list(Path(tmp).glob("*.jpg"))), 1)
            manifest = json.loads(
                (Path(tmp) / "index.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["entries"][key]["original_hash"], key)
            self.assertEqual(
                manifest["entries"][key]["file"],
                f"{CACHE_VERSION}-{key}.jpg",
            )

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
            self.assertEqual(len(list(Path(tmp).glob("*.jpg"))), 1)

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
