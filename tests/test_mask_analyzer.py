import json
import unittest
from unittest.mock import patch

import numpy as np

from core import mask_analyzer


class MaskAnalyzerTests(unittest.TestCase):
    def test_find_gray_pixels_supports_16bit_images(self):
        image = np.full((20, 20, 3), 40000.0, dtype=np.float32)

        mask = mask_analyzer._find_gray_pixels(image)

        self.assertEqual(mask.shape, image.shape[:2])
        self.assertTrue(mask.all())

    def test_vlm_neutral_gray_retries_after_transient_failure(self):
        image = np.full((20, 20, 3), 40000.0, dtype=np.float32)
        attempts = 0

        def fake_call(**_kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise RuntimeError("temporary API failure")
            return json.dumps({
                "regions": [{
                    "description": "neutral gray wall",
                    "center": [0.5, 0.5],
                    "rgb": [100, 110, 105],
                }]
            })

        with patch("core.cast_detector._call_vlm_api", side_effect=fake_call):
            result = mask_analyzer._vlm_find_neutral_gray(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
            })

        self.assertEqual(attempts, 2)
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "vlm_gray")

    def test_vlm_neutral_gray_reports_empty_regions(self):
        image = np.full((20, 20, 3), 40000.0, dtype=np.float32)

        with patch(
            "core.cast_detector._call_vlm_api",
            return_value=json.dumps({"regions": []}),
        ):
            result = mask_analyzer._vlm_find_neutral_gray(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
            })

        self.assertIsNone(result)

    def test_vlm_invalid_regions_do_not_raise(self):
        image = np.full((20, 20, 3), 40000.0, dtype=np.float32)

        with patch(
            "core.cast_detector._call_vlm_api",
            return_value=json.dumps({"regions": [{"description": "bad"}]}),
        ):
            result = mask_analyzer._vlm_find_neutral_gray(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
            })

        self.assertIsNone(result)

    def test_analyze_mask_prefers_vlm_over_edge_reference(self):
        image = np.full((40, 40, 3), 40000.0, dtype=np.float32)
        vlm_result = mask_analyzer.MaskResult(
            method="vlm_gray",
            ref_r=100.0,
            ref_g=110.0,
            ref_b=105.0,
            scale_r=1.0,
            scale_g=1.0,
            scale_b=1.0,
            confidence=0.8,
            detail="neutral paper",
        )

        with patch.object(
            mask_analyzer,
            "_vlm_find_neutral_gray",
            return_value=vlm_result,
        ) as find_vlm:
            result = mask_analyzer.analyze_mask(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
            })

        find_vlm.assert_called_once()
        self.assertEqual(result.method, "vlm_gray")

    def test_neutral_gray_prompt_requests_primary_material_selection(self):
        image = np.full((40, 40, 3), 40000.0, dtype=np.float32)
        captured = {}

        def fake_call(**kwargs):
            captured["prompt"] = kwargs["prompt"]
            return json.dumps({
                "regions": [{
                    "description": "neutral paper",
                    "location": "upper left",
                    "center": [0.5, 0.5],
                    "rgb": [100, 110, 105],
                    "confidence": 0.9,
                }]
            })

        with patch("core.cast_detector._call_vlm_api", side_effect=fake_call):
            result = mask_analyzer._vlm_find_neutral_gray(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
            })

        self.assertEqual(result.method, "vlm_gray")
        self.assertIn("PRIMARY reference selector", captured["prompt"])
        self.assertIn("neutral white or gray paper/card", captured["prompt"])
        self.assertIn("strong shadows", captured["prompt"])

    def test_adaptive_sampling_returns_window_mean_and_bounded_radius(self):
        image = np.zeros((40, 40, 3), dtype=np.float32)
        image[18:22, 18:22, :] = [100.0, 200.0, 300.0]

        res = mask_analyzer._sample_neighborhood_adaptive(
            image, [0.5, 0.5], side_min=3, side_max=11)

        self.assertIsNotNone(res)
        ref, r = res
        self.assertGreaterEqual(r, 1)     # side>=3 -> r>=1
        self.assertLessEqual(r, 5)        # side<=11 -> r<=5
        self.assertEqual(ref.shape, (3,))

    def test_adaptive_sampling_returns_none_when_no_bounds_fit(self):
        # 2x2 图：最小 side=3 已在中心越界，全部候选不可用 -> None
        image = np.zeros((2, 2, 3), dtype=np.float32)
        res = mask_analyzer._sample_neighborhood_adaptive(
            image, [0.5, 0.5], side_min=3, side_max=11)
        self.assertIsNone(res)

    def test_adaptive_sampling_rejects_bad_center(self):
        image = np.zeros((20, 20, 3), dtype=np.float32)
        self.assertIsNone(mask_analyzer._sample_neighborhood_adaptive(
            image, [0.5], side_min=3, side_max=11))          # 长度不对
        self.assertIsNone(mask_analyzer._sample_neighborhood_adaptive(
            image, [1.5, 0.5], side_min=3, side_max=11))      # 越出 0-1
        self.assertIsNone(mask_analyzer._sample_neighborhood_adaptive(
            image, ["a", "b"], side_min=3, side_max=11))      # 非数值

    def test_vlm_receives_algorithmically_precorrected_preview(self):
        image = np.full((40, 40, 3), 40000.0, dtype=np.float32)
        captured = {}

        def fake_encode(preview):
            captured["preview"] = preview.copy()
            return "encoded"

        with patch("core.cast_detector._encode_image", side_effect=fake_encode), \
             patch(
                 "core.cast_detector._call_vlm_api",
                 return_value=json.dumps({"regions": []}),
             ):
            result = mask_analyzer._vlm_find_neutral_gray(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
                "levels_percentile": 0.2,
            })

        self.assertIsNone(result)
        preview = captured["preview"]
        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(preview.shape, image.shape)
        self.assertFalse(np.array_equal(preview, image.astype(np.uint8)))

    def test_analyze_mask_falls_back_to_edge_after_vlm_failure(self):
        image = np.full((40, 40, 3), 40000.0, dtype=np.float32)

        with patch.object(
            mask_analyzer,
            "_vlm_find_neutral_gray",
            return_value=None,
        ):
            result = mask_analyzer.analyze_mask(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
            })

        self.assertEqual(result.method, "edge")

    def test_analyze_mask_falls_back_to_global_after_vlm_and_edge_failure(self):
        rng = np.random.default_rng(1234)
        image = rng.integers(0, 65535, size=(40, 40, 3)).astype(np.float32)

        with patch.object(
            mask_analyzer,
            "_vlm_find_neutral_gray",
            return_value=None,
        ):
            result = mask_analyzer.analyze_mask(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
            })

        self.assertEqual(result.method, "global")

    def test_16bit_edge_variation_uses_edge_fallback(self):
        image = np.full((40, 40, 3), 40000.0, dtype=np.float32)
        image[:, :2, :] += np.array([100.0, -100.0, 50.0], dtype=np.float32)

        with patch.object(
            mask_analyzer,
            "_vlm_find_neutral_gray",
            return_value=None,
        ):
            result = mask_analyzer.analyze_mask(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
            })

        self.assertEqual(result.method, "edge")

    def test_edge_fallback_checks_all_four_sides(self):
        image = np.zeros((40, 40, 3), dtype=np.float32)
        image[:, :20, 0] = np.tile(np.linspace(0, 65535, 20), (40, 1))
        image[:, :20, 1] = np.tile(np.linspace(65535, 0, 20), (40, 1))
        image[:, :20, 2] = 32768.0
        image[:, 20:, :] = [40000.0, 42000.0, 41000.0]

        with patch.object(
            mask_analyzer,
            "_vlm_find_neutral_gray",
            return_value=None,
        ):
            result = mask_analyzer.analyze_mask(image, {
                "api_base": "https://example.invalid/v1",
                "api_key": "test-key",
                "model": "test-model",
                "timeout": 1,
            })

        self.assertEqual(result.method, "edge")
        self.assertEqual(result.ref_r, 40000.0)
        self.assertEqual(result.ref_g, 42000.0)
        self.assertEqual(result.ref_b, 41000.0)

    def test_vlm_native_16bit_rgb_is_not_scaled_twice(self):
        restored = mask_analyzer._restore_vlm_rgb(
            np.array([30000, 36000, 37000]), 55511,
        )

        np.testing.assert_allclose(restored, [30000, 36000, 37000])


if __name__ == "__main__":
    unittest.main()
