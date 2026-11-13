import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.cast_detector import DetectorFactory, _call_vlm_api
from core.config_manager import ConfigManager
from core.model_provider import APIConfig, build_vlm_config


def _fake_response():
    response = Mock()
    response.ok = True
    response.json.return_value = {
        "choices": [{"message": {"content": '{"cast": "ok"}'}}]
    }
    return response


class ReasoningEffortTests(unittest.TestCase):
    def _post_payload(self, api_base, model, reasoning_effort):
        with patch(
            "core.cast_detector.requests.post",
            return_value=_fake_response(),
        ) as post:
            _call_vlm_api(
                api_base=api_base,
                api_key="test-key",
                model=model,
                prompt="test",
                image_b64="test",
                reasoning_effort=reasoning_effort,
            )
        return post.call_args.kwargs["json"]

    def test_auto_does_not_send_reasoning_parameter(self):
        payload = self._post_payload(
            "https://example.invalid/v1", "gpt-5", "auto")

        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("reasoning", payload)

    def test_openai_compatible_sends_reasoning_effort(self):
        payload = self._post_payload(
            "https://example.invalid/v1", "gpt-5", "high")

        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("reasoning", payload)

    def test_off_sends_none_for_openai_compatible_api(self):
        payload = self._post_payload(
            "https://example.invalid/v1", "gpt-5", "off")

        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("reasoning", payload)

    def test_off_uses_minimal_for_gemini_35(self):
        payload = self._post_payload(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-3.5-flash-lite",
            "off",
        )

        self.assertEqual(payload["reasoning_effort"], "minimal")
        self.assertNotIn("reasoning", payload)

    def test_off_uses_none_for_gemini_31(self):
        payload = self._post_payload(
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-3.1-flash-lite",
            "off",
        )

        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("reasoning", payload)

    def test_openrouter_uses_reasoning_object(self):
        payload = self._post_payload(
            "https://openrouter.ai/api/v1",
            "openai/gpt-5",
            "medium",
        )

        self.assertEqual(payload["reasoning"], {"effort": "medium"})
        self.assertNotIn("reasoning_effort", payload)

    def test_off_disables_reasoning_for_openrouter(self):
        payload = self._post_payload(
            "https://openrouter.ai/api/v1",
            "openai/gpt-5",
            "off",
        )

        self.assertEqual(payload["reasoning"], {"enabled": False})
        self.assertNotIn("reasoning_effort", payload)

    def test_unknown_value_is_ignored(self):
        payload = self._post_payload(
            "https://example.invalid/v1", "gpt-5", "extreme")

        self.assertNotIn("reasoning_effort", payload)

    def test_detector_factory_passes_reasoning_effort(self):
        detector, warning = DetectorFactory.create({
            "detector_mode": "vlm_api",
            "vlm_api_key": "test-key",
            "vlm_model": "gpt-5",
            "vlm_reasoning_effort": "low",
        })

        self.assertEqual(warning, "")
        self.assertEqual(detector.reasoning_effort, "low")

    def test_config_manager_reads_reasoning_effort(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json5"
            config_path.write_text(
                '{ detector: { reasoning_effort: "high" } }',
                encoding="utf-8",
            )
            config = ConfigManager(str(config_path)).load()

        self.assertEqual(config.detector.reasoning_effort, "high")

    def test_api_config_round_trip_preserves_reasoning_effort(self):
        api_config = APIConfig(reasoning_effort="medium")

        restored = APIConfig.from_dict(api_config.to_dict())
        vlm_config = build_vlm_config(restored)

        self.assertEqual(restored.reasoning_effort, "medium")
        self.assertEqual(vlm_config["vlm_reasoning_effort"], "medium")


if __name__ == "__main__":
    unittest.main()
