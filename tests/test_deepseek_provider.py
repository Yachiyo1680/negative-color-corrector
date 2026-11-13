import tempfile
import unittest
from pathlib import Path

from core.config_manager import ConfigManager, DEFAULT_CONFIG_CONTENT
from core.model_provider import PROVIDERS, get_default_models


class DeepSeekProviderTests(unittest.TestCase):
    def test_provider_is_registered_with_default_model(self):
        provider = PROVIDERS["deepseek"]

        self.assertEqual(provider.base_url, "https://api.deepseek.com")
        self.assertEqual(provider.api_key_env, "DEEPSEEK_API_KEY")
        self.assertEqual(provider.default_model, "deepseek-flash")

    def test_default_model_is_available_as_fallback(self):
        self.assertIn("deepseek-flash", get_default_models("deepseek"))

    def test_default_config_contains_deepseek_key(self):
        self.assertIn("deepseek_api_key", DEFAULT_CONFIG_CONTENT)
        self.assertIn("${DEEPSEEK_API_KEY}", DEFAULT_CONFIG_CONTENT)

    def test_config_manager_reads_deepseek_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json5"
            config_path.write_text(
                '{ providers: { deepseek_api_key: "test-deepseek-key" } }',
                encoding="utf-8",
            )
            manager = ConfigManager(str(config_path))
            manager.load()

            self.assertEqual(
                manager.get_api_key("deepseek"),
                "test-deepseek-key",
            )


if __name__ == "__main__":
    unittest.main()
