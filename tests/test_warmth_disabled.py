import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.config_manager import ConfigManager
from core.engine import CorrectionConfig, Engine


class WarmthDisabledTests(unittest.TestCase):
    def test_warmth_is_disabled_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = ConfigManager(str(Path(tmp) / "config.json5"))
            config = manager.load()

        self.assertFalse(config.correction.warmth_enabled)

    def test_config_can_enable_warmth(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.json5"
            config_path.write_text(
                '{ correction: { warmth_enabled: true } }',
                encoding="utf-8",
            )
            config = ConfigManager(str(config_path)).load()

        self.assertTrue(config.correction.warmth_enabled)

    def test_engine_skips_warmth_when_disabled(self):
        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        config = CorrectionConfig(
            film_type="positive",
            warmth_style="natural",
            warmth_enabled=False,
            detector_mode="heuristic",
            max_iterations=0,
        )

        with patch("core.engine.apply_warmth") as apply_warmth:
            result = Engine(config).correct(image)

        apply_warmth.assert_not_called()
        self.assertEqual(result.warm_style, "none")


if __name__ == "__main__":
    unittest.main()
