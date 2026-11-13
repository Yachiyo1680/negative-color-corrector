import os
import tempfile
import unittest

from main import _default_output, _normalize_output_filename


class OutputFilenameTests(unittest.TestCase):
    def test_explicit_extension_is_kept(self):
        self.assertEqual(
            _normalize_output_filename("comparison.jpg", "scan.tiff"),
            "comparison.jpg",
        )

    def test_missing_extension_inherits_input_extension(self):
        self.assertEqual(
            _normalize_output_filename("comparison", "scan.tiff"),
            "comparison.tiff",
        )

    def test_directory_is_rejected(self):
        with self.assertRaises(ValueError):
            _normalize_output_filename("output/comparison.jpg", "scan.tiff")

    def test_empty_filename_is_rejected(self):
        with self.assertRaises(ValueError):
            _normalize_output_filename("   ", "scan.tiff")

    def test_custom_filename_still_uses_corrected_subdirectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = os.path.join(tmp, "scan.tiff")
            output_path = _default_output(input_path, "comparison.jpg")

            self.assertEqual(
                output_path,
                os.path.join(tmp, "_corrected", "comparison.jpg"),
            )
            self.assertTrue(os.path.isdir(os.path.join(tmp, "_corrected")))


if __name__ == "__main__":
    unittest.main()
