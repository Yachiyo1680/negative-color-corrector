import base64
import io
import unittest

import numpy as np
from PIL import Image

from core.cast_detector import VLM_MAX_IMAGE_EDGE, _encode_image


class VLMImageSizeTests(unittest.TestCase):
    def test_large_image_is_downscaled_before_upload(self):
        image = np.zeros((100, 4200, 3), dtype=np.uint8)

        encoded = _encode_image(image)
        decoded = base64.b64decode(encoded)
        with Image.open(io.BytesIO(decoded)) as img:
            self.assertEqual(max(img.size), VLM_MAX_IMAGE_EDGE)


if __name__ == "__main__":
    unittest.main()
