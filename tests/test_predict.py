"""
Unit tests for unified production inference API (`src.inference.predict`).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from PIL import Image

from src.inference.predict import MultimodalPredictor, detect_language


class TestInferenceAPI(unittest.TestCase):

    def test_language_detection(self):
        self.assertEqual(detect_language("Four people trapped in flooded house"), "English")
        self.assertEqual(detect_language("लोग फँसे हुए हैं बाढ़ के पानी में"), "Hindi")
        self.assertEqual(detect_language("Paani ghar ke andar aa gaya hai four people phase hue hain"), "Hinglish")
        self.assertEqual(detect_language(""), "N/A")

    def test_text_only_prediction(self):
        predictor = MultimodalPredictor(image_model_type="swin_t")
        res = predictor.predict(text_report="Four people trapped in rising floodwater, urgent rescue needed")
        d = res.to_dict()

        self.assertTrue(0.0 <= res.priority_score <= 100.0)
        self.assertIn(res.priority_level, ["Very Low", "Low", "Medium", "High", "Critical"])
        self.assertEqual(res.language_detected, "English")
        self.assertEqual(res.input_mode, "Text Only")
        self.assertIn("disaster_type", d)
        self.assertIn("priority_score", d)

    def test_hinglish_text_prediction(self):
        predictor = MultimodalPredictor(image_model_type="swin_t")
        res = predictor.predict(text_report="Jangal ki aag ka dhua bohot zyada hai, 4 log phase hue hain")

        self.assertTrue(0.0 <= res.priority_score <= 100.0)
        self.assertEqual(res.language_detected, "Hinglish")
        self.assertEqual(res.input_mode, "Text Only")

    def test_image_only_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "test.jpg"
            img = Image.new("RGB", (224, 224), color="blue")
            img.save(img_path)

            predictor = MultimodalPredictor(image_model_type="swin_t")
            res = predictor.predict(image_input=img_path)

            self.assertEqual(res.input_mode, "Image Only")
            self.assertIn(res.disaster_type, ["Flood", "Earthquake", "Urban Fire", "Wild Fire", "Landslide", "Drought"])
            self.assertTrue(0.0 <= res.confidence <= 1.0)
            self.assertEqual(res.language_detected, "N/A")

    def test_multimodal_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            img_path = Path(tmp_dir) / "test_multi.jpg"
            img = Image.new("RGB", (224, 224), color="red")
            img.save(img_path)

            predictor = MultimodalPredictor(image_model_type="swin_t")
            res = predictor.predict(image_input=img_path, text_report="Ghar me paani bhar gaya hai rescue team bhejo")

            self.assertEqual(res.input_mode, "Multimodal (Image + Text)")
            self.assertEqual(res.language_detected, "Hinglish")
            self.assertTrue(0.0 <= res.priority_score <= 100.0)
            self.assertTrue(0.0 <= res.confidence <= 1.0)


if __name__ == "__main__":
    unittest.main()
