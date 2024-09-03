# tests/test_models.py

import unittest
import torch
from src.models.foundation_model import DetrObjectDetectionModel

class TestDetrObjectDetectionModel(unittest.TestCase):
    def setUp(self):
        self.model = DetrObjectDetectionModel(num_classes=91)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)

    def test_forward_pass(self):
        pixel_values = torch.randn(1, 3, 224, 224).to(self.device)  # Dummy input
        outputs = self.model.forward(pixel_values=pixel_values)
        
        self.assertIsNotNone(outputs, "The model output should not be None.")
        self.assertIn("logits", outputs, "The model output should contain 'logits'.")
        self.assertIn("pred_boxes", outputs, "The model output should contain 'pred_boxes'.")

    def test_model_save_and_load(self):
        self.model.save("tests/test_model")
        loaded_model = DetrObjectDetectionModel(num_classes=91)
        loaded_model.load("tests/test_model")

        self.assertEqual(self.model.config.to_dict(), loaded_model.config.to_dict(), "The loaded model's configuration should match the saved model.")

if __name__ == '__main__':
    unittest.main()
