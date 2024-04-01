import unittest
import torch

from src.model.wavenet.wavenet_model import WaveNetModel


class TestWaveNetModel(unittest.TestCase):
    def setUp(self):
        # Initialize WaveNetModel with default parameters
        self.model = WaveNetModel()

    def test_forward(self):
        # Test the forward pass of the model
        batch_size = 4
        input_channels = 256
        input_length = 100
        input_tensor = torch.randn(batch_size, input_channels, input_length)
        output = self.model.forward(input_tensor)
        # Ensure output tensor has the correct shape
        self.assertEqual(output.shape, torch.Size([batch_size * self.model.output_length, self.model.classes]))
