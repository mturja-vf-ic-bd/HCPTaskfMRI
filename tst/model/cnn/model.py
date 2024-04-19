import unittest
import torch

from src.model.cnn.model import FMRI_CNN


class TestFMRI_CNN(unittest.TestCase):
    def setUp(self):
        self.model = FMRI_CNN(8, 360, 64, 3, 16)
        print(self.model)

    def test_forward(self):
        x = torch.randn(2, 16, 360)
        y = self.model(x)
        self.assertEqual(y.shape, (2, 16))
