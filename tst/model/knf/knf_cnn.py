import unittest
import torch

from src.model.knf.knf_cnn import KNF_CNN


class TestKNFCNN(unittest.TestCase):
    def setUp(self):
        self.batch_size, self.num_nodes, self.time_points = 2, 360, 16
        self.x = torch.randn(self.batch_size, self.time_points, self.num_nodes)
        self.y = torch.randn(self.batch_size, 4, self.num_nodes)
        self.label = torch.randint(low=0, high=21, size=(self.batch_size, 1))
        self.model = KNF_CNN(4, 16, 8, 128, 256, 0.1, self.time_points, num_classes=21)
        print(self.model)

    def testForward(self):
        x, y, l = self.x, self.y, self.label
        model = self.model
        output = model(x, y, l)
        K = output["KoopmanOp"]
        x_recon = output["recon"]
        x_lookback = output["lookback"]
        x_lookahead = output["lookahead"]
        self.assertEqual(K.shape, (self.batch_size, self.num_nodes, self.num_nodes))
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)
        self.assertEqual(x.shape, x_recon.shape)
        self.assertEqual(x[:, :, 1:].shape, x_lookback.shape)
        self.assertEqual(x[:, :, 1:].shape, x_lookback.shape)
        self.assertEqual(y.shape, x_lookahead.shape)
