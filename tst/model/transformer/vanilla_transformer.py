import torch
import unittest
from src.model.transformer.vanilla_transformer import VanillaTransformer


class TestVanillaTransformer(unittest.TestCase):
    def setUp(self):
        # Initialize the model parameters for testing
        self.num_layers = 3
        self.d_model = 360
        self.dim_feedforward = 64
        self.num_dense_layers = 2
        self.num_classes = 21
        self.num_heads = 10
        self.dropout = 0.1

        # Initialize the model
        self.model = VanillaTransformer(
            num_layers=self.num_layers,
            d_model=self.d_model,
            num_heads=self.num_heads,
            dim_feedforward=self.dim_feedforward,
            num_dense_layers=self.num_dense_layers,
            num_classes=self.num_classes,
            dropout=self.dropout
        )

    def test_forward_pass(self):
        """Test the forward pass with a dummy input."""
        batch_size = 4
        seq_length = 16
        # Create a dummy input tensor
        dummy_input = torch.randn(batch_size, seq_length, self.d_model)
        # Forward pass
        output = self.model(dummy_input)
        # Check the output shape
        self.assertEqual(output.size(), (batch_size, self.num_classes))

    def test_model_structure(self):
        """Test the model structure, e.g., the number of layers."""
        # Check the number of transformer layers
        self.assertEqual(len(self.model.transformer_encoder.layers), self.num_layers)
        # Check the number of dense layers
        self.assertEqual(len(self.model.dense_layers), self.num_dense_layers)

    def test_dropout_rate(self):
        """Test if the dropout rate is correctly set."""
        for layer in self.model.transformer_encoder.layers:
            self.assertEqual(layer.dropout.p, self.dropout)


# This line is necessary to run the tests when the script is executed
if __name__ == '__main__':
    unittest.main()
