import torch.nn as nn


class VanillaTransformer(nn.Module):
    def __init__(self,
                 num_layers,
                 num_heads,
                 d_model,
                 dim_feedforward,
                 num_dense_layers,
                 num_classes,
                 dropout=0.1):
        super().__init__()
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_block, num_layers=num_layers)

        # Output dense layers
        self.dense_layers = nn.ModuleList()
        in_features = d_model  # Assuming the transformer's output matches d_model
        for l in range(num_dense_layers):
            out_features = num_classes if l == num_dense_layers - 1 else 32
            self.dense_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features  # Update in_features for the next layer

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        x = self.transformer_encoder(input)
        x = x.mean(dim=1)
        for module in self.dense_layers:
            x = module(x)
        return x

