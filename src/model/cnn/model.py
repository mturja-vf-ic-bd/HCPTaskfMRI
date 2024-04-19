import torch.nn as nn
from torch.nn import init


class FMRI_CNN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_channels,
                 hidden_channel,
                 num_dense_layers,
                 num_classes,
                 dropout=0.1):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.mx_pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Convolution Layers
        for l in range(num_layers):
            in_ch = in_channels if l == 0 else hidden_channel
            layer = nn.Conv2d(in_channels=in_ch,
                              out_channels=hidden_channel,
                              kernel_size=(1, 3),
                              padding='same')
            self.conv_layers.append(layer)

        # Output dense layers
        self.dense_layers = nn.ModuleList()
        for l in range(num_dense_layers):
            in_ch = hidden_channel if l == 0 else 32
            out_ch = num_classes if l == num_dense_layers - 1 else 32
            self.dense_layers.append(nn.Linear(in_ch, out_ch))

        self.init_weights()

    def init_weights(self):
        # Initialize convolutional layers
        for conv_layer in self.conv_layers:
            init.xavier_uniform_(conv_layer.weight)
            if conv_layer.bias is not None:
                init.constant_(conv_layer.bias, 0)

        # Initialize dense layers
        for dense_layer in self.dense_layers:
            if isinstance(dense_layer, nn.Linear):
                init.xavier_uniform_(dense_layer.weight)
                if dense_layer.bias is not None:
                    init.constant_(dense_layer.bias, 0)

    def forward(self, input):
        x = input.transpose(1, 2).unsqueeze(2)

        # Convolution Operations
        for l in range(len(self.conv_layers) // 2):
            x = self.dropout(x)
            x = self.conv_layers[2 * l](x)
            x = self.conv_layers[2 * l + 1](x)
            x = self.mx_pool(x)
            x = nn.ReLU()(x)

        x = x.flatten(start_dim=1)
        for l, dense_layer in enumerate(self.dense_layers):
            x = self.dropout(x)
            x = dense_layer(x)
            if l < len(self.dense_layers) - 1:
                x = nn.ReLU()(x)
        return x


class FMRI_CNN_SIMPLE(nn.Module):
    def __init__(self,
                 num_nodes,
                 num_dense_layers,
                 num_classes,
                 dropout=0.1):
        super().__init__()

        # Output dense layers
        self.dense_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)
        for l in range(num_dense_layers):
            in_ch = num_nodes if l == 0 else 32
            out_ch = num_classes if l == num_dense_layers - 1 else 32
            self.dense_layers.append(nn.Linear(in_ch, out_ch))
        self.init_weights()

    def init_weights(self):
        # Initialize dense layers
        for dense_layer in self.dense_layers:
            if isinstance(dense_layer, nn.Linear):
                init.xavier_uniform_(dense_layer.weight)
                if dense_layer.bias is not None:
                    init.constant_(dense_layer.bias, 0)

    def forward(self, input):
        x = input.mean(dim=1)
        for l, dense_layer in enumerate(self.dense_layers):
            x = self.dropout(x)
            x = dense_layer(x)
            if l < len(self.dense_layers) - 1:
                x = nn.ReLU()(x)
        return x
