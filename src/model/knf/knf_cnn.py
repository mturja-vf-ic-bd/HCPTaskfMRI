import torch.nn as nn
import torch
from torch.nn import init

from src.model.cnn.model import FMRI_CNN, FMRI_CNN_SIMPLE


class KNF_CNN(nn.Module):
    def __init__(self,
                 backbone_layers,
                 backbone_hidden_channel,
                 cls_layers,
                 cls_hidden_channel,
                 transformer_layers,
                 transformer_hidden,
                 dropout=0.1,
                 time_points=16,
                 num_nodes=360,
                 num_classes=21,
                 add_global_attention=False
                 ):
        super(KNF_CNN, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_t_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Convolution Layers
        for l in range(backbone_layers):
            in_ch = 1 if l == 0 else backbone_hidden_channel
            layer = nn.Conv2d(in_channels=in_ch,
                              out_channels=backbone_hidden_channel,
                              kernel_size=(1, 3),
                              padding='same')
            self.conv_layers.append(layer)
        self.proj = nn.Linear(backbone_hidden_channel, 1)

        # Classification Layer
        self.classifier = FMRI_CNN(
            num_layers=cls_layers,
            in_channels=num_nodes,
            hidden_channel=cls_hidden_channel,
            num_dense_layers=1,
            num_classes=num_classes,
            dropout=dropout
        )

        # Reconstruction (Deconvolution) Layer
        for l in range(backbone_layers):
            in_ch = 1 if l == 0 else backbone_hidden_channel
            layer = nn.ConvTranspose2d(
                in_channels=in_ch,
                out_channels=backbone_hidden_channel,
                kernel_size=(1, 3))
            self.conv_t_layers.append(layer)
        self.proj_t = nn.Linear(backbone_hidden_channel, 1)

        # Koopman Attention Layer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=time_points,
            nhead=1,
            dim_feedforward=transformer_hidden)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=transformer_layers)
        if add_global_attention:
            self.global_attention = nn.Parameter(
                torch.FloatTensor(num_classes, num_nodes, num_nodes))
            nn.init.xavier_uniform_(self.global_attention, gain=0.001)
        else:
            self.global_attention = None

        self.attention = nn.MultiheadAttention(
            embed_dim=time_points,
            num_heads=1,
            batch_first=True)
        self.init_weights()

    def init_weights(self):
        # Initialize convolutional layers
        for conv_layer in self.conv_layers:
            init.xavier_uniform_(conv_layer.weight)
            if conv_layer.bias is not None:
                init.constant_(conv_layer.bias, 0)

        # Initialize dense layers
        for dense_layer in self.modules():
            if isinstance(dense_layer, nn.Linear):
                init.xavier_uniform_(dense_layer.weight)
                if dense_layer.bias is not None:
                    init.constant_(dense_layer.bias, 0)

    def compute_koopman_attention(self, z):
        """

        :param z: shape of z: (batch, N, T)
        :return: attention matrix of shape (batch, N, N)
        """
        z = self.transformer_encoder(z)
        K = self.attention(z, z, z)[1]
        return K

    def compute_global_attention(self, cls_scores):
        """

        :param cls_scores:
        :return:
        """
        cls_prob = nn.Softmax(dim=-1)(cls_scores)  # (batch_size, num_classes)
        K = torch.einsum('bc, cxy -> bxy',
                         cls_prob,
                         self.global_attention)
        return K

    def forward_prediction(self, z_init, K, iter):
            # Collect predicted measurements on the lookback window
            predictions = []
            forw = z_init
            for i in range(iter):
                forw = torch.einsum("bnl, blh -> bnh", K, forw)
                predictions.append(forw)
            embed_preds = torch.cat(predictions, dim=-1)
            return embed_preds

    def compute_classification_scores(self, x):
        x = x.unsqueeze(2)
        for i, cls_layer in enumerate(self.cls_layers):
            x = self.dropout(x)
            x = cls_layer(x)
            if i < len(self.cls_layers) - 1:
                x = nn.ReLU()(x)
                if i % 2 == 1:
                    x = self.mx_pool(x)
        cls_score = self.cls_head(x.mean(dim=-1).squeeze(2))
        return cls_score

    def encode(self, x):
        # Encoder propagates through backbone layer
        # to learn the node-wise projection to the Koopman Invariant Space
        for i, backbone_layer in enumerate(self.conv_layers):
            if i < len(self.conv_layers) - 1:
                x = self.dropout(x)
            x = backbone_layer(x)
            if i < len(self.conv_layers) - 1:
                x = nn.ReLU()(x)
        x = self.proj(x.transpose(1, 3)).squeeze(-1).transpose(1, 2)
        return x

    def decode(self, x):
        # Decoder to reconstruct original signal
        # from Koopman Invariant Space
        time_points = x.shape[2]
        x = x.unsqueeze(1)
        for i, backbone_layer in enumerate(self.conv_t_layers):
            if i < len(self.conv_t_layers) - 1:
                x = self.dropout(x)
            x = backbone_layer(x)
            x = x[:, :, :, :time_points]
            if i < len(self.conv_t_layers) - 1:
                x = nn.ReLU()(x)
        x = self.proj_t(x.transpose(1, 3)).squeeze(-1).transpose(1, 2)
        return x

    def mse(self, true, pred):
        loss_fn = nn.MSELoss()
        return loss_fn(true, pred)

    def crossentropy(self, true_label, cls_scores):
        loss_fn = nn.CrossEntropyLoss()
        return loss_fn(cls_scores, true_label.squeeze(1))

    def forward(self, x, y, labels):
        x = x.transpose(1, 2)
        y = y.transpose(1, 2)
        x = x.unsqueeze(1)
        z = self.encode(x)
        cls_scores = self.classifier(z.transpose(1, 2))
        K = self.compute_koopman_attention(z)
        if self.global_attention is not None:
            K_global = self.compute_global_attention(cls_scores)
        else:
            K_global = torch.zeros_like(K)
        lookback_predictions = self.forward_prediction(z[:, :, :1], K + K_global, iter=z.shape[-1] - 1)
        lookahead_predictions = self.forward_prediction(z[:, :, -1:], K + K_global, iter=y.shape[-1])

        # Reconstructions
        x_recon = self.decode(z)
        lookback_recon = self.decode(lookback_predictions)
        lookahead_recon = self.decode(lookahead_predictions)

        # Computer losses
        ### Reconstruction ###
        x = x.squeeze(1)
        recon_loss = self.mse(x_recon, x)
        lookback_loss = self.mse(lookback_recon, x[:, :, 1:])
        lookahead_loss = self.mse(lookahead_recon, y)
        latent_pred_loss = self.mse(lookback_predictions, z[:, :, 1:])

        ### Classification ###
        cls_loss = self.crossentropy(labels, cls_scores)

        output = {
            "recon": x_recon,
            "lookback": lookback_recon,
            "lookahead": lookahead_recon,
            "KoopmanOp": K,
            "cls_scores": cls_scores,
            "loss": {
                "recon": recon_loss,
                "lookback": lookback_loss,
                "lookahead": lookahead_loss,
                "latent": latent_pred_loss,
                "cls": cls_loss
            }
        }
        return output
