import torch.nn as nn
import torch

from src.model.cnn.model import FMRI_CNN


class KNF_CNN_GLOBAL(nn.Module):
    def __init__(self,
                 backbone_layers,
                 backbone_hidden_channel,
                 cls_layers,
                 cls_hidden_channel,
                 dropout=0.1,
                 num_nodes=360,
                 num_classes=21
                 ):
        super(KNF_CNN_GLOBAL, self).__init__()
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
        self.attention = nn.Parameter(torch.FloatTensor(num_classes, num_nodes, num_nodes))
        nn.init.xavier_normal_(self.attention)

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
        cls_prob = nn.Softmax(dim=-1)(cls_scores)  # (batch_size, num_classes)
        K = torch.einsum('bc, cxy -> bxy', cls_prob, self.attention)             # (batch_size, num_nodes, num_nodes)
        lookback_predictions = self.forward_prediction(z[:, :, :1], K, iter=z.shape[-1] - 1)
        lookahead_predictions = self.forward_prediction(z[:, :, -1:], K, iter=y.shape[-1])

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
