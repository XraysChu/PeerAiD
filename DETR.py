import torch
from torchvision.models import resnet50

import torch.nn as nn
import torch.nn.functional as F

class DETR(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, nheads=8, num_encoder_layers=6, num_decoder_layers=6):
        super(DETR, self).__init__()

        # Backbone
        self.backbone = resnet50(pretrained=True)
        backbone_out_channels = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # Transformer
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers
        )

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.zeros(100, hidden_dim))

        # Class and bounding box heads
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.bbox_embed = nn.Linear(hidden_dim, 4)

    def forward(self, inputs):
        # Backbone feature extraction
        features = self.backbone(inputs)

        # Positional encoding
        positional_encoding = self.positional_encoding.unsqueeze(1).repeat(1, features.shape[0], 1)
        features = features + positional_encoding

        # Transformer encoding
        encodings = self.transformer(features)

        # Class and bounding box predictions
        class_predictions = self.class_embed(encodings)
        bbox_predictions = self.bbox_embed(encodings)

        return class_predictions, bbox_predictions