import torch
import torch.nn as nn

try:
    from .image_encoder import ImageEncoder
except ImportError:
    from image_encoder import ImageEncoder


class ImageOnlyModel(nn.Module):
    def __init__(self, num_classes=5, freeze_encoder=False):
        super().__init__()

        # Extract visual features using pretrained ResNet18
        self.image_encoder = ImageEncoder(
            pretrained=True,
            freeze_encoder=freeze_encoder
        )

        # ResNet18 produces 512-dimensional features
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, images):
        # Extract image features
        image_features = self.image_encoder(images)

        # Predict one of five sentiment classes
        logits = self.classifier(image_features)

        return logits