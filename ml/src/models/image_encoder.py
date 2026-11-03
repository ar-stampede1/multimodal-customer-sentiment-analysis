
import torch
import torch.nn as nn

from torchvision.models import resnet18, ResNet18_Weights


class ImageEncoder(nn.Module):

    def __init__(self, pretrained=True, freeze_encoder=False):

        super().__init__()

        print("Loading ResNet18 image encoder...")

        weights = (
            ResNet18_Weights.DEFAULT
            if pretrained else None
        )

        self.encoder = resnet18(weights=weights)

        # Remove the original ImageNet classification layer
        self.encoder.fc = nn.Identity()

        self.feature_dim = 512

        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        print("ResNet18 loaded successfully!")
        print("Feature dimensions:", self.feature_dim)

    def forward(self, images):

        features = self.encoder(images)

        return features


def main():

    print("=" * 65)
    print("RESNET18 IMAGE ENCODER TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nUsing device:", device)

    model = ImageEncoder(
        pretrained=True,
        freeze_encoder=False
    ).to(device)

    model.eval()

    # Simulated batch of four preprocessed images
    images = torch.randn(
        4, 3, 224, 224
    ).to(device)

    with torch.no_grad():

        features = model(images)

    print("\nOUTPUT VERIFICATION")
    print("-" * 45)

    print("Input image shape:", (4, 3, 224, 224))
    print("Output feature shape:", tuple(features.shape))
    print("Output datatype:", features.dtype)

    print(
        "All values finite:",
        torch.isfinite(features).all().item()
    )

    print("\nRESNET18 ENCODER TEST COMPLETED")


if __name__ == "__main__":
    main()
