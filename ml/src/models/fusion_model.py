
import torch
import torch.nn as nn

try:
    from .text_encoder import TextEncoder
    from .image_encoder import ImageEncoder
except ImportError:
    from text_encoder import TextEncoder
    from image_encoder import ImageEncoder


class MultimodalFusionModel(nn.Module):

    def __init__(
        self,
        num_classes=5,
        pretrained=True,
        freeze_encoders=False
    ):

        super().__init__()

        print("Initializing multimodal fusion model...")

        # Text encoder: DistilBERT
        self.text_encoder = TextEncoder(
            pretrained=pretrained,
            freeze_encoder=freeze_encoders
        )

        # Image encoder: ResNet18
        self.image_encoder = ImageEncoder(
            pretrained=pretrained,
            freeze_encoder=freeze_encoders
        )

        text_dim = self.text_encoder.feature_dim
        image_dim = self.image_encoder.feature_dim

        combined_dim = text_dim + image_dim

        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

        print("Multimodal fusion model initialized!")
        print("Text feature dimensions:", text_dim)
        print("Image feature dimensions:", image_dim)
        print("Combined dimensions:", combined_dim)
        print("Output classes:", num_classes)

    def forward(
        self,
        input_ids,
        attention_mask,
        images
    ):

        # Extract text features
        text_features = self.text_encoder(
            input_ids,
            attention_mask
        )

        # Extract image features
        image_features = self.image_encoder(images)

        # Concatenate both feature representations
        combined_features = torch.cat(
            [text_features, image_features],
            dim=1
        )

        # Predict five satisfaction classes
        logits = self.classifier(combined_features)

        return logits


def main():

    print("=" * 65)
    print("MULTIMODAL FUSION MODEL TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nUsing device:", device)

    model = MultimodalFusionModel(
        num_classes=5,
        pretrained=True,
        freeze_encoders=False
    ).to(device)

    model.eval()

    batch_size = 4
    sequence_length = 256

    input_ids = torch.randint(
        0,
        model.text_encoder.encoder.config.vocab_size,
        (batch_size, sequence_length),
        dtype=torch.long
    ).to(device)

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long
    ).to(device)

    images = torch.randn(
        batch_size, 3, 224, 224
    ).to(device)

    with torch.no_grad():

        logits = model(
            input_ids,
            attention_mask,
            images
        )

    print("\nMODEL OUTPUT VERIFICATION")
    print("-" * 45)

    print("Input IDs:", tuple(input_ids.shape))
    print("Attention mask:", tuple(attention_mask.shape))
    print("Images:", tuple(images.shape))
    print("Output logits:", tuple(logits.shape))

    print("Output datatype:", logits.dtype)

    print(
        "All values finite:",
        torch.isfinite(logits).all().item()
    )

    probabilities = torch.softmax(logits, dim=1)

    predictions = torch.argmax(
        probabilities,
        dim=1
    )

    print("\nPredicted classes:", predictions.tolist())

    print(
        "Probability sums:",
        probabilities.sum(dim=1).tolist()
    )

    print("\nCUDA memory allocated:")

    if torch.cuda.is_available():

        memory_mb = (
            torch.cuda.memory_allocated(device) / 1024**2
        )

        print(round(memory_mb, 2), "MB")

    print("\nMULTIMODAL FUSION TEST COMPLETED")


if __name__ == "__main__":
    main()
