import torch

from models.image_only_model import ImageOnlyModel

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)

    model = ImageOnlyModel(num_classes=5).to(device)
    model.eval()

    # Simulate a batch of 4 preprocessed images
    images = torch.randn(4, 3, 224, 224).to(device)

    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)

    print("Logits shape:", logits.shape)
    print("Probabilities shape:", probabilities.shape)
    print("Probability sums:", probabilities.sum(dim=1))
    print("All outputs finite:", torch.isfinite(logits).all().item())

    print("\nImage-only model test completed successfully!")


if __name__ == "__main__":
    main()