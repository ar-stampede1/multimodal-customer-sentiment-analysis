
import json
from pathlib import Path

import torch
from PIL import Image, UnidentifiedImageError
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "splits"
)

IMAGE_SIZE = 224

# ImageNet normalization used with pretrained ResNet18
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def preprocess_image(image_path):
    """
    Load and preprocess one customer review image.
    """

    with Image.open(image_path) as img:

        img = img.convert("RGB")

        tensor = IMAGE_TRANSFORM(img)

    return tensor


def main():

    print("=" * 65)
    print("RESNET18 IMAGE PREPROCESSING")
    print("=" * 65)

    train_file = SPLIT_DIR / "train.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(train_file)

    with open(train_file, "r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file if line.strip()
        ]

    print("\nTraining records loaded:", len(records))

    print("\nIMAGE PREPROCESSING TEST")
    print("-" * 45)

    successful = 0
    failed = 0

    for index, record in enumerate(records[:100]):

        relative_path = record.get("image_path")

        if not relative_path:
            failed += 1
            continue

        image_path = PROJECT_ROOT / relative_path

        try:

            if not image_path.exists():
                raise FileNotFoundError(image_path)

            image_tensor = preprocess_image(image_path)

            if image_tensor.shape != (3, 224, 224):
                raise ValueError(
                    f"Unexpected tensor shape: {image_tensor.shape}"
                )

            if not torch.isfinite(image_tensor).all():
                raise ValueError("Non-finite values detected")

            successful += 1

            if index == 0:

                print("\nExample image:")
                print("Image path:", relative_path)
                print("Tensor shape:", tuple(image_tensor.shape))
                print("Tensor dtype:", image_tensor.dtype)
                print(
                    "Tensor minimum:",
                    round(image_tensor.min().item(), 4)
                )
                print(
                    "Tensor maximum:",
                    round(image_tensor.max().item(), 4)
                )

        except (
            OSError,
            ValueError,
            UnidentifiedImageError
        ) as error:

            failed += 1

            print(f"\nImage {index} failed: {error}")

    print("\nPREPROCESSING SUMMARY")
    print("-" * 45)

    print("Images tested:", min(100, len(records)))
    print("Successful:", successful)
    print("Failed:", failed)

    print("Target image size:", f"{IMAGE_SIZE} x {IMAGE_SIZE}")

    print("\nIMAGE PREPROCESSING TEST COMPLETED")


if __name__ == "__main__":
    main()
