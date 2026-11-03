
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(ML_SRC))

from multimodal_dataset import create_dataloaders
from models.text_only_model import TextOnlyModel


def main():

    print("=" * 60)
    print("TEXT-ONLY MODEL TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    device = torch.device("cuda")

    print("GPU:", torch.cuda.get_device_name(0))

    # Load actual review text.
    train_loader, _, _ = create_dataloaders()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)

    print("\nInput IDs:", tuple(input_ids.shape))
    print("Attention mask:", tuple(attention_mask.shape))
    print("Labels:", tuple(labels.shape))

    # Initialize model.
    model = TextOnlyModel(num_classes=5).to(device)
    model.eval()

    # Inference test.
    with torch.no_grad():

        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(logits, dim=1)

        predictions = torch.argmax(logits, dim=1)

    print("\nOUTPUT")
    print("-" * 45)

    print("Logits shape:", tuple(logits.shape))
    print("Probabilities shape:", tuple(probabilities.shape))
    print("Predictions:", predictions.tolist())
    print("Actual labels:", labels.tolist())

    print(
        "Probability sums:",
        probabilities.sum(dim=1).tolist()
    )

    if logits.shape != (4, 5):
        raise RuntimeError("Unexpected logits shape.")

    if not torch.isfinite(logits).all():
        raise RuntimeError("Non-finite logits detected.")

    if not torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(4, device=device),
        atol=1e-5
    ):
        raise RuntimeError("Invalid probability sums.")

    print("\nTEXT-ONLY MODEL TEST PASSED")


if __name__ == "__main__":
    main()
