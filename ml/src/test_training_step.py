
from pathlib import Path
import sys

import torch
import torch.nn as nn

# Add the project directories to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_SRC = PROJECT_ROOT / "ml" / "src"

sys.path.insert(0, str(ML_SRC))

from multimodal_dataset import create_dataloaders
from models.fusion_model import MultimodalFusionModel


def main():
    print("=" * 60)
    print("MULTIMODAL TRAINING-STEP SMOKE TEST")
    print("=" * 60)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. GPU training cannot be tested.")

    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load one real batch from the training dataset.
    print("\nLoading dataset...")

    train_loader, _, _ = create_dataloaders()

    batch = next(iter(train_loader))

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    print(f"Input IDs: {tuple(input_ids.shape)}")
    print(f"Attention mask: {tuple(attention_mask.shape)}")
    print(f"Images: {tuple(images.shape)}")
    print(f"Labels: {tuple(labels.shape)}")

    # Initialize the multimodal model.
    print("\nInitializing fusion model...")

    model = MultimodalFusionModel(num_classes=5).to(device)
    model.train()

    # Loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    optimizer.zero_grad(set_to_none=True)

    # Forward pass.
    print("\nRunning forward pass...")

    logits = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images
    )

    loss = criterion(logits, labels)

    if not torch.isfinite(loss):
        raise RuntimeError(f"Invalid loss: {loss.item()}")

    print(f"Logits shape: {tuple(logits.shape)}")
    print(f"Loss: {loss.item():.4f}")

    # Backward pass.
    print("\nRunning backward pass...")

    loss.backward()

    gradients_found = any(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if not gradients_found:
        raise RuntimeError("No valid gradients found.")

    optimizer.step()

    print("Backward pass: SUCCESS")
    print("Optimizer step: SUCCESS")

    # GPU memory report.
    torch.cuda.synchronize()

    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 3)

    print("\nGPU MEMORY")
    print(f"Current allocated: {allocated:.2f} GB")
    print(f"Peak allocated: {peak:.2f} GB")

    print("\n" + "=" * 60)
    print("TRAINING-STEP SMOKE TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
