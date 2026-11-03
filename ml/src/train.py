
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

from multimodal_dataset import create_dataloaders
from models.fusion_model import MultimodalFusionModel


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "ml" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = MODEL_DIR / "best_multimodal_model.pth"
HISTORY_PATH = MODEL_DIR / "training_history.json"

BATCH_SIZE = 4
MAX_LENGTH = 256
EPOCHS = 5

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

PATIENCE = 2
NUM_CLASSES = 5

SEED = 42


# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# CLASS WEIGHTS
# --------------------------------------------------

def calculate_class_weights(train_loader, device):

    labels = []

    for batch in train_loader:
        labels.extend(batch["label"].tolist())

    labels = np.array(labels)

    classes = np.arange(NUM_CLASSES)

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    weights = torch.tensor(
        weights,
        dtype=torch.float32
    ).to(device)

    print("\nCLASS WEIGHTS")
    print("-" * 45)

    for class_id, weight in enumerate(weights):
        print(
            f"Class {class_id}: "
            f"{weight.item():.4f}"
        )

    return weights


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate(model, loader, criterion, device):

    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images
                )

                loss = criterion(logits, labels)

            total_loss += loss.item()

            predictions = torch.argmax(
                logits,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    average_loss = total_loss / len(loader)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    accuracy = np.mean(
        np.array(all_predictions) == np.array(all_labels)
    )

    return average_loss, accuracy, macro_f1


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train():

    print("=" * 65)
    print("MULTIMODAL SENTIMENT ANALYSIS - TRAINING")
    print("=" * 65)

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this training.")

    device = torch.device("cuda")

    print("\nGPU:", torch.cuda.get_device_name(0))

    # Load datasets
    print("\nLoading datasets...")

    train_loader, val_loader, _ = create_dataloaders()

    print("\nTraining batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # Initialize model
    print("\nInitializing model...")

    model = MultimodalFusionModel(
        num_classes=NUM_CLASSES
    ).to(device)

    # Calculate weights using training labels only
    class_weights = calculate_class_weights(
        train_loader,
        device
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scaler = GradScaler("cuda")

    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    history = []

    # Training loop
    for epoch in range(EPOCHS):

        print("\n" + "=" * 65)
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print("=" * 65)

        model.train()

        running_loss = 0.0

        for batch_index, batch in enumerate(train_loader):

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)

            # Forward pass with mixed precision
            with autocast(
                device_type="cuda",
                dtype=torch.float16
            ):

                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images
                )

                loss = criterion(logits, labels)

            # Backward pass
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()

            if (batch_index + 1) % 25 == 0:

                print(
                    f"Batch {batch_index + 1}/{len(train_loader)} "
                    f"| Loss: {loss.item():.4f}"
                )

        train_loss = running_loss / len(train_loader)

        # Validation
        val_loss, val_accuracy, val_macro_f1 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        print("\nEPOCH RESULTS")
        print("-" * 45)

        print(f"Training Loss:   {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Acc:  {val_accuracy:.4f}")
        print(f"Validation Macro F1: {val_macro_f1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "validation_accuracy": float(val_accuracy),
            "validation_macro_f1": float(val_macro_f1)
        })

        # Save best checkpoint
        if val_macro_f1 > best_macro_f1:

            best_macro_f1 = val_macro_f1
            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_macro_f1": best_macro_f1,
                "class_weights": class_weights.cpu(),
                "num_classes": NUM_CLASSES,
                "max_length": MAX_LENGTH
            }, CHECKPOINT_PATH)

            print("\nBest model checkpoint saved.")

        else:

            epochs_without_improvement += 1

            print(
                f"\nNo improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        # Save training history
        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(history, file, indent=4)

        # Early stopping
        if epochs_without_improvement >= PATIENCE:

            print("\nEarly stopping triggered.")
            break

    print("\n" + "=" * 65)
    print("TRAINING COMPLETED")
    print("=" * 65)

    print(f"Best Validation Macro F1: {best_macro_f1:.4f}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Training History: {HISTORY_PATH}")


if __name__ == "__main__":
    train()
