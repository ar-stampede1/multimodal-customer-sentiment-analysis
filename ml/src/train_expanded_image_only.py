
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score

from models.image_only_model import ImageOnlyModel
from multimodal_dataset import create_dataloaders


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT / "ml" / "saved_models" / "expanded"
)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 2
NUM_CLASSES = 5

MODEL_PATH = (
    MODEL_DIR / "best_image_only_model_expanded.pth"
)

HISTORY_PATH = (
    MODEL_DIR / "image_only_training_history_expanded.json"
)


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

    counts = np.bincount(
        labels,
        minlength=NUM_CLASSES
    )

    if np.any(counts == 0):
        raise ValueError(
            "At least one class is missing from the training data."
        )

    weights = len(labels) / (NUM_CLASSES * counts)

    print("\nTRAINING CLASS DISTRIBUTION")
    print("-" * 45)

    for class_id, count in enumerate(counts):
        print(
            f"Class {class_id}: {count} "
            f"| Weight: {weights[class_id]:.4f}"
        )

    return torch.tensor(
        weights,
        dtype=torch.float32
    ).to(device)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device
):

    model.train()

    total_loss = 0.0

    all_predictions = []
    all_labels = []

    for batch_index, batch in enumerate(loader):

        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(device.type == "cuda")
        ):

            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        all_labels.extend(
            labels.cpu().tolist()
        )

        if (batch_index + 1) % 100 == 0:

            print(
                f"Batch {batch_index + 1}/"
                f"{len(loader)} "
                f"| Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    return average_loss, accuracy, macro_f1


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

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda")
            ):

                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item()

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    average_loss = total_loss / len(loader)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        average="macro",
        zero_division=0
    )

    return average_loss, accuracy, macro_f1


# --------------------------------------------------
# MAIN TRAINING FUNCTION
# --------------------------------------------------

def main():

    print("=" * 65)
    print("EXPANDED IMAGE-ONLY SENTIMENT MODEL TRAINING")
    print("=" * 65)

    set_seed(SEED)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    device = torch.device("cuda")

    print("\nGPU:", torch.cuda.get_device_name(0))

    # Load expanded datasets
    print("\nLoading expanded datasets...")

    train_loader, val_loader, _ = create_dataloaders(
        dataset_version="expanded"
    )

    print("\nTraining records:", len(train_loader.dataset))
    print("Validation records:", len(val_loader.dataset))
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # Initialize image-only model
    print("\nInitializing ResNet18 image-only model...")

    model = ImageOnlyModel(
        num_classes=NUM_CLASSES,
        freeze_encoder=False
    ).to(device)

    # Calculate class weights
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

    scaler = GradScaler(
        "cuda",
        enabled=(device.type == "cuda")
    )

    best_val_f1 = -1.0
    patience_counter = 0

    history = []

    print("\nStarting expanded image-only training...")

    # Epoch loop
    for epoch in range(EPOCHS):

        print("\n" + "=" * 65)
        print(f"EPOCH {epoch + 1}/{EPOCHS}")
        print("=" * 65)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device
        )

        val_loss, val_acc, val_f1 = validate(
            model,
            val_loader,
            criterion,
            device
        )

        epoch_result = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "train_macro_f1": float(train_f1),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_acc),
            "val_macro_f1": float(val_f1)
        }

        history.append(epoch_result)

        print("\nEPOCH RESULTS")
        print("-" * 45)

        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_acc:.4f}")
        print(f"Training Macro F1: {train_f1:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Macro F1: {val_f1:.4f}")

        # Save best checkpoint
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_macro_f1": best_val_f1,
                    "class_weights": class_weights.cpu(),
                    "num_classes": NUM_CLASSES
                },
                MODEL_PATH
            )

            print("\nBest expanded image-only checkpoint saved!")

        else:

            patience_counter += 1

            print(
                f"\nNo validation improvement: "
                f"{patience_counter}/{PATIENCE}"
            )

        # Save training history
        with open(
            HISTORY_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(history, file, indent=4)

        # Early stopping
        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")
            break

    print("\n" + "=" * 65)
    print("EXPANDED IMAGE-ONLY TRAINING COMPLETED")
    print("=" * 65)

    print(
        "Best Validation Macro F1:",
        round(best_val_f1, 4)
    )

    print("Best model:", MODEL_PATH)
    print("Training history:", HISTORY_PATH)


if __name__ == "__main__":
    main()
