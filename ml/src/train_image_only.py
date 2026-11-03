import json
import os
import random
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

SEED = 42
EPOCHS = 5
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
PATIENCE = 2
NUM_CLASSES = 5

MODEL_PATH = "ml/saved_models/best_image_only_model.pth"
HISTORY_PATH = "ml/saved_models/image_only_training_history.json"


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

    counts = np.bincount(labels, minlength=NUM_CLASSES)

    weights = len(labels) / (NUM_CLASSES * counts)

    print("\nTraining class distribution:")
    for class_id, count in enumerate(counts):
        print(f"Class {class_id}: {count}")

    return torch.tensor(weights, dtype=torch.float32).to(device)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()

    total_loss = 0
    all_predictions = []
    all_labels = []

    for batch in loader:
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

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(predictions.detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return average_loss, accuracy, macro_f1


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0
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

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    average_loss = total_loss / len(loader)
    accuracy = accuracy_score(all_labels, all_predictions)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return average_loss, accuracy, macro_f1


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("IMAGE-ONLY SENTIMENT MODEL TRAINING")
    print("=" * 65)

    print("Device:", device)

    # Load existing train, validation and test splits.
    train_loader, val_loader, _ = create_dataloaders()

    print("\nDataset loaded:")
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))

    # Initialize model.
    model = ImageOnlyModel(
        num_classes=NUM_CLASSES,
        freeze_encoder=False
    ).to(device)

    # Handle class imbalance.
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

    best_val_f1 = 0.0
    patience_counter = 0

    history = []

    print("\nStarting training...")

    for epoch in range(EPOCHS):

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 50)

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
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1
        }

        history.append(epoch_result)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Train Macro F1: {train_f1:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        print(f"Validation Macro F1: {val_f1:.4f}")

        # Save the best checkpoint using validation macro F1.
        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "best_val_macro_f1": best_val_f1
                },
                MODEL_PATH
            )

            print("Best model checkpoint saved!")

        else:
            patience_counter += 1
            print("No validation improvement.")

        if patience_counter >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # Save training history.
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)

    with open(HISTORY_PATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    print("\n" + "=" * 65)
    print("IMAGE-ONLY TRAINING COMPLETED")
    print("=" * 65)

    print("Best Validation Macro F1:", round(best_val_f1, 4))
    print("Best model:", MODEL_PATH)
    print("Training history:", HISTORY_PATH)


if __name__ == "__main__":
    main()