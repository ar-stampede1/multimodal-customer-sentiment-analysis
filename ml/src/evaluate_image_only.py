import json
import os
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    cohen_kappa_score
)

from models.image_only_model import ImageOnlyModel
from multimodal_dataset import create_dataloaders


CLASS_NAMES = [
    "Very Dissatisfied",
    "Dissatisfied",
    "Neutral",
    "Satisfied",
    "Very Satisfied"
]

MODEL_PATH = "ml/saved_models/best_image_only_model.pth"
REPORT_PATH = "ml/evaluation/image_only_test_evaluation.json"


def main():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 65)
    print("IMAGE-ONLY MODEL TEST EVALUATION")
    print("=" * 65)

    print("Device:", device)

    # Load the existing test split.
    _, _, test_loader = create_dataloaders()

    # Load the saved best model.
    model = ImageOnlyModel(num_classes=5).to(device)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint from epoch:", checkpoint["epoch"])
    print("Best validation macro F1:", checkpoint["best_val_macro_f1"])

    all_predictions = []
    all_labels = []
    all_probabilities = []

    print("\nEvaluating test dataset...")

    with torch.no_grad():

        for batch in test_loader:

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_probabilities.extend(probabilities.cpu().tolist())

    # Calculate evaluation metrics.
    accuracy = accuracy_score(all_labels, all_predictions)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
    )

    mae = mean_absolute_error(all_labels, all_predictions)

    qwk = cohen_kappa_score(
        all_labels,
        all_predictions,
        weights="quadratic"
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4]
    )

    # Display results.
    print("\n" + "=" * 65)
    print("TEST RESULTS")
    print("=" * 65)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"MAE:         {mae:.4f}")
    print(f"QWK:         {qwk:.4f}")

    print("\nCLASSIFICATION REPORT")
    print("-" * 65)

    print(
        classification_report(
            all_labels,
            all_predictions,
            labels=[0, 1, 2, 3, 4],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("CONFUSION MATRIX")
    print(matrix)

    # Save evaluation report.
    results = {
        "model": "Image-only ResNet18",
        "checkpoint_epoch": checkpoint["epoch"],
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mae": mae,
        "quadratic_weighted_kappa": qwk,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "total_test_samples": len(all_labels)
    }

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)

    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nEvaluation report saved to:", REPORT_PATH)


if __name__ == "__main__":
    main()