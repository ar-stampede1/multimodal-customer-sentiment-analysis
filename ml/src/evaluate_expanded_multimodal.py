
import json
from pathlib import Path

import numpy as np
import torch
from torch.amp import autocast
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    cohen_kappa_score
)

from multimodal_dataset import create_dataloaders
from models.fusion_model import MultimodalFusionModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "ml" / "saved_models" / "expanded" /
    "best_multimodal_model_expanded.pth"
)

REPORT_PATH = (
    PROJECT_ROOT / "ml" / "evaluation" /
    "expanded_multimodal_test_evaluation.json"
)

CLASS_NAMES = [
    "Very Dissatisfied",
    "Dissatisfied",
    "Neutral",
    "Satisfied",
    "Very Satisfied"
]

NUM_CLASSES = 5


def evaluate():

    print("=" * 65)
    print("EXPANDED MULTIMODAL MODEL - TEST EVALUATION")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    device = torch.device("cuda")

    print("\nGPU:", torch.cuda.get_device_name(0))

    # Load expanded test dataset
    print("\nLoading expanded test dataset...")

    _, _, test_loader = create_dataloaders(
        dataset_version="expanded"
    )

    print("Test records:", len(test_loader.dataset))

    # Load trained model
    print("\nLoading expanded model checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model = MultimodalFusionModel(
        num_classes=NUM_CLASSES
    ).to(device)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Best checkpoint epoch:", checkpoint["epoch"])
    print(
        "Best validation macro F1:",
        round(checkpoint["best_validation_macro_f1"], 4)
    )

    # Generate predictions
    all_predictions = []
    all_labels = []

    print("\nRunning predictions...")

    with torch.no_grad():

        for batch_index, batch in enumerate(test_loader):

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

            predictions = torch.argmax(logits, dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

            if (batch_index + 1) % 50 == 0:
                print(
                    f"Processed {batch_index + 1}/"
                    f"{len(test_loader)} batches"
                )

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    mae = mean_absolute_error(y_true, y_pred)

    qwk = cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic"
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    # Display results
    print("\n" + "=" * 65)
    print("EXPANDED MULTIMODAL TEST RESULTS")
    print("=" * 65)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")
    print(f"MAE:         {mae:.4f}")
    print(f"QWK:         {qwk:.4f}")

    print("\nPER-CLASS PERFORMANCE")
    print("-" * 65)

    for class_name in CLASS_NAMES:

        metrics = report[class_name]

        print(f"\n{class_name}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-score:  {metrics['f1-score']:.4f}")
        print(f"Support:   {metrics['support']}")

    print("\nCONFUSION MATRIX")
    print("-" * 65)
    print(matrix)

    # Save evaluation report
    results = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(
            checkpoint["best_validation_macro_f1"]
        ),
        "test_accuracy": float(accuracy),
        "test_macro_f1": float(macro_f1),
        "test_weighted_f1": float(weighted_f1),
        "test_mae": float(mae),
        "test_quadratic_weighted_kappa": (
            float(qwk) if np.isfinite(qwk) else None
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": CLASS_NAMES,
        "test_samples": int(len(y_true))
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print("\nReport saved to:")
    print(REPORT_PATH)

    print("\nEXPANDED MULTIMODAL TEST EVALUATION COMPLETED")


if __name__ == "__main__":
    evaluate()
