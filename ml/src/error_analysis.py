
import json
from pathlib import Path

import torch
from torch.amp import autocast
from sklearn.metrics import confusion_matrix

from multimodal_dataset import create_dataloaders
from models.fusion_model import MultimodalFusionModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "ml" / "saved_models" /
    "best_multimodal_model.pth"
)

TEST_MANIFEST = (
    PROJECT_ROOT / "ml" / "data" / "processed" /
    "splits" / "test.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT / "ml" / "evaluation" /
    "error_analysis_detailed.json"
)

CLASS_NAMES = [
    "Very Dissatisfied",
    "Dissatisfied",
    "Neutral",
    "Satisfied",
    "Very Satisfied"
]


def load_test_records():

    with open(
        TEST_MANIFEST,
        "r",
        encoding="utf-8"
    ) as file:

        return [
            json.loads(line)
            for line in file if line.strip()
        ]


def analyze_errors():

    print("=" * 65)
    print("DETAILED MULTIMODAL ERROR ANALYSIS")
    print("=" * 65)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required.")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    device = torch.device("cuda")

    # Load original test records.
    records = load_test_records()

    # Load test dataset.
    _, _, test_loader = create_dataloaders()

    if len(records) != len(test_loader.dataset):
        raise ValueError(
            "Manifest and test dataset record counts do not match."
        )

    # Load trained model.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model = MultimodalFusionModel(num_classes=5).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("\nCheckpoint epoch:", checkpoint["epoch"])
    print("Test records:", len(records))

    all_predictions = []
    all_labels = []
    all_errors = []

    record_index = 0

    # Generate predictions.
    with torch.no_grad():

        for batch in test_loader:

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

            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            for i in range(len(labels)):

                actual = int(labels[i].item())
                predicted = int(predictions[i].item())

                record = records[record_index]
                record_index += 1

                all_labels.append(actual)
                all_predictions.append(predicted)

                if actual != predicted:

                    all_errors.append({

                        "review_id": record.get("review_id"),

                        "product_id": record.get("product_id"),

                        "parent_asin": record.get("parent_asin"),

                        "review_title": record.get("review_title"),

                        "review_text": record.get("review_text"),

                        "actual_rating": record.get("rating"),

                        "actual_class": actual,

                        "actual_label": CLASS_NAMES[actual],

                        "predicted_class": predicted,

                        "predicted_label": CLASS_NAMES[predicted],

                        "confidence": float(
                            probabilities[i][predicted].item()
                        ),

                        "prediction_probabilities": [
                            float(value)
                            for value in probabilities[i].cpu().tolist()
                        ],

                        "image_path": record.get("image_path"),

                        "image_url": record.get("image_url")
                    })

    if record_index != len(records):
        raise ValueError(
            "Not all test records were processed."
        )

    # Calculate confusion matrix.
    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(5))
    )

    correct = len(all_labels) - len(all_errors)

    print("\nTOTAL RECORDS:", len(all_labels))
    print("CORRECT:", correct)
    print("INCORRECT:", len(all_errors))

    print("\nERRORS BY ACTUAL CLASS")
    print("-" * 65)

    error_counts = {}

    for class_id, class_name in enumerate(CLASS_NAMES):

        count = sum(
            1 for error in all_errors
            if error["actual_class"] == class_id
        )

        error_counts[class_name] = count

        print(f"{class_name}: {count}")

    # Most common mistakes.
    mistake_counts = {}

    for error in all_errors:

        key = (
            error["actual_label"],
            error["predicted_label"]
        )

        mistake_counts[key] = (
            mistake_counts.get(key, 0) + 1
        )

    sorted_mistakes = sorted(
        mistake_counts.items(),
        key=lambda item: item[1],
        reverse=True
    )

    print("\nMOST COMMON MISCLASSIFICATIONS")
    print("-" * 65)

    for (actual, predicted), count in sorted_mistakes:

        print(f"{actual} -> {predicted}: {count}")

    # Save detailed report.
    results = {

        "checkpoint_epoch": int(checkpoint["epoch"]),

        "total_test_records": len(all_labels),

        "correct_predictions": correct,

        "incorrect_predictions": len(all_errors),

        "confusion_matrix": matrix.tolist(),

        "errors_by_actual_class": error_counts,

        "common_misclassifications": [
            {
                "actual": actual,
                "predicted": predicted,
                "count": count
            }
            for (actual, predicted), count in sorted_mistakes
        ],

        "individual_errors": all_errors
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nDetailed report saved to:")
    print(OUTPUT_PATH)

    print("\nDETAILED ERROR ANALYSIS COMPLETED")


if __name__ == "__main__":
    analyze_errors()
