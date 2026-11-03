
import json
import csv
from pathlib import Path
from datetime import datetime

# Project paths
ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = ROOT / "ml" / "evaluation"
COMPARISON_DIR = EVALUATION_DIR / "comparison"

REPORT_FILES = {
    "Text-only": EVALUATION_DIR / "expanded_text_only_test_evaluation.json",
    "Image-only": EVALUATION_DIR / "expanded_image_only_test_evaluation.json",
    "Multimodal": EVALUATION_DIR / "expanded_multimodal_test_evaluation.json",
}

COMPARISON_CSV = COMPARISON_DIR / "model_comparison.csv"
OUTPUT_FILE = EVALUATION_DIR / "FINAL_MODEL_EVALUATION.md"


def load_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Required report not found: {path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def percentage(value):
    return f"{value * 100:.2f}%"


def metric(data, *keys):
    """Read a metric using possible key names."""
    for key in keys:
        if key in data:
            return data[key]

    raise KeyError(f"None of these metric keys were found: {keys}")


def load_comparison_csv():
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {COMPARISON_CSV}")

    with open(COMPARISON_CSV, "r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def build_report():
    reports = {
        name: load_json(path)
        for name, path in REPORT_FILES.items()
    }

    comparison_rows = load_comparison_csv()

    lines = []

    def add(text=""):
        lines.append(str(text))

    add("# Multimodal Customer Sentiment Analysis System")
    add()
    add("## Final Model Evaluation Report")
    add()
    add(f"**Generated:** {datetime.now().strftime('%d %B %Y, %I:%M %p')}")
    add()
    add("---")
    add()

    # Dataset
    add("## 1. Dataset Overview")
    add()
    add("- Dataset: Amazon Reviews 2023 – Electronics")
    add("- Original expanded collection: 10,000 image-review pairs")
    add("- Clean image-review pairs: 9,105")
    add("- Training samples: 7,283")
    add("- Validation samples: 911")
    add("- Test samples: 911")
    add("- Split strategy: Product-aware stratified group splitting")
    add("- Product overlap between splits: 0")
    add("- Image hash overlap between splits: 0")
    add()
    add("### Sentiment Classes")
    add()
    add("| Label | Class |")
    add("|---:|---|")
    for i, label in enumerate([
        "Very Dissatisfied",
        "Dissatisfied",
        "Neutral",
        "Satisfied",
        "Very Satisfied",
    ]):
        add(f"| {i} | {label} |")
    add()

    # Architecture
    add("## 2. Model Architectures")
    add()
    add("| Model | Architecture |")
    add("|---|---|")
    add("| Text-only | DistilBERT text encoder + classifier |")
    add("| Image-only | ResNet18 image encoder + classifier |")
    add("| Multimodal | DistilBERT + ResNet18 + feature fusion classifier |")
    add()
    add("The multimodal model concatenates the 768-dimensional text representation")
    add("and 512-dimensional image representation into a 1,280-dimensional feature")
    add("vector before classification into five sentiment classes.")
    add()

    # Results
    add("## 3. Test Set Performance")
    add()
    add("| Metric | Text-only | Image-only | Multimodal |")
    add("|---|---:|---:|---:|")

    metric_rows = [
        ("Accuracy", ("test_accuracy",)),
        ("Macro F1", ("test_macro_f1",)),
        ("Weighted F1", ("test_weighted_f1",)),
        ("Mean Absolute Error", ("test_mae",)),
        ("Quadratic Weighted Kappa", ("test_quadratic_weighted_kappa",)),
    ]

    for label, keys in metric_rows:
        values = []
        for model in ["Text-only", "Image-only", "Multimodal"]:
            value = metric(reports[model], *keys)
            values.append(f"{value:.4f}")
        add(f"| {label} | " + " | ".join(values) + " |")

    add()
    add("### Interpretation")
    add()
    add("- Text-only achieved the highest accuracy, weighted F1, and QWK.")
    add("- Multimodal achieved the highest Macro F1.")
    add("- Image-only performed substantially worse than the other models.")
    add("- Text appears to be the strongest individual modality in these experiments.")
    add("- Fusion provided a modest Macro F1 improvement, but did not improve every metric.")
    add()

    # Per-class performance
    add("## 4. Per-Class F1 Comparison")
    add()
    add("| Sentiment Class | Text-only | Image-only | Multimodal |")
    add("|---|---:|---:|---:|")

    class_names = [
        "Very Dissatisfied",
        "Dissatisfied",
        "Neutral",
        "Satisfied",
        "Very Satisfied",
    ]

    for index, class_name in enumerate(class_names):
        values = []
        for model in ["Text-only", "Image-only", "Multimodal"]:
            report = reports[model]
            class_report = report.get("classification_report", report.get("report", {}))

            # Support common sklearn report structures.
            if class_name in class_report:
                class_data = class_report[class_name]
            elif str(index) in class_report:
                class_data = class_report[str(index)]
            else:
                raise KeyError(
                    f"Could not find class '{class_name}' in {model} report. "
                    "Inspect the classification_report structure."
                )

            values.append(f"{class_data['f1-score']:.4f}")

        add(f"| {class_name} | " + " | ".join(values) + " |")

    add()
    add("## 5. Training and Validation Analysis")
    add()
    add("Best validation Macro F1 scores:")
    add()
    add("| Model | Best Epoch | Best Validation Macro F1 |")
    add("|---|---:|---:|")
    history_files = {
        "Text-only": ROOT / "ml/saved_models/expanded/text_only_training_history_expanded.json",
        "Image-only": ROOT / "ml/saved_models/expanded/image_only_training_history_expanded.json",
        "Multimodal": ROOT / "ml/saved_models/expanded/multimodal_training_history_expanded.json",
    }

    for model, path in history_files.items():
        history = load_json(path)

        if not isinstance(history, list) or not history:
            raise ValueError(f"Invalid training history format: {path}")

        # Support both validation key naming conventions.
        f1_key = (
            "validation_macro_f1"
            if "validation_macro_f1" in history[0]
            else "val_macro_f1"
        )

        best_epoch = max(history, key=lambda record: record[f1_key])

        add(
            f"| {model} | {best_epoch['epoch']} | "
            f"{best_epoch[f1_key]:.4f} |"
        )


    add()
    add("The validation loss and Macro F1 plots are available in the comparison folder.")
    add()

    # Confusion matrices
    add("## 6. Confusion Matrices")
    add()
    add("### Text-only")
    add()
    add("![Text-only confusion matrix](comparison/text_only_confusion_matrix.png)")
    add()
    add("### Image-only")
    add()
    add("![Image-only confusion matrix](comparison/image_only_confusion_matrix.png)")
    add()
    add("### Multimodal")
    add()
    add("![Multimodal confusion matrix](comparison/multimodal_confusion_matrix.png)")
    add()

    # Graphs
    add("## 7. Training Graphs")
    add()
    add("### Validation Loss")
    add()
    add("![Validation loss](comparison/training_validation_loss.png)")
    add()
    add("### Validation Accuracy")
    add()
    add("![Validation accuracy](comparison/validation_accuracy_comparison.png)")
    add()
    add("### Validation Macro F1")
    add()
    add("![Validation Macro F1](comparison/validation_macro_f1_comparison.png)")
    add()
    add("### Per-Class F1")
    add()
    add("![Per-class F1](comparison/per_class_f1_comparison.png)")
    add()

    # Limitations
    add("## 8. Limitations")
    add()
    add("1. The dataset has substantial class imbalance, with Very Satisfied")
    add("   representing approximately 67.7% of the clean dataset.")
    add("2. The image-only model struggled to distinguish minority sentiment classes.")
    add("3. The multimodal and text-only models showed signs of overfitting.")
    add("4. Image sentiment is not always directly related to review sentiment.")
    add("5. Results are based on one product-aware test split and may vary across")
    add("   different datasets and product categories.")
    add()

    # Conclusion
    add("## 9. Conclusion")
    add()
    add("The experiments demonstrate that review text is the strongest individual")
    add("modality for this dataset. Combining text and image features produced the")
    add("highest Macro F1 score, indicating a modest benefit for performance across")
    add("the five sentiment classes. However, the text-only model achieved better")
    add("accuracy and ordinal agreement. The results support retaining multimodal")
    add("fusion as the main project architecture while using text-only and image-only")
    add("models as comparison baselines.")
    add()
    add("---")
    add()
    add("*This report was generated from the project's saved evaluation reports and")
    add("training histories.*")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(lines), encoding="utf-8")

    print("=" * 70)
    print("FINAL EVALUATION REPORT GENERATED")
    print("=" * 70)
    print(f"Output: {OUTPUT_FILE}")
    print(f"Models included: {len(reports)}")
    print(f"Comparison rows loaded: {len(comparison_rows)}")
    print(f"Report size: {OUTPUT_FILE.stat().st_size:,} bytes")


if __name__ == "__main__":
    build_report()
