
import json
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_DIR = PROJECT_ROOT / "ml" / "evaluation"

OUTPUT_DIR = EVALUATION_DIR / "comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


REPORTS = {
    "Text-only": (
        EVALUATION_DIR /
        "expanded_text_only_test_evaluation.json"
    ),
    "Image-only": (
        EVALUATION_DIR /
        "expanded_image_only_test_evaluation.json"
    ),
    "Multimodal": (
        EVALUATION_DIR /
        "expanded_multimodal_test_evaluation.json"
    )
}


CLASS_NAMES = [
    "Very Dissatisfied",
    "Dissatisfied",
    "Neutral",
    "Satisfied",
    "Very Satisfied"
]


# --------------------------------------------------
# LOAD EVALUATION REPORTS
# --------------------------------------------------

def load_reports():

    results = {}

    print("\nLoading evaluation reports...")

    for model_name, report_path in REPORTS.items():

        if not report_path.exists():
            raise FileNotFoundError(report_path)

        with open(
            report_path,
            "r",
            encoding="utf-8"
        ) as file:

            results[model_name] = json.load(file)

        print(f"Loaded: {model_name}")

    # Verify all reports use the same test set size.
    sample_counts = {
        report["test_samples"]
        for report in results.values()
    }

    if len(sample_counts) != 1:
        raise ValueError(
            "The models were not evaluated on the same "
            "number of test samples."
        )

    return results


# --------------------------------------------------
# CREATE METRICS TABLE
# --------------------------------------------------

def create_metrics_table(results):

    metrics = [
        "Accuracy",
        "Macro F1",
        "Weighted F1",
        "MAE",
        "QWK"
    ]

    metric_keys = {
        "Accuracy": "test_accuracy",
        "Macro F1": "test_macro_f1",
        "Weighted F1": "test_weighted_f1",
        "MAE": "test_mae",
        "QWK": "test_quadratic_weighted_kappa"
    }

    rows = []

    print("\n" + "=" * 75)
    print("THREE-MODEL PERFORMANCE COMPARISON")
    print("=" * 75)

    print(
        f"{'Metric':<20}"
        f"{'Text-only':>15}"
        f"{'Image-only':>15}"
        f"{'Multimodal':>15}"
    )

    print("-" * 75)

    for metric in metrics:

        values = {}

        for model_name, report in results.items():

            value = report[metric_keys[metric]]

            values[model_name] = value

        rows.append({
            "Metric": metric,
            **values
        })

        def display(value):
            return "N/A" if value is None else f"{value:.4f}"

        print(
            f"{metric:<20}"
            f"{display(values['Text-only']):>15}"
            f"{display(values['Image-only']):>15}"
            f"{display(values['Multimodal']):>15}"
        )

    return rows


# --------------------------------------------------
# SAVE METRICS CSV
# --------------------------------------------------

def save_metrics_csv(rows):

    output_path = OUTPUT_DIR / "model_comparison.csv"

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "Metric",
                "Text-only",
                "Image-only",
                "Multimodal"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    print("\nMetrics CSV saved:", output_path)


# --------------------------------------------------
# CREATE PER-CLASS F1 COMPARISON
# --------------------------------------------------

def create_class_f1_chart(results):

    model_names = list(results.keys())

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    plt.figure(figsize=(12, 6))

    for index, model_name in enumerate(model_names):

        report = results[model_name]["classification_report"]

        f1_values = [
            report[class_name]["f1-score"]
            for class_name in CLASS_NAMES
        ]

        plt.bar(
            x + (index - 1) * width,
            f1_values,
            width,
            label=model_name
        )

    plt.xticks(
        x,
        [
            "Very Dissatisfied",
            "Dissatisfied",
            "Neutral",
            "Satisfied",
            "Very Satisfied"
        ],
        rotation=15
    )

    plt.ylabel("F1 Score")
    plt.xlabel("Sentiment Class")
    plt.title("Per-Class F1 Score Comparison")

    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "per_class_f1_comparison.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Per-class F1 chart saved:", output_path)


# --------------------------------------------------
# CREATE CONFUSION MATRICES
# --------------------------------------------------

def create_confusion_matrices(results):

    for model_name, report in results.items():

        matrix = np.array(
            report["confusion_matrix"]
        )

        fig, ax = plt.subplots(figsize=(8, 7))

        image = ax.imshow(
            matrix,
            interpolation="nearest"
        )

        ax.set_title(
            f"{model_name} - Confusion Matrix"
        )

        fig.colorbar(image, ax=ax)

        positions = np.arange(len(CLASS_NAMES))

        ax.set_xticks(positions)
        ax.set_yticks(positions)

        ax.set_xticklabels(
            CLASS_NAMES,
            rotation=35,
            ha="right"
        )

        ax.set_yticklabels(CLASS_NAMES)

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("Actual Label")

        threshold = matrix.max() / 2

        for row in range(matrix.shape[0]):

            for column in range(matrix.shape[1]):

                ax.text(
                    column,
                    row,
                    str(matrix[row, column]),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix[row, column] > threshold
                        else "black"
                    )
                )

        fig.tight_layout()

        safe_name = model_name.lower().replace("-", "_")

        output_path = (
            OUTPUT_DIR /
            f"{safe_name}_confusion_matrix.png"
        )

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print("Confusion matrix saved:", output_path)


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 75)
    print("MODEL COMPARISON AND EXPERIMENT ANALYSIS")
    print("=" * 75)

    results = load_reports()

    rows = create_metrics_table(results)

    save_metrics_csv(rows)

    create_class_f1_chart(results)

    create_confusion_matrices(results)

    print("\n" + "=" * 75)
    print("MODEL COMPARISON COMPLETED")
    print("=" * 75)

    print("Test samples:", results["Text-only"]["test_samples"])

    print("\nAll comparison files saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
