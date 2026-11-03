
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

REPORT_PATH = (
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


def inspect_errors():

    with open(REPORT_PATH, "r", encoding="utf-8") as file:
        report = json.load(file)

    errors = report["individual_errors"]

    # Show errors with highest prediction confidence first.
    errors.sort(
        key=lambda item: item["confidence"],
        reverse=True
    )

    print("=" * 70)
    print("MISCLASSIFIED REVIEW INSPECTION")
    print("=" * 70)

    print("Total incorrect predictions:", len(errors))

    for index, error in enumerate(errors[:10], start=1):

        print("\n" + "=" * 70)
        print(f"ERROR {index}")
        print("=" * 70)

        print("Review ID:", error["review_id"])
        print("Product ID:", error["product_id"])

        print("\nReview title:")
        print(error["review_title"])

        print("\nActual rating:")
        print(error["actual_rating"], "-", error["actual_label"])

        print("\nPredicted rating:")
        print(error["predicted_label"])

        print(
            "Prediction confidence:",
            f"{error['confidence'] * 100:.2f}%"
        )

        print("\nPrediction probabilities:")

        for class_id, probability in enumerate(
            error["prediction_probabilities"]
        ):

            print(
                f"{class_id} - {CLASS_NAMES[class_id]}: "
                f"{probability * 100:.2f}%"
            )

        print("\nReview text:")
        print(error["review_text"])

        print("\nImage path:")
        print(error["image_path"])

        print("\nImage URL:")
        print(error["image_url"])


if __name__ == "__main__":
    inspect_errors()
