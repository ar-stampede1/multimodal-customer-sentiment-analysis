
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent

BACKEND_SOURCE = PROJECT_ROOT / "backend"

if str(BACKEND_SOURCE) not in sys.path:
    sys.path.insert(0, str(BACKEND_SOURCE))

from app.services.model_loader import load_models
from app.services.prediction_service import PredictionService


def print_result(name, result):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    for key, value in result.items():
        print(f"{key}: {value}")

    probability_sum = sum(result["probabilities"].values())

    print("Probability sum:", round(probability_sum, 6))

    assert result["predicted_class"] in range(5)
    assert 0 <= result["confidence"] <= 1
    assert abs(probability_sum - 1.0) < 1e-5


def main():

    print("LOADING MODELS")
    models = load_models()

    service = PredictionService(models)

    # --------------------------------------------------
    # TEST 1: TEXT ONLY
    # --------------------------------------------------

    text_result = service.predict(
        review_title="Excellent product",
        review_text=(
            "The product works perfectly. "
            "The quality is excellent and I am very satisfied."
        ),
    )

    print_result("TEST 1: TEXT-ONLY PREDICTION", text_result)

    assert text_result["model_used"] == "text_only"
    assert text_result["image_used"] is False

    # --------------------------------------------------
    # TEST 2: IMAGE ONLY
    # --------------------------------------------------

    image_path = (
        PROJECT_ROOT
        / "ml"
        / "data"
        / "processed"
        / "expanded_splits"
        / "train.jsonl"
    )

    import json

    with open(image_path, "r", encoding="utf-8") as file:
        first_record = json.loads(next(line for line in file if line.strip()))

    actual_image_path = PROJECT_ROOT / first_record["image_path"]

    with Image.open(actual_image_path) as img:
        test_image = img.convert("RGB")

    image_result = service.predict(image=test_image)

    print_result("TEST 2: IMAGE-ONLY PREDICTION", image_result)

    assert image_result["model_used"] == "image_only"
    assert image_result["image_used"] is True

    # --------------------------------------------------
    # TEST 3: MULTIMODAL
    # --------------------------------------------------

    multimodal_result = service.predict(
        review_title="Excellent product",
        review_text=(
            "The product works perfectly. "
            "The quality is excellent and I am very satisfied."
        ),
        image=test_image,
    )

    print_result("TEST 3: MULTIMODAL PREDICTION", multimodal_result)

    assert multimodal_result["model_used"] == "multimodal"
    assert multimodal_result["image_used"] is True

    print("\n" + "=" * 60)
    print("ALL THREE PREDICTION TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()
