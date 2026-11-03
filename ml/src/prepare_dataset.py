
import json
import hashlib
from pathlib import Path
from collections import defaultdict, Counter

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "electronics_1000_with_images.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "electronics_clean.jsonl"

REPORT_FILE = (
    PROJECT_ROOT / "ml" / "evaluation" /
    "dataset_preparation_report.json"
)


def calculate_hash(file_path):
    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def is_valid_image(image_path):
    try:
        with Image.open(image_path) as img:
            img.verify()

        with Image.open(image_path) as img:
            return (
                img.format == "JPEG"
                and img.mode == "RGB"
                and img.width >= 32
                and img.height >= 32
            )

    except (OSError, ValueError, UnidentifiedImageError):
        return False


def main():

    print("=" * 65)
    print("MULTIMODAL DATASET PREPARATION")
    print("=" * 65)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file if line.strip()
        ]

    print("\nRecords loaded:", len(records))

    # Track filtering results
    missing_images = 0
    invalid_images = 0
    duplicate_images = 0

    seen_hashes = {}
    clean_records = []

    duplicate_details = []
    rating_counts = Counter()

    for record in records:

        relative_path = record.get("image_path")

        if not relative_path:
            missing_images += 1
            continue

        image_path = PROJECT_ROOT / relative_path

        if not image_path.exists():
            missing_images += 1
            continue

        if not is_valid_image(image_path):
            invalid_images += 1
            continue

        image_hash = calculate_hash(image_path)

        # Exclude exact duplicate image contents
        if image_hash in seen_hashes:

            duplicate_images += 1

            original = seen_hashes[image_hash]

            duplicate_details.append({
                "image_hash": image_hash,
                "kept_review_id": original["review_id"],
                "excluded_review_id": record["review_id"],
                "kept_product": original["parent_asin"],
                "excluded_product": record["parent_asin"],
                "kept_rating": original["rating"],
                "excluded_rating": record["rating"]
            })

            continue

        seen_hashes[image_hash] = {
            "review_id": record["review_id"],
            "parent_asin": record["parent_asin"],
            "rating": record["rating"]
        }

        record["image_hash"] = image_hash

        clean_records.append(record)

        rating_counts[record["rating"]] += 1

    # Save clean manifest
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for record in clean_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Calculate product statistics
    product_counts = Counter(
        record["parent_asin"]
        for record in clean_records
    )

    # Generate report
    report = {
        "original_records": len(records),
        "clean_records": len(clean_records),
        "missing_images": missing_images,
        "invalid_images": invalid_images,
        "duplicate_images_removed": duplicate_images,
        "unique_image_contents": len(seen_hashes),
        "unique_products": len(product_counts),
        "rating_distribution": dict(rating_counts),
        "duplicate_details": duplicate_details
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Display summary
    print("\nPREPARATION SUMMARY")
    print("-" * 45)

    print("Original records:", len(records))
    print("Missing images:", missing_images)
    print("Invalid images:", invalid_images)
    print("Duplicate images removed:", duplicate_images)
    print("Clean records:", len(clean_records))
    print("Unique products:", len(product_counts))

    print("\nCLEAN DATASET RATING DISTRIBUTION")
    print("-" * 45)

    for rating in range(1, 6):
        print(
            f"{rating} stars: "
            f"{rating_counts[rating]}"
        )

    print("\nOutput manifest:", OUTPUT_FILE)
    print("Preparation report:", REPORT_FILE)

    print("\nDATASET PREPARATION COMPLETED")


if __name__ == "__main__":
    main()
