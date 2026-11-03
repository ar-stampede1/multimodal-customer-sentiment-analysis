import json
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_FILE = (
    PROJECT_ROOT
    / "ml"
    / "data"
    / "processed"
    / "electronics_expanded.jsonl"
)


def main():

    print("=" * 65)
    print("EXPANDED DATASET INTEGRITY VERIFICATION")
    print("=" * 65)

    records = []
    invalid_lines = 0

    with open(DATASET_FILE, "r", encoding="utf-8") as file:

        for line in file:

            try:
                records.append(json.loads(line))

            except json.JSONDecodeError:
                invalid_lines += 1

    review_ids = [
        record.get("review_id")
        for record in records
    ]

    parent_asins = [
        record.get("parent_asin")
        for record in records
    ]

    rating_counts = Counter(
        record.get("rating")
        for record in records
    )

    required_fields = [
        "review_id",
        "product_id",
        "parent_asin",
        "review_title",
        "review_text",
        "rating",
        "label_id",
        "image_url"
    ]

    missing_fields = sum(
        1
        for record in records
        if any(
            field not in record or record[field] is None
            for field in required_fields
        )
    )

    print("\nTotal records:", len(records))
    print("Invalid JSON lines:", invalid_lines)
    print("Unique review IDs:", len(set(review_ids)))
    print("Duplicate review IDs:", len(review_ids) - len(set(review_ids)))
    print("Unique products:", len(set(parent_asins)))
    print("Records with missing required fields:", missing_fields)

    print("\nRating distribution:")

    for rating in range(1, 6):
        print(f"{rating} stars: {rating_counts[rating]}")

    print("\nFile size:", round(DATASET_FILE.stat().st_size / (1024 * 1024), 2), "MB")

    success = (
        len(records) == 10000
        and invalid_lines == 0
        and len(set(review_ids)) == len(records)
        and missing_fields == 0
        and all(rating_counts[rating] > 0 for rating in range(1, 6))
    )

    print("\nVerification:", "PASSED" if success else "FAILED")


if __name__ == "__main__":
    main()