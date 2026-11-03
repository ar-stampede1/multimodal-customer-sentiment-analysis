
import json
from pathlib import Path
from collections import Counter

from sklearn.model_selection import StratifiedGroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "electronics_clean.jsonl"
)

SPLIT_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "splits"
)

REPORT_FILE = (
    PROJECT_ROOT / "ml" / "evaluation" /
    "split_report.json"
)

RANDOM_STATE = 42


def load_records():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file if line.strip()
        ]

    if not records:
        raise ValueError("Dataset is empty.")

    return records


def write_jsonl(records, output_path):

    with open(output_path, "w", encoding="utf-8") as file:

        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def get_class_distribution(records):

    counts = Counter(record["label_id"] for record in records)

    return {
        str(label): counts.get(label, 0)
        for label in range(5)
    }


def main():

    print("=" * 65)
    print("PRODUCT-AWARE DATASET SPLITTING")
    print("=" * 65)

    records = load_records()

    print("\nTotal records:", len(records))

    # Validate required fields
    required_fields = [
        "parent_asin",
        "rating",
        "label_id",
        "image_hash",
        "image_path"
    ]

    for index, record in enumerate(records):

        for field in required_fields:

            if field not in record or record[field] is None:
                raise ValueError(
                    f"Missing field '{field}' in record {index}"
                )

        if record["label_id"] != record["rating"] - 1:
            raise ValueError(
                f"Invalid label mapping in record {index}"
            )

    # Prepare grouping and stratification data
    labels = [record["label_id"] for record in records]
    groups = [record["parent_asin"] for record in records]

    indices = list(range(len(records)))

    # First split: 90% train + validation, 10% test
    outer_splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_val_indices, test_indices = next(
        outer_splitter.split(indices, labels, groups)
    )

    # Second split: approximately 80% train, 10% validation
    remaining_records = [
        records[index] for index in train_val_indices
    ]

    remaining_labels = [
        labels[index] for index in train_val_indices
    ]

    remaining_groups = [
        groups[index] for index in train_val_indices
    ]

    inner_splitter = StratifiedGroupKFold(
        n_splits=9,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_local, val_local = next(
        inner_splitter.split(
            list(range(len(remaining_records))),
            remaining_labels,
            remaining_groups
        )
    )

    train_records = [
        remaining_records[index] for index in train_local
    ]

    val_records = [
        remaining_records[index] for index in val_local
    ]

    test_records = [
        records[index] for index in test_indices
    ]

    # Create output directory
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    # Save individual manifests
    write_jsonl(train_records, SPLIT_DIR / "train.jsonl")
    write_jsonl(val_records, SPLIT_DIR / "validation.jsonl")
    write_jsonl(test_records, SPLIT_DIR / "test.jsonl")

    # Verify product leakage
    train_products = {
        record["parent_asin"] for record in train_records
    }

    val_products = {
        record["parent_asin"] for record in val_records
    }

    test_products = {
        record["parent_asin"] for record in test_records
    }

    train_val_overlap = train_products & val_products
    train_test_overlap = train_products & test_products
    val_test_overlap = val_products & test_products

    # Verify image leakage
    train_images = {
        record["image_hash"] for record in train_records
    }

    val_images = {
        record["image_hash"] for record in val_records
    }

    test_images = {
        record["image_hash"] for record in test_records
    }

    image_overlaps = {
        "train_validation": len(train_images & val_images),
        "train_test": len(train_images & test_images),
        "validation_test": len(val_images & test_images)
    }

    # Check that all records are accounted for
    total_split_records = (
        len(train_records) +
        len(val_records) +
        len(test_records)
    )

    if total_split_records != len(records):
        raise ValueError("Some records are missing or duplicated.")

    # Check no leakage
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError("Product leakage detected!")

    if any(image_overlaps.values()):
        raise ValueError("Image leakage detected!")

    # Generate report
    report = {
        "total_records": len(records),

        "split_sizes": {
            "train": len(train_records),
            "validation": len(val_records),
            "test": len(test_records)
        },

        "split_percentages": {
            "train": round(len(train_records) / len(records) * 100, 2),
            "validation": round(len(val_records) / len(records) * 100, 2),
            "test": round(len(test_records) / len(records) * 100, 2)
        },

        "class_distributions": {
            "train": get_class_distribution(train_records),
            "validation": get_class_distribution(val_records),
            "test": get_class_distribution(test_records)
        },

        "product_overlap": {
            "train_validation": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "validation_test": len(val_test_overlap)
        },

        "image_overlap": image_overlaps,

        "random_state": RANDOM_STATE
    }

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(REPORT_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    # Display summary
    print("\nSPLIT SUMMARY")
    print("-" * 45)

    print(
        f"Training:   {len(train_records)} "
        f"({len(train_records) / len(records) * 100:.2f}%)"
    )

    print(
        f"Validation: {len(val_records)} "
        f"({len(val_records) / len(records) * 100:.2f}%)"
    )

    print(
        f"Testing:    {len(test_records)} "
        f"({len(test_records) / len(records) * 100:.2f}%)"
    )

    print("\nCLASS DISTRIBUTION")
    print("-" * 45)

    for name, split_records in [
        ("Training", train_records),
        ("Validation", val_records),
        ("Testing", test_records)
    ]:

        print(f"\n{name}:")

        distribution = get_class_distribution(split_records)

        for label, count in distribution.items():
            print(f"  Class {label}: {count}")

    print("\nLEAKAGE VERIFICATION")
    print("-" * 45)

    print("Train-validation product overlap:", len(train_val_overlap))
    print("Train-test product overlap:", len(train_test_overlap))
    print("Validation-test product overlap:", len(val_test_overlap))

    print("Train-validation image overlap:", image_overlaps["train_validation"])
    print("Train-test image overlap:", image_overlaps["train_test"])
    print("Validation-test image overlap:", image_overlaps["validation_test"])

    print("\nOUTPUT FILES")
    print("-" * 45)

    print("Training:", SPLIT_DIR / "train.jsonl")
    print("Validation:", SPLIT_DIR / "validation.jsonl")
    print("Testing:", SPLIT_DIR / "test.jsonl")
    print("Report:", REPORT_FILE)

    print("\nDATASET SPLITTING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()
