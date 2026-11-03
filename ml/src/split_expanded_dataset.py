
import json
from pathlib import Path
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "electronics_expanded_clean.jsonl"
)

SPLIT_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "expanded_splits"
)

REPORT_FILE = (
    PROJECT_ROOT / "ml" / "evaluation" /
    "expanded_split_report.json"
)

RANDOM_STATE = 42

NUM_CLASSES = 5


# --------------------------------------------------
# LOAD DATASET
# --------------------------------------------------

def load_jsonl(file_path):

    records = []

    with open(file_path, "r", encoding="utf-8") as file:

        for line in file:

            if line.strip():
                records.append(json.loads(line))

    return records


# --------------------------------------------------
# SAVE SPLIT
# --------------------------------------------------

def save_jsonl(records, output_path):

    with open(output_path, "w", encoding="utf-8") as file:

        for record in records:

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


# --------------------------------------------------
# CLASS DISTRIBUTION
# --------------------------------------------------

def get_class_distribution(records):

    counts = Counter(
        record["label_id"]
        for record in records
    )

    return {
        str(class_id): counts.get(class_id, 0)
        for class_id in range(NUM_CLASSES)
    }


# --------------------------------------------------
# MAIN SPLITTING PIPELINE
# --------------------------------------------------

def main():

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("EXPANDED DATASET - PRODUCT-AWARE SPLITTING")
    print("=" * 65)

    records = load_jsonl(INPUT_FILE)

    print("\nTotal clean records:", len(records))

    labels = np.array([
        record["label_id"]
        for record in records
    ])

    groups = np.array([
        record["parent_asin"]
        for record in records
    ])

    indices = np.arange(len(records))

    # --------------------------------------------------
    # OUTER SPLIT: 80% TRAIN+VALIDATION, 20% TEST
    # --------------------------------------------------

    outer_splitter = StratifiedGroupKFold(
        n_splits=10,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    train_val_indices, test_indices = next(
        outer_splitter.split(
            indices,
            labels,
            groups
        )
    )

    # --------------------------------------------------
    # INNER SPLIT: 90% TRAIN, 10% VALIDATION
    # --------------------------------------------------

    inner_splitter = StratifiedGroupKFold(
        n_splits=9,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    inner_train_indices, val_indices = next(
        inner_splitter.split(
            train_val_indices,
            labels[train_val_indices],
            groups[train_val_indices]
        )
    )

    train_indices = train_val_indices[inner_train_indices]

    # --------------------------------------------------
    # CREATE RECORD LISTS
    # --------------------------------------------------

    train_records = [
        records[index]
        for index in train_indices
    ]

    val_records = [
        records[index]
        for index in train_val_indices[val_indices]
    ]

    test_records = [
        records[index]
        for index in test_indices
    ]

    # --------------------------------------------------
    # SAVE SPLITS
    # --------------------------------------------------

    save_jsonl(
        train_records,
        SPLIT_DIR / "train.jsonl"
    )

    save_jsonl(
        val_records,
        SPLIT_DIR / "validation.jsonl"
    )

    save_jsonl(
        test_records,
        SPLIT_DIR / "test.jsonl"
    )

    # --------------------------------------------------
    # CHECK PRODUCT LEAKAGE
    # --------------------------------------------------

    train_products = {
        record["parent_asin"]
        for record in train_records
    }

    val_products = {
        record["parent_asin"]
        for record in val_records
    }

    test_products = {
        record["parent_asin"]
        for record in test_records
    }

    train_val_overlap = train_products & val_products
    train_test_overlap = train_products & test_products
    val_test_overlap = val_products & test_products

    # --------------------------------------------------
    # CHECK IMAGE HASH LEAKAGE
    # --------------------------------------------------

    train_hashes = {
        record["image_sha256"]
        for record in train_records
    }

    val_hashes = {
        record["image_sha256"]
        for record in val_records
    }

    test_hashes = {
        record["image_sha256"]
        for record in test_records
    }

    image_overlaps = {
        "train_validation": len(train_hashes & val_hashes),
        "train_test": len(train_hashes & test_hashes),
        "validation_test": len(val_hashes & test_hashes)
    }

    # --------------------------------------------------
    # GENERATE REPORT
    # --------------------------------------------------

    report = {
        "total_records": len(records),

        "split_sizes": {
            "train": len(train_records),
            "validation": len(val_records),
            "test": len(test_records)
        },

        "split_percentages": {
            "train": len(train_records) / len(records) * 100,
            "validation": len(val_records) / len(records) * 100,
            "test": len(test_records) / len(records) * 100
        },

        "class_distribution": {
            "train": get_class_distribution(train_records),
            "validation": get_class_distribution(val_records),
            "test": get_class_distribution(test_records)
        },

        "product_overlap": {
            "train_validation": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "validation_test": len(val_test_overlap)
        },

        "image_hash_overlap": image_overlaps,

        "unique_products": len(set(groups)),

        "random_state": RANDOM_STATE
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:

        json.dump(report, file, indent=4)

    # --------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------

    print("\nSPLIT SUMMARY")
    print("-" * 65)

    for name, split_records in [
        ("Training", train_records),
        ("Validation", val_records),
        ("Testing", test_records)
    ]:

        percentage = len(split_records) / len(records) * 100

        print(
            f"{name}: {len(split_records)} "
            f"({percentage:.2f}%)"
        )

        print(
            "Class distribution:",
            get_class_distribution(split_records)
        )

    print("\nPRODUCT OVERLAP")
    print("Train-Validation:", len(train_val_overlap))
    print("Train-Test:", len(train_test_overlap))
    print("Validation-Test:", len(val_test_overlap))

    print("\nIMAGE HASH OVERLAP")
    print(image_overlaps)

    print("\nSplit report:", REPORT_FILE)

    # --------------------------------------------------
    # FINAL VALIDATION
    # --------------------------------------------------

    total_split_records = (
        len(train_records)
        + len(val_records)
        + len(test_records)
    )

    success = (
        total_split_records == len(records)
        and not train_val_overlap
        and not train_test_overlap
        and not val_test_overlap
        and all(value == 0 for value in image_overlaps.values())
    )

    print(
        "\nSPLIT VALIDATION:",
        "PASSED" if success else "FAILED"
    )


if __name__ == "__main__":
    main()
