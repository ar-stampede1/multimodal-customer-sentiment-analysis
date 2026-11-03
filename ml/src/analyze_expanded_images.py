
import json
import hashlib
from pathlib import Path
from collections import Counter

from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "electronics_expanded_with_images.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "ml" / "data" / "processed"
)

CLEAN_FILE = OUTPUT_DIR / "electronics_expanded_clean.jsonl"

REPORT_FILE = (
    PROJECT_ROOT / "ml" / "evaluation" /
    "expanded_image_quality_report.json"
)

MIN_WIDTH = 32
MIN_HEIGHT = 32


# --------------------------------------------------
# HASH FUNCTION
# --------------------------------------------------

def calculate_sha256(image_path):

    sha256 = hashlib.sha256()

    with open(image_path, "rb") as file:

        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# --------------------------------------------------
# MAIN ANALYSIS
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("=" * 65)
    print("EXPANDED DATASET IMAGE QUALITY ANALYSIS")
    print("=" * 65)

    print("Total records loaded:", len(records))

    stats = Counter()
    rating_counts = Counter()

    seen_hashes = {}
    clean_records = []
    duplicate_groups = {}

    total_image_size = 0

    for record in tqdm(records, desc="Analyzing images"):

        image_path_value = record.get("image_path")

        if not image_path_value:
            stats["missing"] += 1
            continue

        image_path = PROJECT_ROOT / image_path_value

        if not image_path.exists():
            stats["missing"] += 1
            continue

        try:

            # Check image integrity.
            with Image.open(image_path) as image:
                image.verify()

            # Read image dimensions.
            with Image.open(image_path) as image:

                width, height = image.size
                image_format = image.format

            if width < MIN_WIDTH or height < MIN_HEIGHT:

                stats["invalid_dimensions"] += 1
                continue

            # Calculate file hash.
            image_hash = calculate_sha256(image_path)

            record["image_width"] = width
            record["image_height"] = height
            record["image_format"] = image_format
            record["image_sha256"] = image_hash

            file_size = image_path.stat().st_size
            total_image_size += file_size

            # Detect exact duplicate image contents.
            if image_hash in seen_hashes:

                stats["duplicate_images_removed"] += 1

                original_id = seen_hashes[image_hash]

                duplicate_groups.setdefault(
                    image_hash,
                    [original_id]
                ).append(record["review_id"])

                continue

            seen_hashes[image_hash] = record["review_id"]

            clean_records.append(record)

            stats["valid_images"] += 1
            rating_counts[record["rating"]] += 1

        except (
            OSError,
            ValueError,
            UnidentifiedImageError
        ):

            stats["corrupted"] += 1
            continue

    # --------------------------------------------------
    # SAVE CLEAN MANIFEST
    # --------------------------------------------------

    with open(CLEAN_FILE, "w", encoding="utf-8") as file:

        for record in clean_records:

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # --------------------------------------------------
    # QUALITY REPORT
    # --------------------------------------------------

    report = {
        "total_records": len(records),
        "valid_unique_records": len(clean_records),
        "missing_images": stats["missing"],
        "corrupted_images": stats["corrupted"],
        "invalid_dimensions": stats["invalid_dimensions"],
        "duplicate_images_removed": stats["duplicate_images_removed"],
        "unique_image_hashes": len(seen_hashes),
        "unique_products": len({
            record["parent_asin"]
            for record in clean_records
        }),
        "average_image_size_kb": (
            total_image_size / max(stats["valid_images"], 1) / 1024
        ),
        "rating_distribution": dict(rating_counts),
        "duplicate_groups": duplicate_groups
    }

    with open(REPORT_FILE, "w", encoding="utf-8") as file:

        json.dump(report, file, indent=4)

    # --------------------------------------------------
    # PRINT SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("IMAGE QUALITY SUMMARY")
    print("=" * 65)

    print("Original records:", len(records))
    print("Valid unique images:", len(clean_records))
    print("Missing images:", stats["missing"])
    print("Corrupted images:", stats["corrupted"])
    print("Invalid dimensions:", stats["invalid_dimensions"])
    print(
        "Duplicate images removed:",
        stats["duplicate_images_removed"]
    )

    print("\nUnique products:", report["unique_products"])

    print(
        "Average image size:",
        f"{report['average_image_size_kb']:.2f} KB"
    )

    print("\nRating distribution after cleaning:")

    for rating in range(1, 6):

        count = rating_counts[rating]

        percentage = (
            count / max(len(clean_records), 1)
        ) * 100

        print(
            f"{rating} stars: {count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nClean manifest:", CLEAN_FILE)
    print("Quality report:", REPORT_FILE)

    print("\nImage quality analysis completed!")


if __name__ == "__main__":
    main()
