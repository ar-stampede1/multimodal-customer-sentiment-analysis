import json
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "electronics_1000_with_images.jsonl"
)

REPORT_DIR = PROJECT_ROOT / "ml" / "evaluation"

RATING_LABELS = {
    1: "Very Dissatisfied",
    2: "Dissatisfied",
    3: "Neutral",
    4: "Satisfied",
    5: "Very Satisfied"
}


def calculate_hash(file_path):

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def main():

    print("=" * 65)
    print("MULTIMODAL DATASET QUALITY ANALYSIS")
    print("=" * 65)

    if not MANIFEST.exists():
        raise FileNotFoundError(MANIFEST)

    with open(MANIFEST, "r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file if line.strip()
        ]

    rating_counts = Counter()
    product_counts = Counter()
    image_hashes = defaultdict(list)
    image_records = defaultdict(list)

    valid_images = 0
    missing_images = 0
    corrupted_images = 0

    image_sizes = []
    image_dimensions = []

    for record in records:

        rating_counts[record["rating"]] += 1
        product_counts[record["parent_asin"]] += 1

        relative_path = record.get("image_path")

        if not relative_path:
            missing_images += 1
            continue

        image_path = PROJECT_ROOT / relative_path

        if not image_path.exists():
            missing_images += 1
            continue

        try:
            with Image.open(image_path) as img:
                img.verify()

            with Image.open(image_path) as img:
                width, height = img.size

                if img.format != "JPEG":
                    raise ValueError("Image is not JPEG")

                if img.mode != "RGB":
                    raise ValueError("Image is not RGB")

            file_size = image_path.stat().st_size
            image_hash = calculate_hash(image_path)

            valid_images += 1

            image_sizes.append(file_size)
            image_dimensions.append((width, height))

            image_hashes[image_hash].append(record["review_id"])

            image_records[image_hash].append({
                "review_id": record["review_id"],
                "parent_asin": record["parent_asin"],
                "product_id": record.get("product_id"),
                "rating": record["rating"],
                "review_title": record.get("review_title"),
                "image_path": relative_path
            })
        except (
            OSError,
            ValueError,
            UnidentifiedImageError
        ):

            corrupted_images += 1

    duplicate_groups = []

    for image_hash, review_ids in image_hashes.items():

        if len(review_ids) > 1:

            duplicate_groups.append({
                "image_hash": image_hash,
                "number_of_records": len(review_ids),
                "same_product": (
                    len({
                        item["parent_asin"]
                        for item in image_records[image_hash]
                    }) == 1
                ),
                "records": image_records[image_hash]
            })

    unique_products = len(product_counts)

    total_reviews = len(records)

    print("\nDATASET SUMMARY")
    print("-" * 45)

    print("Total records:", total_reviews)
    print("Valid images:", valid_images)
    print("Missing images:", missing_images)
    print("Corrupted images:", corrupted_images)

    print("\nRATING DISTRIBUTION")
    print("-" * 45)

    for rating in range(1, 6):

        count = rating_counts[rating]

        percentage = (
            count / total_reviews * 100
            if total_reviews else 0
        )

        print(
            f"{rating} stars ({RATING_LABELS[rating]}): "
            f"{count} ({percentage:.2f}%)"
        )

    print("\nIMAGE STATISTICS")
    print("-" * 45)

    print("Unique image contents:", len(image_hashes))
    print("Duplicate image groups:", len(duplicate_groups))

    if image_sizes:
        print(
            "Average image size:",
            round(sum(image_sizes) / len(image_sizes) / 1024, 2),
            "KB"
        )

        print(
            "Smallest image:",
            round(min(image_sizes) / 1024, 2),
            "KB"
        )

        print(
            "Largest image:",
            round(max(image_sizes) / 1024, 2),
            "KB"
        )

    if image_dimensions:

        widths = [w for w, h in image_dimensions]
        heights = [h for w, h in image_dimensions]

        print(
            "Image width range:",
            min(widths), "-", max(widths)
        )

        print(
            "Image height range:",
            min(heights), "-", max(heights)
        )

    print("\nPRODUCT STATISTICS")
    print("-" * 45)

    print("Unique parent products:", unique_products)

    print(
        "Products with multiple reviews:",
        sum(1 for count in product_counts.values() if count > 1)
    )

    print(
        "Maximum reviews per product:",
        max(product_counts.values(), default=0)
    )

    print("\nDUPLICATE IMAGE INVESTIGATION")
    print("-" * 45)
    print("Duplicate image groups:", len(duplicate_groups))

    for index, group in enumerate(duplicate_groups, start=1):

        print(f"\nGroup {index}")
        print("Image hash:", group["image_hash"][:16] + "...")
        print("Number of records:", group["number_of_records"])
        print("Same product:", group["same_product"])

        for item in group["records"]:
            print(
                "  Review:", item["review_id"],
                "| Product:", item["parent_asin"],
                "| Rating:", item["rating"],
                "| Image:", item["image_path"]
            )


    # Save report
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    report = {
        "total_records": total_reviews,
        "valid_images": valid_images,
        "missing_images": missing_images,
        "corrupted_images": corrupted_images,
        "rating_distribution": dict(rating_counts),
        "unique_image_contents": len(image_hashes),
        "duplicate_image_groups": len(duplicate_groups),
        "duplicate_image_details": duplicate_groups,
        "unique_products": unique_products,
        "product_review_counts": dict(product_counts),
        "average_image_size_kb": (
            round(sum(image_sizes) / len(image_sizes) / 1024, 2)
            if image_sizes else None
        )
    }

    report_path = REPORT_DIR / "pilot_quality_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\nReport saved:", report_path)

    print("\nDATASET QUALITY ANALYSIS COMPLETED")


if __name__ == "__main__":
    main()
