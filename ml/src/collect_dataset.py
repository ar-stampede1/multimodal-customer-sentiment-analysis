
import json
import hashlib
from pathlib import Path
from collections import Counter

from datasets import load_dataset
from tqdm import tqdm


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_NAME = "McAuley-Lab/Amazon-Reviews-2023"

REVISION = "32b99ef490c11e6d63223d962253bc696bdae583"

SHARD_URL = (
    f"https://huggingface.co/datasets/{DATASET_NAME}/resolve/"
    f"{REVISION}/raw_review_Electronics/"
    "full-00000-of-00034.parquet"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "electronics_pilot.jsonl"

TARGET_PAIRS = 1000
MAX_RECORDS_TO_SCAN = 50000


# Original rating -> model class
RATING_MAP = {
    1: (0, "Very Dissatisfied"),
    2: (1, "Dissatisfied"),
    3: (2, "Neutral"),
    4: (3, "Satisfied"),
    5: (4, "Very Satisfied"),
}


# --------------------------------------------------
# IMAGE URL EXTRACTION
# --------------------------------------------------

def extract_image_url(images):

    if not isinstance(images, list):
        return None

    for image in images:

        if not isinstance(image, dict):
            continue

        if image.get("attachment_type") != "IMAGE":
            continue

        # Prefer medium resolution
        url = (
            image.get("medium_image_url")
            or image.get("large_image_url")
            or image.get("small_image_url")
        )

        if isinstance(url, str) and url.startswith("https://"):
            return url

    return None


# --------------------------------------------------
# RECORD VALIDATION
# --------------------------------------------------

def process_record(record):

    rating = record.get("rating")
    title = record.get("title")
    review_text = record.get("text")

    if not isinstance(rating, (int, float)):
        return None

    if not float(rating).is_integer():
        return None

    rating = int(rating)

    if rating not in RATING_MAP:
        return None

    if not isinstance(review_text, str) or not review_text.strip():
        return None

    title = title.strip() if isinstance(title, str) else ""

    image_url = extract_image_url(record.get("images"))

    if not image_url:
        return None

    asin = record.get("asin")
    parent_asin = record.get("parent_asin") or asin

    if not asin or not parent_asin:
        return None

    label_id, satisfaction_label = RATING_MAP[rating]

    # Generate deterministic review identifier
    unique_string = (
        f"{asin}|{parent_asin}|{rating}|"
        f"{title}|{review_text}|{image_url}"
    )

    review_id = hashlib.sha256(
        unique_string.encode("utf-8")
    ).hexdigest()

    return {
        "review_id": review_id,
        "product_id": str(asin),
        "parent_asin": str(parent_asin),
        "review_title": title,
        "review_text": review_text.strip(),
        "rating": rating,
        "label_id": label_id,
        "satisfaction_label": satisfaction_label,
        "image_url": image_url,
        "timestamp": record.get("timestamp"),
        "verified_purchase": record.get("verified_purchase"),
    }


# --------------------------------------------------
# MAIN COLLECTION PIPELINE
# --------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AMAZON ELECTRONICS - PILOT DATA COLLECTION")
    print("=" * 65)

    print("\nLoading Parquet shard...")

    dataset = load_dataset(
        "parquet",
        data_files={"train": SHARD_URL},
        split="train",
        streaming=True
    )

    print("Dataset connected successfully!")

    collected = []
    seen_ids = set()

    rating_counts = Counter()

    scanned = 0

    print("\nScanning records...\n")

    with tqdm(total=MAX_RECORDS_TO_SCAN) as progress:

        for record in dataset:

            scanned += 1
            progress.update(1)

            processed = process_record(record)

            if processed is None:
                pass

            elif processed["review_id"] in seen_ids:
                pass

            else:
                seen_ids.add(processed["review_id"])
                collected.append(processed)

                rating_counts[processed["rating"]] += 1

            if len(collected) >= TARGET_PAIRS:
                break

            if scanned >= MAX_RECORDS_TO_SCAN:
                break

    # Save JSONL manifest
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for record in collected:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print summary
    print("\n" + "=" * 65)
    print("COLLECTION SUMMARY")
    print("=" * 65)

    print(f"Records scanned: {scanned}")
    print(f"Valid pairs collected: {len(collected)}")
    print(f"Target pairs: {TARGET_PAIRS}")

    print("\nRating distribution:")

    for rating in range(1, 6):

        count = rating_counts[rating]

        print(
            f"{rating} stars: {count} "
            f"({count / max(len(collected), 1) * 100:.2f}%)"
        )

    print(f"\nOutput file: {OUTPUT_FILE}")

    if len(collected) < TARGET_PAIRS:
        print(
            "\nPilot target not reached. "
            "We may need to scan additional shards."
        )

    print("\nCollection completed!")


if __name__ == "__main__":
    main()
