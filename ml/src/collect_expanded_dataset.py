
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "ml" / "data" / "processed"

OUTPUT_FILE = OUTPUT_DIR / "electronics_expanded.jsonl"

TARGET_PAIRS = 10000

# Scan shards sequentially until the target is reached.
TOTAL_SHARDS = 34

# Prevent duplicate review records across shards.
seen_ids = set()


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
    print("AMAZON ELECTRONICS - EXPANDED DATA COLLECTION")
    print("=" * 65)

    print("Target:", TARGET_PAIRS)
    print("Available shards:", TOTAL_SHARDS)

    collected = []
    rating_counts = Counter()

    total_scanned = 0

    for shard_number in range(TOTAL_SHARDS):

        shard_name = (
            f"full-{shard_number:05d}-of-00034.parquet"
        )

        shard_url = (
            f"https://huggingface.co/datasets/"
            f"{DATASET_NAME}/resolve/{REVISION}/"
            f"raw_review_Electronics/{shard_name}"
        )

        print(f"\nScanning shard {shard_number + 1}/{TOTAL_SHARDS}")
        print(shard_name)

        try:

            dataset = load_dataset(
                "parquet",
                data_files={"train": shard_url},
                split="train",
                streaming=True
            )

            shard_scanned = 0

            for record in dataset:

                total_scanned += 1
                shard_scanned += 1

                processed = process_record(record)

                if processed is None:
                    continue

                review_id = processed["review_id"]

                if review_id in seen_ids:
                    continue

                seen_ids.add(review_id)

                collected.append(processed)

                rating_counts[processed["rating"]] += 1

                if len(collected) >= TARGET_PAIRS:
                    break

                if shard_scanned % 10000 == 0:
                    print(
                        f"Scanned: {total_scanned:,} | "
                        f"Collected: {len(collected):,}"
                    )

        except Exception as error:

            print(f"Shard failed: {error}")
            continue

        print(
            f"Shard scanned: {shard_scanned:,} | "
            f"Total collected: {len(collected):,}"
        )

        # Save progress after every shard.
        with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

            for item in collected:
                file.write(
                    json.dumps(item, ensure_ascii=False) + "\n"
                )

        if len(collected) >= TARGET_PAIRS:
            break

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("EXPANDED COLLECTION SUMMARY")
    print("=" * 65)

    print("Total records scanned:", total_scanned)
    print("Valid unique pairs collected:", len(collected))
    print("Target:", TARGET_PAIRS)

    print("\nRating distribution:")

    for rating in range(1, 6):

        count = rating_counts[rating]

        percentage = (
            count / max(len(collected), 1)
        ) * 100

        print(
            f"{rating} stars: {count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nOutput file:", OUTPUT_FILE)

    if len(collected) < TARGET_PAIRS:
        print(
            "\nTarget not reached. "
            "Additional shards or a different collection strategy "
            "may be required."
        )

    print("\nCollection process completed!")


if __name__ == "__main__":
    main()
