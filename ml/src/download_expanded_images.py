
import json
import time
import io
from pathlib import Path
from collections import Counter

import requests
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "electronics_expanded.jsonl"
)

IMAGE_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "images" / "expanded_downloaded"
)

FAILED_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "images" / "expanded_failed"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "electronics_expanded_with_images.jsonl"
)

FAILED_FILE = FAILED_DIR / "failed_downloads.json"

MAX_IMAGE_SIZE = 10 * 1024 * 1024

TIMEOUT = (10, 30)

MAX_RETRIES = 3

SAVE_INTERVAL = 100


# --------------------------------------------------
# IMAGE DOWNLOADER
# --------------------------------------------------

def download_image(session, url, review_id):

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.get(
                url,
                timeout=TIMEOUT,
                stream=True
            )

            response.raise_for_status()

            content_type = response.headers.get(
                "Content-Type", ""
            ).lower()

            if not content_type.startswith("image/"):
                return None, "Invalid content type"

            content = bytearray()

            for chunk in response.iter_content(8192):

                if not chunk:
                    continue

                content.extend(chunk)

                if len(content) > MAX_IMAGE_SIZE:
                    return None, "Image exceeds 10 MB limit"

            image_bytes = bytes(content)

            # Verify image integrity.
            with Image.open(io.BytesIO(image_bytes)) as img:
                img.verify()

            # Reopen and convert to RGB.
            with Image.open(io.BytesIO(image_bytes)) as img:

                img = img.convert("RGB")

                if img.width < 32 or img.height < 32:
                    return None, "Image dimensions too small"

                output_path = IMAGE_DIR / f"{review_id}.jpg"

                img.save(
                    output_path,
                    format="JPEG",
                    quality=95
                )

            return output_path, None

        except (
            requests.RequestException,
            UnidentifiedImageError,
            OSError,
            ValueError
        ) as error:

            if attempt == MAX_RETRIES:
                return None, str(error)

            time.sleep(attempt * 2)

    return None, "Maximum retries exceeded"


# --------------------------------------------------
# SAVE PROGRESS
# --------------------------------------------------

def save_manifest(records):

    temporary_file = OUTPUT_FILE.with_suffix(".tmp")

    with open(temporary_file, "w", encoding="utf-8") as file:

        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    temporary_file.replace(OUTPUT_FILE)


def save_failures(failed_records):

    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    with open(FAILED_FILE, "w", encoding="utf-8") as file:
        json.dump(failed_records, file, indent=4)


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------

def main():

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("=" * 65)
    print("EXPANDED CUSTOMER REVIEW IMAGE DOWNLOADER")
    print("=" * 65)

    print("Records loaded:", len(records))
    print("Image directory:", IMAGE_DIR)

    stats = Counter()
    failed_records = []

    # Load previous failure information if available.
    if FAILED_FILE.exists():

        with open(FAILED_FILE, "r", encoding="utf-8") as file:
            failed_records = json.load(file)

    # Keep only the latest failure for each review.
    failed_by_id = {
        item["review_id"]: item
        for item in failed_records
    }

    with requests.Session() as session:

        session.headers.update({
            "User-Agent": "MultimodalSentimentProject/1.0"
        })

        for index, record in enumerate(
            tqdm(records, desc="Downloading images"),
            start=1
        ):

            review_id = record["review_id"]

            existing_path = IMAGE_DIR / f"{review_id}.jpg"

            # Resume valid previously downloaded images.
            if existing_path.exists():

                try:
                    with Image.open(existing_path) as img:
                        img.verify()

                    record["image_path"] = str(
                        existing_path.relative_to(PROJECT_ROOT)
                    )

                    record["image_status"] = "success"

                    stats["success"] += 1

                    failed_by_id.pop(review_id, None)

                except (OSError, UnidentifiedImageError):

                    existing_path.unlink(missing_ok=True)

            # Download if no valid image exists.
            if not existing_path.exists():

                image_path, error = download_image(
                    session,
                    record["image_url"],
                    review_id
                )

                if image_path:

                    record["image_path"] = str(
                        image_path.relative_to(PROJECT_ROOT)
                    )

                    record["image_status"] = "success"

                    stats["success"] += 1

                    failed_by_id.pop(review_id, None)

                else:

                    record["image_path"] = None
                    record["image_status"] = "failed"

                    stats["failed"] += 1

                    failed_by_id[review_id] = {
                        "review_id": review_id,
                        "image_url": record["image_url"],
                        "error": error
                    }

            # Periodically save progress.
            if index % SAVE_INTERVAL == 0:

                save_manifest(records)
                save_failures(list(failed_by_id.values()))

    # Final save.
    save_manifest(records)
    save_failures(list(failed_by_id.values()))

    print("\n" + "=" * 65)
    print("EXPANDED IMAGE DOWNLOAD SUMMARY")
    print("=" * 65)

    print("Total records:", len(records))
    print("Successful images:", stats["success"])
    print("Failed images:", stats["failed"])

    success_rate = (
        stats["success"] / max(len(records), 1)
    ) * 100

    print(f"Success rate: {success_rate:.2f}%")

    print("\nUpdated manifest:", OUTPUT_FILE)
    print("Failed downloads:", FAILED_FILE)

    print("\nImage downloading completed!")


if __name__ == "__main__":
    main()
