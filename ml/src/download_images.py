
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
    "processed" / "electronics_pilot.jsonl"
)

IMAGE_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "images" / "downloaded"
)

FAILED_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "images" / "failed"
)

OUTPUT_FILE = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "electronics_1000_with_images.jsonl"
)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10 MB

TIMEOUT = (10, 30)

MAX_RETRIES = 3


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

            # Verify image integrity
            image_bytes = bytes(content)

            with Image.open(io.BytesIO(image_bytes)) as img:

                img.verify()

            # Reopen and convert to RGB
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
# MAIN PIPELINE
# --------------------------------------------------

def main():

    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    FAILED_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_FILE}"
        )

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    print("=" * 60)
    print("CUSTOMER REVIEW IMAGE DOWNLOADER")
    print("=" * 60)

    print(f"\nRecords loaded: {len(records)}")

    stats = Counter()
    failed_records = []
    updated_records = []

    with requests.Session() as session:

        session.headers.update({
            "User-Agent": "MultimodalSentimentProject/1.0"
        })

        for record in tqdm(records, desc="Downloading images"):

            review_id = record["review_id"]

            existing_path = IMAGE_DIR / f"{review_id}.jpg"

            if existing_path.exists():

                try:
                    with Image.open(existing_path) as img:
                        img.verify()

                    record["image_path"] = str(
                        existing_path.relative_to(PROJECT_ROOT)
                    )

                    record["image_status"] = "success"

                    stats["success"] += 1
                    updated_records.append(record)

                    continue

                except (OSError, UnidentifiedImageError):
                    existing_path.unlink(missing_ok=True)

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

            else:

                record["image_path"] = None
                record["image_status"] = "failed"

                stats["failed"] += 1

                failed_records.append({
                    "review_id": review_id,
                    "image_url": record["image_url"],
                    "error": error
                })

            updated_records.append(record)

    # Save updated manifest
    with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

        for record in updated_records:

            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

    # Save failed downloads separately
    failed_file = FAILED_DIR / "failed_downloads.json"

    with open(failed_file, "w", encoding="utf-8") as file:

        json.dump(
            failed_records,
            file,
            indent=4
        )

    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    print(f"Total records: {len(records)}")
    print(f"Successful downloads: {stats['success']}")
    print(f"Failed downloads: {stats['failed']}")

    print(f"\nUpdated manifest: {OUTPUT_FILE}")
    print(f"Failed records: {failed_file}")

    print("\nImage downloading completed!")


if __name__ == "__main__":
    main()
