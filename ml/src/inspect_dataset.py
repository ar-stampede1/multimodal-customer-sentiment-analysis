
from datasets import load_dataset

def main():
    print("=" * 60)
    print("AMAZON REVIEWS 2023 - PARQUET INSPECTION")
    print("=" * 60)

    # Pin the revision containing raw_review_Electronics.
    revision = "32b99ef490c11e6d63223d962253bc696bdae583"

    shard_url = (
        "https://huggingface.co/datasets/"
        "McAuley-Lab/Amazon-Reviews-2023/resolve/"
        f"{revision}/raw_review_Electronics/"
        "full-00000-of-00034.parquet"
    )

    print("\nConnecting to Electronics review dataset...")

    dataset = load_dataset(
        "parquet",
        data_files={"train": shard_url},
        split="train",
        streaming=True
    )

    print("Dataset connected successfully!")

    print("\nInspecting first 3 records...")
    print("=" * 60)

    for index, record in enumerate(dataset):

        print(f"\nRECORD {index + 1}")
        print("-" * 50)

        print("Available fields:", list(record.keys()))

        print("Rating:", record.get("rating"))
        print("Title:", record.get("title"))

        review = record.get("text", "")
        print("Review:", str(review)[:300])

        images = record.get("images")

        print("Images:", images)

        if isinstance(images, list):
            print("Number of images:", len(images))

        print("Product ID:", record.get("asin"))
        print("Parent Product ID:", record.get("parent_asin"))

        if index == 2:
            break

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()
