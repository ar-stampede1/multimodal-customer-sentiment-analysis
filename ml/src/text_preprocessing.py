
from pathlib import Path
import json

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "splits"
)

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256


def load_tokenizer():
    """
    Load the pretrained DistilBERT tokenizer.
    """

    print("Loading DistilBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizer loaded successfully!")

    return tokenizer


def prepare_text(record):
    """
    Combine review title and review text.
    """

    title = str(record.get("review_title") or "").strip()
    review = str(record.get("review_text") or "").strip()

    if title and review:
        return title + " " + review

    return title or review


def tokenize_review(text, tokenizer):
    """
    Convert review text into DistilBERT input tensors.
    """

    encoding = tokenizer(
        text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    return {
        "input_ids": encoding["input_ids"].squeeze(0),
        "attention_mask": encoding["attention_mask"].squeeze(0)
    }


def main():

    print("=" * 65)
    print("DISTILBERT TEXT PREPROCESSING")
    print("=" * 65)

    tokenizer = load_tokenizer()

    train_file = SPLIT_DIR / "train.jsonl"

    if not train_file.exists():
        raise FileNotFoundError(train_file)

    with open(train_file, "r", encoding="utf-8") as file:
        records = [
            json.loads(line)
            for line in file if line.strip()
        ]

    print("\nTraining records loaded:", len(records))

    print("\nTOKENIZATION TEST")
    print("-" * 45)

    total_tokens = 0
    truncated_reviews = 0

    for index, record in enumerate(records[:100]):

        text = prepare_text(record)

        if not text:
            raise ValueError(
                f"Empty review text found at record {index}"
            )

        # Count tokens before truncation
        original_tokens = tokenizer(
            text,
            add_special_tokens=True,
            truncation=False
        )["input_ids"]

        # Tokenize with fixed maximum length
        encoded = tokenize_review(text, tokenizer)

        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]

        actual_tokens = int(attention_mask.sum().item())

        if len(original_tokens) > MAX_LENGTH:
            truncated_reviews += 1

        if input_ids.shape[0] != MAX_LENGTH:
            raise ValueError("Unexpected input ID length")

        if attention_mask.shape[0] != MAX_LENGTH:
            raise ValueError("Unexpected attention mask length")

        total_tokens += actual_tokens

        if index == 0:

            print("\nExample review:")
            print(text[:300])

            print("\nToken IDs shape:", tuple(input_ids.shape))
            print(
                "Attention mask shape:",
                tuple(attention_mask.shape)
            )

            print("Actual tokens:", actual_tokens)
            print("First 15 token IDs:", input_ids[:15].tolist())

    tested_records = min(100, len(records))

    print("\nPREPROCESSING SUMMARY")
    print("-" * 45)

    print("Records tested:", tested_records)
    print("Maximum sequence length:", MAX_LENGTH)

    print(
        "Average tokens per review:",
        round(total_tokens / tested_records, 2)
        if tested_records else 0
    )

    print("Reviews requiring truncation:", truncated_reviews)

    print("\nTEXT PREPROCESSING TEST COMPLETED")


if __name__ == "__main__":
    main()
