
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, UnidentifiedImageError
from transformers import AutoTokenizer
from torchvision import transforms


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PILOT_SPLIT_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "splits"
)

EXPANDED_SPLIT_DIR = (
    PROJECT_ROOT / "ml" / "data" /
    "processed" / "expanded_splits"
)

MODEL_NAME = "distilbert-base-uncased"

MAX_LENGTH = 256
BATCH_SIZE = 4

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# MULTIMODAL DATASET
# --------------------------------------------------

class MultimodalReviewDataset(Dataset):

    def __init__(self, manifest_path, tokenizer):

        self.manifest_path = Path(manifest_path)
        self.tokenizer = tokenizer

        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)

        with open(
            self.manifest_path,
            "r",
            encoding="utf-8"
        ) as file:

            self.records = [
                json.loads(line)
                for line in file if line.strip()
            ]

        if not self.records:
            raise ValueError("Dataset contains no records.")

        print(
            f"Loaded {len(self.records)} records from "
            f"{self.manifest_path.name}"
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        record = self.records[index]

        # Prepare review text
        title = str(record.get("review_title") or "").strip()
        review = str(record.get("review_text") or "").strip()

        text = f"{title} {review}".strip()

        if not text:
            raise ValueError(
                f"Empty review text at index {index}"
            )

        # Tokenize review text
        encoding = self.tokenizer(
            text,
            max_length=MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )

        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # Load image
        relative_path = record.get("image_path")

        if not relative_path:
            raise ValueError(
                f"Missing image path at index {index}"
            )

        image_path = PROJECT_ROOT / relative_path

        if not image_path.exists():
            raise FileNotFoundError(image_path)

        try:

            with Image.open(image_path) as img:
                image = IMAGE_TRANSFORM(img.convert("RGB"))

        except (OSError, UnidentifiedImageError) as error:

            raise RuntimeError(
                f"Unable to load image at index {index}: {error}"
            ) from error

        # Validate satisfaction label
        label = int(record["label_id"])

        if label not in range(5):
            raise ValueError(
                f"Invalid label {label} at index {index}"
            )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image": image,
            "label": torch.tensor(label, dtype=torch.long)
        }


# --------------------------------------------------
# DATALOADER FACTORY
# --------------------------------------------------

def create_dataloaders(dataset_version="pilot"):

    if dataset_version == "pilot":
        split_dir = PILOT_SPLIT_DIR

    elif dataset_version == "expanded":
        split_dir = EXPANDED_SPLIT_DIR

    else:
        raise ValueError(
            "dataset_version must be 'pilot' or 'expanded'"
        )

    print("=" * 65)
    print(f"{dataset_version.upper()} MULTIMODAL DATASET")
    print("=" * 65)

    print("\nLoading DistilBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = MultimodalReviewDataset(
        split_dir / "train.jsonl",
        tokenizer
    )

    val_dataset = MultimodalReviewDataset(
        split_dir / "validation.jsonl",
        tokenizer
    )

    test_dataset = MultimodalReviewDataset(
        split_dir / "test.jsonl",
        tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader


# --------------------------------------------------
# TEST
# --------------------------------------------------

def main():

    train_loader, val_loader, test_loader = (
        create_dataloaders(dataset_version="expanded")
    )

    print("\nEXPANDED DATALOADER SUMMARY")
    print("-" * 45)

    print("Training records:", len(train_loader.dataset))
    print("Validation records:", len(val_loader.dataset))
    print("Testing records:", len(test_loader.dataset))

    print("\nBATCH COUNTS")

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
    print("Testing batches:", len(test_loader))

    print("\nTESTING ONE TRAINING BATCH")

    batch = next(iter(train_loader))

    print("Input IDs shape:", tuple(batch["input_ids"].shape))
    print("Attention mask shape:", tuple(batch["attention_mask"].shape))
    print("Image shape:", tuple(batch["image"].shape))
    print("Labels shape:", tuple(batch["label"].shape))

    print("\nTensor data types:")

    print("Input IDs:", batch["input_ids"].dtype)
    print("Attention mask:", batch["attention_mask"].dtype)
    print("Images:", batch["image"].dtype)
    print("Labels:", batch["label"].dtype)

    print("\nLabels:", batch["label"].tolist())

    print("\nCUDA available:", torch.cuda.is_available())

    print("\nEXPANDED DATALOADER TEST COMPLETED")


if __name__ == "__main__":
    main()
