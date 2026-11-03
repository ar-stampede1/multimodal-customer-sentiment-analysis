
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from PIL import Image

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ML_SOURCE = PROJECT_ROOT / "ml" / "src"

if str(ML_SOURCE) not in sys.path:
    sys.path.insert(0, str(ML_SOURCE))

# --------------------------------------------------
# IMPORT PREPROCESSING FUNCTIONS
# --------------------------------------------------

from text_preprocessing import prepare_text
from image_preprocessing import IMAGE_TRANSFORM

from text_preprocessing import (
    load_tokenizer,
    prepare_text,
    tokenize_review,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

CLASS_NAMES = [
    "Very Dissatisfied",
    "Dissatisfied",
    "Neutral",
    "Satisfied",
    "Very Satisfied",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# TOKENIZER
# --------------------------------------------------

print("Loading sentiment tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Tokenizer loaded successfully!")


# --------------------------------------------------
# PREDICTION SERVICE
# --------------------------------------------------

class PredictionService:

    def __init__(self, models):
        self.tokenizer = load_tokenizer()

        self.models = models

        for model in self.models.values():
            model.eval()

        print("Prediction service initialized!")

    # --------------------------------------------------
    # TEXT PREPROCESSING
    # --------------------------------------------------

    def preprocess_text(self, review_title, review_text):

        record = {
            "review_title": review_title or "",
            "review_text": review_text or "",
        }

        text = prepare_text(record)

        if not text.strip():
            raise ValueError("Review text cannot be empty.")

        encoded = tokenize_review(text, self.tokenizer)

        return {
            "input_ids": encoded["input_ids"].unsqueeze(0).to(DEVICE),
            "attention_mask": encoded["attention_mask"].unsqueeze(0).to(DEVICE),
        }

    # --------------------------------------------------
    # IMAGE PREPROCESSING
    # --------------------------------------------------

    def preprocess_image(self, image):

        if image.mode != "RGB":
            image = image.convert("RGB")

        processed_image = IMAGE_TRANSFORM(image)

        return processed_image.unsqueeze(0).to(DEVICE)

    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    @torch.inference_mode()
    def predict(self, review_title="", review_text="", image=None):

        has_text = bool(
            (review_title or "").strip() or
            (review_text or "").strip()
        )

        if not has_text and image is None:
            raise ValueError(
                "Provide review text, an image, or both."
            )

        # Text + image: multimodal model
        if has_text and image is not None:

            model = self.models["multimodal"]

            text_inputs = self.preprocess_text(
                review_title,
                review_text,
            )

            image_tensor = self.preprocess_image(image)

            logits = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
                images=image_tensor,
            )

            model_used = "multimodal"

        # Text only
        elif has_text:

            model = self.models["text_only"]

            text_inputs = self.preprocess_text(
                review_title,
                review_text,
            )

            logits = model(
                input_ids=text_inputs["input_ids"],
                attention_mask=text_inputs["attention_mask"],
            )

            model_used = "text_only"

        # Image only
        else:

            model = self.models["image_only"]

            image_tensor = self.preprocess_image(image)

            logits = model(image_tensor)

            model_used = "image_only"

        # --------------------------------------------------
        # CONVERT LOGITS TO PROBABILITIES
        # --------------------------------------------------

        probabilities = F.softmax(logits, dim=1)[0]

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        confidence = float(
            probabilities[predicted_class].item()
        )

        probability_dict = {
            CLASS_NAMES[i]: float(probabilities[i].item())
            for i in range(len(CLASS_NAMES))
        }

        return {
            "sentiment": CLASS_NAMES[predicted_class],
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probabilities": probability_dict,
            "model_used": model_used,
            "image_used": image is not None,
        }
