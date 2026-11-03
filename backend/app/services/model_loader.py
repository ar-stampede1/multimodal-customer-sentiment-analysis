import sys
from pathlib import Path

import torch

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

# model_loader.py:
# project/backend/app/services/model_loader.py

PROJECT_ROOT = Path(__file__).resolve().parents[3]

ML_SOURCE = PROJECT_ROOT / "ml" / "src"

CHECKPOINT_DIR = (
    PROJECT_ROOT / "ml" / "saved_models" / "expanded"
)

# Register ML source directory
if str(ML_SOURCE) not in sys.path:
    sys.path.insert(0, str(ML_SOURCE))

# Verify paths
print("Project root:", PROJECT_ROOT)
print("ML source:", ML_SOURCE)
print("ML source exists:", ML_SOURCE.exists())
print("Models directory exists:", (ML_SOURCE / "models").exists())

# --------------------------------------------------
# MODEL IMPORTS
# --------------------------------------------------

from models.fusion_model import MultimodalFusionModel
from models.text_only_model import TextOnlyModel
from models.image_only_model import ImageOnlyModel


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

NUM_CLASSES = 5

CLASS_NAMES = [
    "Very Dissatisfied",
    "Dissatisfied",
    "Neutral",
    "Satisfied",
    "Very Satisfied",
]

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# CHECKPOINT LOADING
# --------------------------------------------------

def load_checkpoint(filename):

    checkpoint_path = CHECKPOINT_DIR / filename

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"model_state_dict missing in {filename}"
        )

    return checkpoint


# --------------------------------------------------
# LOAD ALL TRAINED MODELS
# --------------------------------------------------

def load_models():

    print("=" * 65)
    print("LOADING TRAINED SENTIMENT MODELS")
    print("=" * 65)

    print("\nSelected device:", DEVICE)

    models = {}

    # 1. Multimodal model
    print("\n[1/3] Loading multimodal model...")

    multimodal_checkpoint = load_checkpoint(
        "best_multimodal_model_expanded.pth"
    )

    multimodal_model = MultimodalFusionModel(
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_encoders=False
    )

    multimodal_model.load_state_dict(
        multimodal_checkpoint["model_state_dict"],
        strict=True
    )

    multimodal_model.to(DEVICE)
    multimodal_model.eval()

    models["multimodal"] = multimodal_model

    print("Multimodal checkpoint loaded successfully!")
    print(
        "Best validation Macro F1:",
        round(
            multimodal_checkpoint[
                "best_validation_macro_f1"
            ],
            4
        )
    )

    # 2. Text-only model
    print("\n[2/3] Loading text-only model...")

    text_checkpoint = load_checkpoint(
        "best_text_only_model_expanded.pth"
    )

    text_model = TextOnlyModel(
        num_classes=NUM_CLASSES
    )

    text_model.load_state_dict(
        text_checkpoint["model_state_dict"],
        strict=True
    )

    text_model.to(DEVICE)
    text_model.eval()

    models["text_only"] = text_model

    print("Text-only checkpoint loaded successfully!")
    print(
        "Best validation Macro F1:",
        round(
            text_checkpoint[
                "best_validation_macro_f1"
            ],
            4
        )
    )

    # 3. Image-only model
    print("\n[3/3] Loading image-only model...")

    image_checkpoint = load_checkpoint(
        "best_image_only_model_expanded.pth"
    )

    image_model = ImageOnlyModel(
        num_classes=NUM_CLASSES,
        freeze_encoder=False
    )

    image_model.load_state_dict(
        image_checkpoint["model_state_dict"],
        strict=True
    )

    image_model.to(DEVICE)
    image_model.eval()

    models["image_only"] = image_model

    print("Image-only checkpoint loaded successfully!")
    print(
        "Best validation Macro F1:",
        round(
            image_checkpoint["best_val_macro_f1"],
            4
        )
    )

    print("\n" + "=" * 65)
    print("ALL THREE MODELS LOADED SUCCESSFULLY")
    print("=" * 65)

    return models
