
import sys
from pathlib import Path

import torch

# Add backend directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"

sys.path.insert(0, str(BACKEND_DIR))

from app.services.model_loader import (
    load_models,
    DEVICE,
    CLASS_NAMES,
)


def main():

    print("\nTESTING BACKEND MODEL LOADING")

    models = load_models()

    print("\nMODEL VERIFICATION")
    print("-" * 50)

    for name, model in models.items():

        parameter_count = sum(
            parameter.numel()
            for parameter in model.parameters()
        )

        print(f"\nModel: {name}")
        print("Device:", next(model.parameters()).device)
        print("Parameters:", f"{parameter_count:,}")
        print("Training mode:", model.training)

        assert not model.training, (
            f"{name} is still in training mode"
        )

    print("\nDevice selected:", DEVICE)
    print("Sentiment classes:", CLASS_NAMES)

    print("\nALL MODEL LOADING TESTS PASSED!")


if __name__ == "__main__":
    main()
