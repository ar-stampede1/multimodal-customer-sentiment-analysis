
import torch
from pathlib import Path

CHECKPOINT_DIR = Path("ml/saved_models/expanded")

files = [
    "best_multimodal_model_expanded.pth",
    "best_text_only_model_expanded.pth",
    "best_image_only_model_expanded.pth",
]

for filename in files:
    path = CHECKPOINT_DIR / filename

    print("\n" + "=" * 70)
    print(f"CHECKPOINT: {filename}")
    print("=" * 70)

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    print("Object type:", type(checkpoint))

    if isinstance(checkpoint, dict):
        print("Top-level keys:", list(checkpoint.keys()))

        for key, value in checkpoint.items():
            if isinstance(value, dict):
                print(f"\n{key}: dictionary")
                print("Number of entries:", len(value))
                print("First 10 parameter names:")

                for name in list(value.keys())[:10]:
                    tensor = value[name]
                    shape = tuple(tensor.shape) if hasattr(tensor, "shape") else "N/A"
                    print(f"  {name}: {shape}")

            elif hasattr(value, "shape"):
                print(f"{key}: Tensor with shape {tuple(value.shape)}")

            else:
                print(f"{key}: {value}")

    print()
