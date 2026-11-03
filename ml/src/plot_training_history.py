
import json
from pathlib import Path

import matplotlib.pyplot as plt


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "ml" / "saved_models" / "expanded"

OUTPUT_DIR = (
    PROJECT_ROOT / "ml" / "evaluation" / "comparison"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


HISTORY_FILES = {
    "Text-only": (
        MODEL_DIR / "text_only_training_history_expanded.json"
    ),
    "Image-only": (
        MODEL_DIR / "image_only_training_history_expanded.json"
    ),
    "Multimodal": (
        MODEL_DIR / "multimodal_training_history_expanded.json"
    )
}


# --------------------------------------------------
# LOAD TRAINING HISTORIES
# --------------------------------------------------

def load_histories():

    histories = {}

    print("\nLoading training histories...")

    for model_name, file_path in HISTORY_FILES.items():

        if not file_path.exists():
            raise FileNotFoundError(file_path)

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as file:

            histories[model_name] = json.load(file)

        print(
            f"{model_name}: "
            f"{len(histories[model_name])} epochs loaded"
        )

    return histories


# --------------------------------------------------
# GET HISTORY VALUES
# --------------------------------------------------

def get_values(history, key):

    return [
        epoch[key]
        for epoch in history
    ]


# --------------------------------------------------
# PLOT TRAINING AND VALIDATION LOSS
# --------------------------------------------------

def plot_loss(histories):

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, history in histories.items():

        epochs = [
            item["epoch"]
            for item in history
        ]

        train_loss = get_values(
            history,
            "train_loss"
        )

        # The text and multimodal scripts use
        # "validation_loss"; image-only uses "val_loss".
        if "validation_loss" in history[0]:
            val_loss = get_values(
                history,
                "validation_loss"
            )
        else:
            val_loss = get_values(
                history,
                "val_loss"
            )

        ax.plot(
            epochs,
            train_loss,
            marker="o",
            label=f"{model_name} Training"
        )

        ax.plot(
            epochs,
            val_loss,
            marker="s",
            linestyle="--",
            label=f"{model_name} Validation"
        )

    ax.set_title("Training and Validation Loss Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.set_xticks(range(1, 6))
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "training_validation_loss.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nLoss graph saved:", output_path)


# --------------------------------------------------
# PLOT VALIDATION ACCURACY
# --------------------------------------------------

def plot_validation_accuracy(histories):

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, history in histories.items():

        epochs = [
            item["epoch"]
            for item in history
        ]

        if "validation_accuracy" in history[0]:
            accuracy = get_values(
                history,
                "validation_accuracy"
            )
        else:
            accuracy = get_values(
                history,
                "val_accuracy"
            )

        ax.plot(
            epochs,
            accuracy,
            marker="o",
            label=model_name
        )

    ax.set_title("Validation Accuracy Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")

    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path = OUTPUT_DIR / "validation_accuracy_comparison.png"

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Accuracy graph saved:", output_path)


# --------------------------------------------------
# PLOT VALIDATION MACRO F1
# --------------------------------------------------

def plot_validation_macro_f1(histories):

    fig, ax = plt.subplots(figsize=(10, 6))

    for model_name, history in histories.items():

        epochs = [
            item["epoch"]
            for item in history
        ]

        if "validation_macro_f1" in history[0]:
            macro_f1 = get_values(
                history,
                "validation_macro_f1"
            )
        else:
            macro_f1 = get_values(
                history,
                "val_macro_f1"
            )

        ax.plot(
            epochs,
            macro_f1,
            marker="o",
            label=model_name
        )

    ax.set_title("Validation Macro F1 Comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")

    ax.set_xticks(range(1, 6))
    ax.set_ylim(0, 1)

    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR / "validation_macro_f1_comparison.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Macro F1 graph saved:", output_path)


# --------------------------------------------------
# PRINT BEST VALIDATION RESULTS
# --------------------------------------------------

def print_best_epochs(histories):

    print("\n" + "=" * 75)
    print("BEST VALIDATION RESULTS")
    print("=" * 75)

    print(
        f"{'Model':<18}"
        f"{'Best Epoch':>15}"
        f"{'Best Macro F1':>20}"
    )

    print("-" * 55)

    for model_name, history in histories.items():

        if "validation_macro_f1" in history[0]:
            key = "validation_macro_f1"
        else:
            key = "val_macro_f1"

        best = max(
            history,
            key=lambda item: item[key]
        )

        print(
            f"{model_name:<18}"
            f"{best['epoch']:>15}"
            f"{best[key]:>20.4f}"
        )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("=" * 75)
    print("TRAINING HISTORY ANALYSIS")
    print("=" * 75)

    histories = load_histories()

    plot_loss(histories)

    plot_validation_accuracy(histories)

    plot_validation_macro_f1(histories)

    print_best_epochs(histories)

    print("\n" + "=" * 75)
    print("TRAINING HISTORY ANALYSIS COMPLETED")
    print("=" * 75)

    print("\nGraphs saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
