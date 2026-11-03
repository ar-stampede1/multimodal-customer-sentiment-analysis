
import torch
import torch.nn as nn

from transformers import AutoModel


MODEL_NAME = "distilbert-base-uncased"


class TextEncoder(nn.Module):

    def __init__(self, pretrained=True, freeze_encoder=False):

        super().__init__()

        print("Loading DistilBERT text encoder...")

        self.encoder = AutoModel.from_pretrained(MODEL_NAME)

        self.feature_dim = self.encoder.config.hidden_size

        # Optionally freeze pretrained encoder
        if freeze_encoder:
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

        print("DistilBERT loaded successfully!")
        print("Feature dimensions:", self.feature_dim)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Token-level hidden states: (batch, sequence, 768)
        hidden_states = outputs.last_hidden_state

        # Masked mean pooling:
        # Exclude padding tokens from the average.
        mask = attention_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )

        masked_hidden_states = hidden_states * mask

        summed_embeddings = masked_hidden_states.sum(dim=1)

        token_counts = mask.sum(dim=1).clamp(min=1e-9)

        pooled_features = summed_embeddings / token_counts

        return pooled_features


def main():

    print("=" * 65)
    print("DISTILBERT TEXT ENCODER TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nUsing device:", device)

    model = TextEncoder(
        pretrained=True,
        freeze_encoder=False
    ).to(device)

    model.eval()

    batch_size = 4
    sequence_length = 256

    input_ids = torch.randint(
        low=0,
        high=model.encoder.config.vocab_size,
        size=(batch_size, sequence_length),
        dtype=torch.long
    ).to(device)

    attention_mask = torch.ones(
        batch_size,
        sequence_length,
        dtype=torch.long
    ).to(device)

    with torch.no_grad():

        features = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    print("\nOUTPUT VERIFICATION")
    print("-" * 45)

    print("Input IDs shape:", tuple(input_ids.shape))
    print("Attention mask shape:", tuple(attention_mask.shape))
    print("Output feature shape:", tuple(features.shape))
    print("Output datatype:", features.dtype)

    print(
        "All values finite:",
        torch.isfinite(features).all().item()
    )

    print("\nDISTILBERT ENCODER TEST COMPLETED")


if __name__ == "__main__":
    main()
