
import torch
import torch.nn as nn
from transformers import AutoModel


class TextOnlyModel(nn.Module):

    def __init__(
        self,
        model_name="distilbert-base-uncased",
        num_classes=5,
        dropout=0.3
    ):

        super().__init__()

        print("Loading DistilBERT text encoder...")

        self.encoder = AutoModel.from_pretrained(model_name)

        self.feature_dim = self.encoder.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

        print("Text-only model initialized!")
        print("Feature dimensions:", self.feature_dim)
        print("Output classes:", num_classes)

    def forward(self, input_ids, attention_mask):

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state

        # Mask padding tokens before mean pooling.
        mask = attention_mask.unsqueeze(-1).to(
            hidden_states.dtype
        )

        summed_embeddings = (hidden_states * mask).sum(dim=1)

        token_counts = mask.sum(dim=1).clamp(min=1e-9)

        pooled_features = summed_embeddings / token_counts

        logits = self.classifier(pooled_features)

        return logits
