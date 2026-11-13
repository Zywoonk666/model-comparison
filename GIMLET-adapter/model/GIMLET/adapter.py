"""Residual bottleneck adapters for GIMLET few-shot tuning."""

import torch.nn as nn


class BottleneckAdapter(nn.Module):
    """Apply the same feature transformation to every graph/text token."""

    def __init__(self, d_model, bottleneck_dim=64):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("Adapter bottleneck_dim must be positive")

        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, d_model)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden_states):
        correction = self.up(self.activation(self.down(self.norm(hidden_states))))
        return hidden_states + correction


def configure_adapter_training(model):
    """Train adapters and the LM head, preserving T5's existing weight tying."""
    adapters = getattr(model.encoder, "adapters", None)
    if adapters is None or len(adapters) == 0:
        raise ValueError("Adapter tuning requires an encoder with adapters")

    model.requires_grad_(False)
    adapters.requires_grad_(True)
    model.lm_head.requires_grad_(True)
    # model.parameters() deduplicates weights shared by the LM head/embeddings.
    return [param for param in model.parameters() if param.requires_grad]
