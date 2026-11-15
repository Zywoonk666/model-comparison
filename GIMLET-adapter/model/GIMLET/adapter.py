"""Residual bottleneck adapters for GIMLET few-shot tuning."""

import torch
import torch.nn as nn

from adapter_budget import adapter_spec


class BottleneckAdapter(nn.Module):
    """Residual bottleneck with optional graph-only or text-only routing.

    Shared preserves legacy parameter names and behavior. Routing describes
    direct residual updates; later attention can propagate between modalities.
    """

    def __init__(self, d_model, bottleneck_dim=64, strategy="shared"):
        super().__init__()
        if bottleneck_dim <= 0:
            raise ValueError("Adapter bottleneck_dim must be positive")
        if strategy not in ("shared", "graph", "text"):
            raise ValueError("Use SeparateAdapters for independent branches")
        self.strategy = strategy

        self.norm = nn.LayerNorm(d_model)
        self.down = nn.Linear(d_model, bottleneck_dim)
        self.activation = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, d_model)

        nn.init.xavier_uniform_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def adapt(self, hidden_states):
        correction = self.up(self.activation(self.down(self.norm(hidden_states))))
        return hidden_states + correction

    def forward(self, hidden_states, graph_length=None):
        if self.strategy == "shared":
            return self.adapt(hidden_states)
        validate_boundary(hidden_states, graph_length)
        graph, text = hidden_states[:, :graph_length], hidden_states[:, graph_length:]
        if self.strategy == "graph":
            graph = self.adapt(graph)
        else:
            text = self.adapt(text)
        return torch.cat((graph, text), dim=1)


def validate_boundary(hidden_states, graph_length):
    if hidden_states.ndim != 3 or not isinstance(graph_length, int) or not 0 <= graph_length <= hidden_states.shape[1]:
        raise ValueError("graph_length must identify the graph/text boundary in [batch, tokens, features]")


class SeparateAdapters(nn.Module):
    """Independent graph/text bottlenecks, with no shared trainable parameters."""

    def __init__(self, d_model, graph_dim, text_dim):
        super().__init__()
        self.graph = BottleneckAdapter(d_model, graph_dim)
        self.text = BottleneckAdapter(d_model, text_dim)

    def forward(self, hidden_states, graph_length=None):
        validate_boundary(hidden_states, graph_length)
        return torch.cat((self.graph(hidden_states[:, :graph_length]),
                          self.text(hidden_states[:, graph_length:])), dim=1)


def build_adapter(d_model, reference_width, strategy):
    spec = adapter_spec(d_model, reference_width, strategy)
    if strategy == "separate":
        return SeparateAdapters(d_model, spec["adapter_graph_dim"], spec["adapter_text_dim"])
    return BottleneckAdapter(d_model, reference_width, strategy)


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
