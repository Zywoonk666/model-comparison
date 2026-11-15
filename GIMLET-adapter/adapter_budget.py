"""Parameter accounting for four token-routing strategies (no ML imports)."""

STRATEGIES = ("shared", "graph", "text", "separate")


def bottleneck_parameters(d_model, width):
    # LayerNorm (2d), down (dr+r), up (rd+d), including every bias.
    if d_model < 1 or width < 1:
        raise ValueError("Model dimension and bottleneck width must be positive")
    return (2 * d_model + 1) * width + 3 * d_model


def adapter_spec(d_model, reference_width, strategy, layers=1):
    if strategy not in STRATEGIES or layers < 1:
        raise ValueError("Invalid adapter strategy or layer count")
    budget = bottleneck_parameters(d_model, reference_width)
    if strategy == "separate":
        # Independent norms and up biases also count against the SAME ceiling.
        total_width = (budget - 6 * d_model) // (2 * d_model + 1)
        if total_width < 2:
            raise ValueError("Reference budget is too small for two independent adapters")
        graph_width = total_width // 2
        text_width = total_width - graph_width
        count = bottleneck_parameters(d_model, graph_width) + bottleneck_parameters(d_model, text_width)
    else:
        graph_width = reference_width if strategy != "text" else 0
        text_width = reference_width if strategy != "graph" else 0
        count = budget
    return dict(adapter_strategy=strategy, adapter_reference_dim=reference_width,
                adapter_graph_dim=graph_width, adapter_text_dim=text_width,
                adapter_parameters=count * layers, adapter_budget_parameters=budget * layers,
                adapter_budget_gap_fraction=(budget - count) / budget)
