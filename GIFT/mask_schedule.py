"""Epoch schedule for GIFT's training-time graph-to-text attention mask."""


def get_linear_mask_ratio(epoch, total_epochs=60, start_ratio=1.0, end_ratio=0.0):
    """Return a masking probability for a 1-based epoch, including both endpoints.

    With only one epoch, use start_ratio. Equal endpoints reproduce a constant
    mask. Evaluation keeps the encoder's existing deterministic layerwise mask.
    """
    if total_epochs < 1 or not 1 <= epoch <= total_epochs:
        raise ValueError("Require total_epochs >= 1 and 1 <= epoch <= total_epochs")
    if not 0.0 <= end_ratio <= start_ratio <= 1.0:
        raise ValueError("Require 0 <= mask_end_ratio <= mask_start_ratio <= 1")
    if total_epochs == 1:
        return float(start_ratio)
    if epoch == total_epochs:
        return float(end_ratio)
    progress = (epoch - 1) / (total_epochs - 1)
    return start_ratio + (end_ratio - start_ratio) * progress
