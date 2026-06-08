from __future__ import annotations

import torch


def get_device(device: str | torch.device | None = None) -> torch.device:
    """
    Resolve the torch device to use.

    Args:
        device: Optional explicit device.

    Returns:
        A `torch.device`
    """

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
