from __future__ import annotations

from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor


class DenoiserModel(nn.Module, ABC):
    """
    Standard interface for diffusion denoiser models.
    """

    @abstractmethod
    def forward(
        self,
        x_t: Tensor,
        t: Tensor,
        slot_valid_mask: Tensor,
        slot_pin_mask: Tensor,
    ) -> Tensor:
        """
        Predict the denoising target for one diffusion step.

        Args:
            x_t: Noised input tokens with shape `(B, L, D)`.
            t: Diffusion timesteps with shape `(B,)`.
            slot_valid_mask: Valid-slot mask with shape `(B, L)`.
            slot_pin_mask: Pinned-slot mask with shape `(B, L)`.

        Returns:
            Tensor with the same shape as `x_t`.
        """
        
        raise NotImplementedError
