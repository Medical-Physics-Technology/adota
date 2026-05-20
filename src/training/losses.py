"""Loss functions and adaptive weight balancers for ADoTA training.

This module is the single source of truth for the loss objectives used by
the training script. It exposes:

- :class:`LMSE`   - normalized mean squared error on the full dose volume.
- :class:`LPS`    - normalized MSE on the Integrated Depth Dose (IDD).
- :class:`LossLPD` - lateral profile difference at each depth slice.
- :class:`TwoObjectiveBalancer` - adaptive softmax weighting between two losses.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Numerical guard added to denominators so that empty-dose samples
# (all-zero ground truth) yield 0 instead of NaN. Tuned to be small
# relative to the post-normalization dose dynamic range.
_DEFAULT_EPS: float = 1e-8


class LMSE(nn.Module):
    """Normalized mean squared error.

    For each sample :math:`i` in a batch:

    .. math::
        \\mathrm{LMSE}_i = \\frac{\\sum (\\hat{y}_i - y_i)^2}{\\sum y_i^2}

    The final loss is the mean of the per-sample values across the batch.
    """

    def __init__(self, epsilon: float = _DEFAULT_EPS) -> None:
        """Initialize the loss.

        Args:
            epsilon: Small constant added to the denominator for numerical
                stability when the ground-truth dose is identically zero.
        """
        super().__init__()
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            y_pred: Predicted dose, shape ``(B, C, D, H, W)``.
            y_gt: Ground-truth dose, same shape as ``y_pred``.

        Returns:
            Scalar loss tensor on the same device as the inputs.
        """
        if y_pred.shape != y_gt.shape:
            raise ValueError(
                f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs "
                f"y_gt={tuple(y_gt.shape)}"
            )

        batch_size = y_pred.shape[0]
        num = ((y_pred - y_gt) ** 2).reshape(batch_size, -1).sum(dim=1)
        den = (y_gt**2).reshape(batch_size, -1).sum(dim=1) + self.epsilon
        return (num / den).mean()


class LPS(nn.Module):
    """Normalized MSE on the Integrated Depth Dose (IDD).

    The IDD of a dose volume is the lateral sum at each depth slice,
    weighted by the lateral pixel area:

    .. math::
        \\mathrm{IDD}(z) = \\Delta x \\Delta y \\sum_{i, j} D(z, i, j)

    The loss is the normalized MSE between the predicted and ground-truth
    IDD curves, computed per sample and averaged over the batch.
    """

    def __init__(
        self,
        dx: float = 2.0,
        dy: float = 2.0,
        epsilon: float = _DEFAULT_EPS,
    ) -> None:
        """Initialize the loss.

        Args:
            dx: Lateral pixel spacing along the width axis [mm].
            dy: Lateral pixel spacing along the height axis [mm].
            epsilon: Denominator stabilizer (see :class:`LMSE`).
        """
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.epsilon = epsilon

    def forward(self, y_pred: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            y_pred: Predicted dose, shape ``(B, C, D, H, W)``.
            y_gt: Ground-truth dose, same shape as ``y_pred``.

        Returns:
            Scalar loss tensor on the same device as the inputs.
        """
        if y_pred.shape != y_gt.shape:
            raise ValueError(
                f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs "
                f"y_gt={tuple(y_gt.shape)}"
            )

        pixel_area = self.dx * self.dy
        idd_pred = y_pred.sum(dim=(-2, -1)) * pixel_area  # (B, C, D)
        idd_gt = y_gt.sum(dim=(-2, -1)) * pixel_area

        batch_size = y_pred.shape[0]
        num = ((idd_pred - idd_gt) ** 2).reshape(batch_size, -1).sum(dim=1)
        den = (idd_gt**2).reshape(batch_size, -1).sum(dim=1) + self.epsilon
        return (num / den).mean()


class LossLPD(nn.Module):
    """Lateral Profile Difference loss.

    For each depth slice ``z`` and each sample of the batch, the location of
    the ground-truth maximum :math:`(i^\\ast, j^\\ast)` defines two lateral
    profiles:

    - horizontal: ``gt[..., i*, :]`` of length ``W``,
    - vertical:   ``gt[..., :, j*]`` of length ``H``.

    Each profile is normalized by its own ground-truth sum, and a normalized
    MSE is computed against the prediction sampled at the same indices.
    The two directional losses are averaged with equal weight.

    Only the first channel of the input is used.
    """

    def __init__(
        self,
        epsilon: float = 1e-3,
        clamp_max: float = 1.0,
    ) -> None:
        """Initialize the loss.

        Args:
            epsilon: Denominator stabilizer applied to the per-slice
                normalized MSE.
            clamp_max: Upper clamp applied to the per-slice loss before
                averaging. Caps pathological slices to a known range.
        """
        super().__init__()
        self.epsilon = epsilon
        self.clamp_max = clamp_max

    def forward(self, y_pred: torch.Tensor, y_gt: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            y_pred: Predicted dose, shape ``(B, C, D, H, W)``.
            y_gt: Ground-truth dose, same shape as ``y_pred``.

        Returns:
            Scalar loss tensor on the same device as the inputs.
        """
        if y_pred.shape != y_gt.shape:
            raise ValueError(
                f"Shape mismatch: y_pred={tuple(y_pred.shape)} vs "
                f"y_gt={tuple(y_gt.shape)}"
            )

        gt = y_gt[:, 0]
        pr = y_pred[:, 0]
        batch_size, depth, height, width = gt.shape

        flat_idx = gt.reshape(batch_size, depth, -1).argmax(dim=2)
        i_idx = flat_idx // width
        j_idx = flat_idx % width

        i_expand = i_idx.view(batch_size, depth, 1, 1).expand(
            batch_size, depth, 1, width
        )
        prof_gt_x = gt.gather(dim=2, index=i_expand).squeeze(2)
        prof_pr_x = pr.gather(dim=2, index=i_expand).squeeze(2)

        j_expand = j_idx.view(batch_size, depth, 1, 1).expand(
            batch_size, depth, height, 1
        )
        prof_gt_y = gt.gather(dim=3, index=j_expand).squeeze(3)
        prof_pr_y = pr.gather(dim=3, index=j_expand).squeeze(3)

        # Normalize each profile by its own per-slice ground-truth sum so
        # that the directional losses are dimensionless and comparable.
        norm_x = prof_gt_x.sum(dim=2, keepdim=True).clamp_min(_DEFAULT_EPS)
        norm_y = prof_gt_y.sum(dim=2, keepdim=True).clamp_min(_DEFAULT_EPS)
        prof_gt_x = prof_gt_x / norm_x
        prof_pr_x = prof_pr_x / norm_x
        prof_gt_y = prof_gt_y / norm_y
        prof_pr_y = prof_pr_y / norm_y

        mse_x = ((prof_gt_x - prof_pr_x) ** 2).sum(dim=2) / (
            prof_gt_x.pow(2).sum(dim=2) + self.epsilon
        )
        mse_y = ((prof_gt_y - prof_pr_y) ** 2).sum(dim=2) / (
            prof_gt_y.pow(2).sum(dim=2) + self.epsilon
        )

        mse_x = torch.nan_to_num(mse_x, nan=0.0).clamp(0.0, self.clamp_max)
        mse_y = torch.nan_to_num(mse_y, nan=0.0).clamp(0.0, self.clamp_max)

        return (0.5 * mse_x + 0.5 * mse_y).mean()


class TwoObjectiveBalancer:
    """Adaptive softmax weighting between two loss objectives.

    Given two scalar losses, the balancer computes their relative
    magnitudes and returns weights via a softmax over their logarithms.
    Optionally, it maintains an exponentially smoothed running estimate
    of the weights across calls.

    The returned weights are detached from the autograd graph: they are
    used to scale the losses but do not propagate gradients themselves.
    """

    def __init__(
        self,
        smoothing: float = 0.9,
        epsilon: float = _DEFAULT_EPS,
    ) -> None:
        """Initialize the balancer.

        Args:
            smoothing: Exponential smoothing factor in ``[0, 1]`` applied
                to the running weight estimate. ``0`` disables smoothing.
            epsilon: Floor added to magnitudes before taking the log,
                preventing ``log(0)``.
        """
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(f"smoothing must be in [0, 1]; got {smoothing}")
        self.smoothing = smoothing
        self.epsilon = epsilon
        self.running_weights: Optional[torch.Tensor] = None

    def get_weights(
        self,
        loss1: torch.Tensor,
        loss2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the next pair of objective weights.

        Args:
            loss1: First objective value (scalar tensor).
            loss2: Second objective value (scalar tensor).

        Returns:
            A pair ``(w1, w2)`` of detached scalar tensors that sum to 1.
        """
        magnitudes = torch.stack([loss1.detach(), loss2.detach()]) + self.epsilon
        weights = F.softmax(torch.log(magnitudes), dim=0).detach()

        if self.running_weights is None:
            self.running_weights = weights
        else:
            self.running_weights = (
                self.smoothing * self.running_weights + (1.0 - self.smoothing) * weights
            ).detach()

        return weights[0], weights[1]

    def reset(self) -> None:
        """Clear the running weight estimate.

        Useful when starting a new epoch or resuming from a checkpoint
        where the previous running state is no longer meaningful.
        """
        self.running_weights = None
