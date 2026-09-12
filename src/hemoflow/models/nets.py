"""Neural architectures. Torch-only; nothing else in the package imports this eagerly.

Two models, and the difference between them is the whole experiment:

* `NodeMLP` sees one surface node at a time. It can learn any pointwise function
  of the features and nothing else.
* `VesselUNet` sees the unwrapped `(arc x circumference)` field as a 2D image and
  convolves over it, so it can represent effects that depend on the
  neighbourhood - most importantly the post-stenotic recirculation zone, where
  shear at a node is governed by the geometry several diameters *upstream*.

If the U-Net does not beat the MLP, either the receptive field is too small or
the target has no spatial structure worth capturing. That is a real experimental
result either way, and `hemoflow bench` is what settles it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MixedPadConv2d(nn.Module):
    """Convolution that respects the topology of an unwrapped vessel.

    The unwrapped field is a cylinder, not a rectangle. The circumferential axis
    wraps - column 0 and column `n_theta - 1` are physical neighbours - while the
    axial axis has genuine inlet and outlet boundaries.

    Zero-padding both axes, which is what a stock `Conv2d` does, invents a
    discontinuity along the `theta = 0` seam. On a curved vessel the Dean effect
    puts a smooth circumferential wave in the target, and that artificial seam
    cuts straight through it, so the model learns a visible stripe of error
    exactly where the geometry is most interesting.

    So: circular padding around the circumference, replicate padding along the
    axis.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so padding stays symmetric")
        self.pad = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, arc, theta)
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        return self.conv(x)


def topology_aware_upsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear upsample that respects the cylinder, like the convolutions do.

    A plain `F.interpolate` clamps at the tensor edges, so it treats the two
    sides of the `theta = 0` seam as unrelated boundaries and blends each towards
    itself. That silently undoes the work `MixedPadConv2d` does: measured on a
    depth-2 net, the decoder alone reintroduced a ~30% discontinuity across the
    seam.

    Padding one cell with the correct topology before interpolating, then
    cropping the corresponding border back off, removes it.
    """
    _, _, h, w = x.shape
    target_h, target_w = size
    scale_h = max(target_h // h, 1)
    scale_w = max(target_w // w, 1)

    padded = F.pad(x, (1, 1, 0, 0), mode="circular")
    padded = F.pad(padded, (0, 0, 1, 1), mode="replicate")
    upsampled = F.interpolate(
        padded,
        size=(target_h + 2 * scale_h, target_w + 2 * scale_w),
        mode="bilinear",
        align_corners=False,
    )
    return upsampled[..., scale_h : scale_h + target_h, scale_w : scale_w + target_w]


class ConvBlock(nn.Module):
    """Two topology-aware convolutions with GroupNorm and GELU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = max(1, min(8, out_channels // 4))
        self.body = nn.Sequential(
            MixedPadConv2d(in_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            MixedPadConv2d(out_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class NodeMLP(nn.Module):
    """Pointwise MLP applied independently at every surface node.

    Deliberately has no access to neighbouring nodes. It is the control that
    isolates how much of the target is explained by local geometry alone.
    """

    def __init__(
        self, n_features: int, hidden_dims: tuple[int, ...] = (128, 128, 128), dropout: float = 0.0
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = n_features
        for width in hidden_dims:
            layers += [nn.Linear(prev, width), nn.LayerNorm(width), nn.GELU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = width
        layers.append(nn.Linear(prev, 1))
        self.body = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x `(batch, arc, theta, features)`. Returns `(batch, arc, theta)`."""
        batch, n_arc, n_theta, n_features = x.shape
        flat = x.reshape(-1, n_features)
        return self.body(flat).reshape(batch, n_arc, n_theta)


class VesselUNet(nn.Module):
    """U-Net over the unwrapped `(arc x circumference)` field.

    Args:
        n_features: input channels.
        base_channels: width of the first encoder stage; doubles each level.
        depth: number of downsampling stages. The grid must be divisible by
            `2**depth` in both axes.
    """

    def __init__(self, n_features: int, base_channels: int = 32, depth: int = 3) -> None:
        super().__init__()
        self.depth = depth
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        channels = base_channels
        prev = n_features
        encoder_channels: list[int] = []
        for _ in range(depth):
            self.encoders.append(ConvBlock(prev, channels))
            encoder_channels.append(channels)
            prev, channels = channels, channels * 2

        self.bottleneck = ConvBlock(prev, channels)
        prev = channels

        for skip_channels in reversed(encoder_channels):
            self.decoders.append(ConvBlock(prev + skip_channels, skip_channels))
            prev = skip_channels

        self.head = nn.Conv2d(prev, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x `(batch, arc, theta, features)`. Returns `(batch, arc, theta)`."""
        _, n_arc, n_theta, _ = x.shape
        stride = 2**self.depth
        if n_arc % stride or n_theta % stride:
            raise ValueError(
                f"grid {n_arc}x{n_theta} is not divisible by 2**depth={stride}; "
                "adjust geometry.n_arc / geometry.n_theta or model.depth"
            )

        h = x.permute(0, 3, 1, 2).contiguous()  # -> (batch, channels, arc, theta)

        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = F.max_pool2d(h, 2)

        h = self.bottleneck(h)

        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            h = topology_aware_upsample(h, size=skip.shape[-2:])
            h = decoder(torch.cat([h, skip], dim=1))

        return self.head(h).squeeze(1)


def build_network(name: str, n_features: int, **kwargs: object) -> nn.Module:
    """Instantiate an architecture by name."""
    if name == "mlp":
        return NodeMLP(
            n_features,
            hidden_dims=tuple(kwargs.get("hidden_dims", (128, 128, 128))),  # type: ignore[arg-type]
            dropout=float(kwargs.get("dropout", 0.0)),  # type: ignore[arg-type]
        )
    if name == "unet":
        return VesselUNet(
            n_features,
            base_channels=int(kwargs.get("base_channels", 32)),  # type: ignore[arg-type]
            depth=int(kwargs.get("depth", 3)),  # type: ignore[arg-type]
        )
    raise ValueError(f"unknown architecture {name!r}; expected 'mlp' or 'unet'")
