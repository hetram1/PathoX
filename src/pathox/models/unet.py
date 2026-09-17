from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """Two convolution blocks with GroupNorm and GELU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        groups: int = 8,
    ) -> None:
        super().__init__()

        group_count = min(groups, out_channels)

        while out_channels % group_count != 0:
            group_count -= 1

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                group_count,
                out_channels,
            ),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(
                group_count,
                out_channels,
            ),
            nn.GELU(),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    """Downsampling block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )

        self.conv = DoubleConv(
            out_channels + skip_channels,
            out_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = self.up(x)

        if x.shape[-2:] != skip.shape[-2:]:
            x = nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        x = torch.cat(
            [skip, x],
            dim=1,
        )

        return self.conv(x)


class UNet(nn.Module):
    """Compact U-Net for six-class pathology segmentation."""

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 6,
        base_channels: int = 16,
    ) -> None:
        super().__init__()

        if in_channels <= 0:
            raise ValueError("in_channels must be positive")

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than 1")

        if base_channels <= 0:
            raise ValueError("base_channels must be positive")

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16

        self.stem = DoubleConv(
            in_channels,
            c1,
        )

        self.down1 = DownBlock(
            c1,
            c2,
        )

        self.down2 = DownBlock(
            c2,
            c3,
        )

        self.down3 = DownBlock(
            c3,
            c4,
        )

        self.down4 = DownBlock(
            c4,
            c5,
        )

        self.up1 = UpBlock(
            c5,
            c4,
            c4,
        )

        self.up2 = UpBlock(
            c4,
            c3,
            c3,
        )

        self.up3 = UpBlock(
            c3,
            c2,
            c2,
        )

        self.up4 = UpBlock(
            c2,
            c1,
            c1,
        )

        self.head = nn.Conv2d(
            c1,
            num_classes,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.head(x)
