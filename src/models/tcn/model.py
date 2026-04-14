import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.activation(out + residual)


class FireTCNClassifier(nn.Module):
    def __init__(
        self,
        dyn_features: int,
        static_features: int,
        channels: list[int] | tuple[int, ...] = (32, 32, 32),
        kernel_size: int = 3,
        dropout: float = 0.1,
        static_hidden: int = 16,
    ):
        super().__init__()

        blocks = []
        in_channels = dyn_features
        for i, out_channels in enumerate(channels):
            blocks.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
            )
            in_channels = out_channels
        self.tcn = nn.Sequential(*blocks)

        self.static_encoder = nn.Sequential(
            nn.Linear(static_features, static_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(in_channels + static_hidden, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x_dyn: torch.Tensor, x_static: torch.Tensor) -> torch.Tensor:
        dyn = x_dyn.transpose(1, 2)
        dyn_encoded = self.tcn(dyn)[:, :, -1]
        static_encoded = self.static_encoder(x_static)
        combined = torch.cat([dyn_encoded, static_encoded], dim=1)
        return self.head(combined).squeeze(1)
