import torch
import torch.nn as nn


class MLPRegressor1D(nn.Module):
    """波形データを直接入力とする MLP 回帰モデル。

    (B, 1555) -> Linear -> BatchNorm1d -> ReLU -> Dropout -> ... -> Linear(1)
    """

    def __init__(
        self,
        in_features: int = 1555,
        hidden_dims: list[int] | None = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [1024, 512, 256]

        layers: list[nn.Module] = []
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1555) -> (B, 1)
        return self.net(x)


class ConvRegressor1D(nn.Module):
    """波形データを直接入力とする Conv1d 回帰モデル。

    (B, 1555) -> unsqueeze(1) -> (B, 1, 1555)
    -> [Conv1d + BN + ReLU + MaxPool] x3
    -> AdaptiveAvgPool -> Flatten -> Linear -> ReLU -> Dropout -> Linear(1)
    """

    def __init__(
        self,
        conv_channels: list[int] | None = None,
        kernel_sizes: list[int] | None = None,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        conv_layers: list[nn.Module] = []
        prev_channels = 1  # 入力は1チャンネル（波形をチャンネル方向に持たせる）
        for i in range(len(conv_channels)):
            conv_layers.append(
                nn.Conv1d(prev_channels, conv_channels[i], kernel_sizes[i])
            )
            conv_layers.append(nn.BatchNorm1d(conv_channels[i]))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool1d(2))
            prev_channels = conv_channels[i]

        self.conv = nn.Sequential(*conv_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(conv_channels[-1], conv_channels[-1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(conv_channels[-1], 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1555) -> (B, 1, 1555)
        x = x.unsqueeze(1)
        x = self.conv(x)       # (B, C, L)
        x = self.pool(x)       # (B, C, 1)
        x = x.squeeze(-1)      # (B, C)
        x = self.head(x)       # (B, 1)
        return x
