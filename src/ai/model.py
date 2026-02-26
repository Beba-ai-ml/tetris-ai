"""
Afterstate value network for placement-based Tetris.

Input:  (batch, 2, 20, 10) — board after placement + held piece
Output: scalar V(afterstate) — expected future return from this board state
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AfterstateNet(nn.Module):
    """CNN that evaluates afterstate board positions.

    Same conv backbone as the old TetrisCNN (proven architecture),
    but outputs a single scalar V(s') instead of 80 Q-values.
    No dueling decomposition (meaningless with scalar output).
    """

    def __init__(
        self,
        input_channels: int = 2,
        board_height: int = 20,
        board_width: int = 10,
    ) -> None:
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=(2, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        flat_size = 64 * (board_height // 4) * board_width  # 3200

        self.value_head = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        return self.value_head(x)  # (batch, 1)
