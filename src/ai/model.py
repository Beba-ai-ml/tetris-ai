"""
Dueling DQN CNN for placement-based Tetris.

Input:  (batch, 4, 20, 10) — board + current/next/held piece
Output: 80 Q-values (4 rotations × 10 columns × 2 [place / hold-place])
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TetrisCNN(nn.Module):
    """Dueling DQN with asymmetric stride preserving column resolution."""

    def __init__(
        self,
        input_channels: int = 4,
        board_height: int = 20,
        board_width: int = 10,
        num_actions: int = 80,
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

        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
