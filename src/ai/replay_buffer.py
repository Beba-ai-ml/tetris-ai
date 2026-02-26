"""
Replay buffer for afterstate V-learning.

Stores (afterstate, reward, next_afterstate, done) transitions.
No actions or masks needed — the afterstate framework handles
action selection via enumeration, not stored Q-values.
"""

from __future__ import annotations

import numpy as np
import torch


class AfterstateReplayBuffer:
    """Fixed-size circular buffer storing (afterstate, reward, next_afterstate, done)."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...] = (2, 20, 10),
    ) -> None:
        self.capacity = capacity
        self.size = 0
        self.pos = 0

        self.afterstates = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_afterstates = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(
        self,
        afterstate: np.ndarray,
        reward: float,
        next_afterstate: np.ndarray,
        done: bool,
    ) -> None:
        self.afterstates[self.pos] = afterstate.astype(np.uint8)
        self.next_afterstates[self.pos] = next_afterstate.astype(np.uint8)
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, ...]:
        indices = np.random.choice(self.size, size=batch_size, replace=False)

        afterstates_t = torch.from_numpy(
            self.afterstates[indices].astype(np.float32)
        ).to(device)
        rewards_t = torch.from_numpy(
            self.rewards[indices]
        ).unsqueeze(1).to(device)
        next_afterstates_t = torch.from_numpy(
            self.next_afterstates[indices].astype(np.float32)
        ).to(device)
        dones_t = torch.from_numpy(
            self.dones[indices]
        ).unsqueeze(1).to(device)

        return afterstates_t, rewards_t, next_afterstates_t, dones_t

    def __len__(self) -> int:
        return self.size
