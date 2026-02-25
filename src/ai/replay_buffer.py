"""
Pre-allocated replay buffer with action mask storage.

Stores valid_mask for next states so target Q-values can be properly
masked during Double DQN training.
"""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular buffer storing (s, a, r, s', done, next_mask)."""

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...] = (4, 20, 10),
        num_actions: int = 80,
    ) -> None:
        self.capacity = capacity
        self.size = 0
        self.pos = 0

        self.states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_states = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.next_masks = np.zeros((capacity, num_actions), dtype=np.uint8)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_valid_mask: np.ndarray,
    ) -> None:
        self.states[self.pos] = state.astype(np.uint8)
        self.next_states[self.pos] = next_state.astype(np.uint8)
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = float(done)
        self.next_masks[self.pos] = next_valid_mask.astype(np.uint8)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, ...]:
        indices = np.random.choice(self.size, size=batch_size, replace=False)

        states_t = torch.from_numpy(self.states[indices].astype(np.float32)).to(device)
        actions_t = torch.from_numpy(self.actions[indices]).unsqueeze(1).to(device)
        rewards_t = torch.from_numpy(self.rewards[indices]).unsqueeze(1).to(device)
        next_states_t = torch.from_numpy(self.next_states[indices].astype(np.float32)).to(device)
        dones_t = torch.from_numpy(self.dones[indices]).unsqueeze(1).to(device)
        next_masks_t = torch.from_numpy(self.next_masks[indices].astype(bool)).to(device)

        return states_t, actions_t, rewards_t, next_states_t, dones_t, next_masks_t

    def __len__(self) -> int:
        return self.size
