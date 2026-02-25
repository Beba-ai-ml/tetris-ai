"""
Double DQN agent with action masking for placement-based Tetris.

Key features:
  - Dueling Double DQN
  - Invalid action masking (out-of-bounds placements → -inf Q-value)
  - Soft target updates (polyak averaging over state_dict)
"""

from __future__ import annotations

import pathlib
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.ai.model import TetrisCNN
from src.ai.replay_buffer import ReplayBuffer


class DoubleDQNAgent:
    """Double DQN agent with action masking for placement-based Tetris."""

    def __init__(
        self,
        input_channels: int = 4,
        board_height: int = 20,
        board_width: int = 10,
        num_actions: int = 80,
        gamma: float = 0.95,
        learning_rate: float = 2.5e-4,
        replay_buffer_size: int = 100_000,
        tau: float = 0.005,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.gamma = gamma
        self.tau = tau
        self.num_actions = num_actions

        self.policy_net = TetrisCNN(
            input_channels, board_height, board_width, num_actions
        ).to(self.device)

        self.target_net = TetrisCNN(
            input_channels, board_height, board_width, num_actions
        ).to(self.device)

        obs_shape = (input_channels, board_height, board_width)
        self.replay_buffer = ReplayBuffer(replay_buffer_size, obs_shape, num_actions)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()

        self.sync_target_net(hard=True)
        self.target_net.eval()

    def select_action(self, state: np.ndarray, epsilon: float, valid_mask: np.ndarray) -> int:
        """Select action using epsilon-greedy with masking.

        Args:
            state: Observation (C, H, W).
            epsilon: Exploration probability.
            valid_mask: Boolean mask of valid actions.

        Returns:
            Integer action (0 to num_actions-1).
        """
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return 0  # Fallback (game should be over)

        if random.random() < epsilon:
            return int(np.random.choice(valid_indices))

        self.policy_net.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t).squeeze(0)
            # Mask invalid actions to -inf
            invalid = torch.tensor(~valid_mask, dtype=torch.bool, device=self.device)
            q_values[invalid] = float('-inf')
        self.policy_net.train()
        return int(q_values.argmax().item())

    def train_step(self, batch_size: int) -> tuple[float, float, float, float]:
        """Train with Double DQN + action masking on next states.

        Returns:
            (loss, grad_norm, mean_q, max_q) for logging.
        """
        states, actions, rewards, next_states, dones, next_masks = self.replay_buffer.sample(
            batch_size, self.device
        )

        all_q = self.policy_net(states)
        current_q = all_q.gather(1, actions)

        with torch.no_grad():
            # Policy net selects best VALID next action
            next_q_policy = self.policy_net(next_states)
            next_q_policy[~next_masks] = float('-inf')
            best_next_actions = next_q_policy.argmax(dim=1, keepdim=True)

            # Target net evaluates those actions
            next_q = self.target_net(next_states).gather(1, best_next_actions)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.sync_target_net(hard=False)

        mean_q = all_q.mean().item()
        max_q = all_q.max().item()

        return loss.item(), grad_norm.item(), mean_q, max_q

    def sync_target_net(self, hard: bool = False) -> None:
        """Sync target via hard copy or polyak averaging (includes BN buffers)."""
        if hard:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            target_sd = self.target_net.state_dict()
            policy_sd = self.policy_net.state_dict()
            for key in target_sd:
                target_sd[key] = self.tau * policy_sd[key] + (1.0 - self.tau) * target_sd[key]
            self.target_net.load_state_dict(target_sd)

    def save(self, path: str | pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str | pathlib.Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
