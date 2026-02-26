"""
Placement-based Tetris environment for RL with afterstate evaluation.

Instead of per-frame actions (move/rotate), the agent chooses WHERE to place
each piece: (rotation, column). The piece is instantly hard-dropped to that
position. One decision per piece = immediate reward = no credit assignment problem.

Action space: 80 discrete actions.
  Actions 0-39:  Place current piece at (rotation, column).
  Actions 40-79: HOLD first, then place the resulting piece at (rotation, column).

Afterstate observation: (2, 20, 10) — board after placement + held piece.
  Ch0: Board grid after placement + line clear (binary)
  Ch1: Held piece shape (top-left corner, zeros if None)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.game.board import Board
from src.game.tetris import TetrisGame


# Line clear rewards
LINE_REWARDS = {0: 0.0, 1: 5.0, 2: 15.0, 3: 40.0, 4: 150.0}


def calculate_reward(
    lines_cleared: int,
    holes_delta: int,
    height_delta: int,
    bumpiness_delta: int,
    game_over: bool,
    piece_locked: bool = False,
    new_holes_from_lock: int = 0,
    holes_weight: float = 0.3,
    height_weight: float = 0.03,
    bumpiness_weight: float = 0.01,
    game_over_penalty: float = 35.0,
) -> float:
    """Calculate shaped reward for a placement."""
    if game_over:
        return -game_over_penalty

    reward = LINE_REWARDS.get(lines_cleared, 0.0)
    reward -= holes_weight * holes_delta
    reward -= height_weight * height_delta
    reward -= bumpiness_weight * bumpiness_delta

    if piece_locked:
        if new_holes_from_lock == 0:
            reward += 0.5  # clean placement bonus

    if not game_over and piece_locked:
        reward += 0.1  # survival bonus

    return reward


class TetrisEnv:
    """Placement-based Tetris environment with hold action.

    Agent chooses (rotation, column) per piece, optionally holding first.
    Piece is hard-dropped instantly. Every step = one piece placed.

    Actions 0-39:  place current piece (rotation=action//10, column=action%10)
    Actions 40-79: hold, then place resulting piece (same encoding, offset by 40)

    Attributes:
        game: The underlying TetrisGame instance.
        action_space_n: Number of discrete actions (80).
        observation_shape: Shape of observations (4, 20, 10).
    """

    VISIBLE_HEIGHT = 20
    NUM_CHANNELS = 4
    NUM_ROTATIONS = 4

    def __init__(
        self,
        board_width: int = 10,
        board_height: int = 30,
        **kwargs: Any,
    ) -> None:
        self.board_width = board_width
        self.board_height = board_height
        self.game = TetrisGame(board_width, board_height)
        self._place_actions = self.NUM_ROTATIONS * board_width  # 40
        self.action_space_n: int = self._place_actions * 2      # 80 (place + hold-place)
        self._buffer_rows = board_height - self.VISIBLE_HEIGHT
        self.observation_shape: tuple[int, int, int] = (
            self.NUM_CHANNELS, self.VISIBLE_HEIGHT, board_width
        )

        self._reward_params = {
            "holes_weight": kwargs.get("holes_weight", 0.3),
            "height_weight": kwargs.get("height_weight", 0.03),
            "bumpiness_weight": kwargs.get("bumpiness_weight", 0.01),
            "game_over_penalty": kwargs.get("game_over_penalty", 35.0),
        }

        self._prev_holes: int = 0
        self._prev_height: int = 0
        self._prev_bumpiness: int = 0

    def reset(self) -> np.ndarray:
        """Reset game and return initial observation."""
        self.game.reset()
        self._prev_holes, self._prev_height, self._prev_bumpiness = self._get_board_metrics()
        return self._build_observation()

    def _check_piece_fits(self, piece: dict, rotation: int, column: int) -> bool:
        """Check if a piece fits at (rotation, column) at spawn height."""
        shape = piece["rotations"][rotation]
        piece_height = shape.shape[0]
        spawn_y = self._buffer_rows - piece_height + 1
        return self.game.board.is_valid_position(piece, column, spawn_y, rotation)

    def get_valid_mask(self) -> np.ndarray:
        """Return boolean mask of valid actions (80,).

        Actions 0-39:  valid if current piece fits at (rotation, column).
        Actions 40-79: valid if hold is allowed AND the post-hold piece fits.
        """
        mask = np.zeros(self.action_space_n, dtype=bool)
        piece = self.game.current_piece
        if piece is None:
            return mask

        # Actions 0-39: place current piece
        for action in range(self._place_actions):
            rotation = action // self.board_width
            column = action % self.board_width
            if self._check_piece_fits(piece, rotation, column):
                mask[action] = True

        # Actions 40-79: hold + place
        if self.game.can_hold:
            # Peek at what piece we'd have after holding (no state mutation)
            if self.game.held_piece is not None:
                post_hold_piece = self.game.held_piece
            else:
                post_hold_piece = self.game.next_piece

            if post_hold_piece is not None:
                for action in range(self._place_actions):
                    rotation = action // self.board_width
                    column = action % self.board_width
                    if self._check_piece_fits(post_hold_piece, rotation, column):
                        mask[self._place_actions + action] = True

        return mask

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Place a piece (optionally holding first) and hard drop.

        Args:
            action: Integer 0-79.
                0-39:  place current piece (rotation=a//10, col=a%10)
                40-79: hold first, then place (same encoding, offset by 40)

        Returns:
            (observation, reward, done, info) where info contains 'valid_mask'.
        """
        if self.game.game_over:
            mask = self.get_valid_mask()
            return self._build_observation(), 0.0, True, self._build_info(0, mask)

        # Handle hold action
        is_hold = action >= self._place_actions
        if is_hold:
            action -= self._place_actions
            self._perform_hold()

        pre_holes = self._prev_holes
        pre_height = self._prev_height
        pre_bumpiness = self._prev_bumpiness

        rotation = action // self.board_width
        column = action % self.board_width

        piece = self.game.current_piece

        # Set piece rotation and column
        self.game.current_rotation = rotation
        self.game.current_x = column

        # Move piece to spawn position for this rotation
        shape = piece["rotations"][rotation]
        piece_height = shape.shape[0]
        self.game.current_y = self._buffer_rows - piece_height + 1

        # Hard drop: find lowest valid row
        while self.game.board.is_valid_position(
            piece, self.game.current_x, self.game.current_y + 1, rotation
        ):
            self.game.current_y += 1

        # Lock piece
        holes_before = self.game.board.get_holes()
        lines_cleared = self.game._lock_piece()
        self.game._lines_cleared_last = lines_cleared
        new_holes = max(0, self.game.board.get_holes() - holes_before)

        # Scoring
        score_gained = self.game._calculate_score(lines_cleared)
        self.game.score += score_gained
        if lines_cleared > 0:
            self.game.total_lines += lines_cleared
            self.game._update_level()

        # Spawn next piece
        done = False
        if not self.game.game_over:
            if not self.game._spawn_piece():
                self.game.game_over = True
                done = True

        # Check if new piece has any valid placements
        if not done:
            next_mask = self.get_valid_mask()
            if not np.any(next_mask):
                self.game.game_over = True
                done = True
        else:
            next_mask = np.zeros(self.action_space_n, dtype=bool)

        if done:
            next_mask = np.zeros(self.action_space_n, dtype=bool)

        # Compute reward
        post_holes, post_height, post_bumpiness = self._get_board_metrics()
        reward = calculate_reward(
            lines_cleared=lines_cleared,
            holes_delta=post_holes - pre_holes,
            height_delta=post_height - pre_height,
            bumpiness_delta=post_bumpiness - pre_bumpiness,
            game_over=done,
            piece_locked=True,
            new_holes_from_lock=new_holes,
            **self._reward_params,
        )

        self._prev_holes = post_holes
        self._prev_height = post_height
        self._prev_bumpiness = post_bumpiness

        obs = self._build_observation()
        info = self._build_info(lines_cleared, next_mask)

        return obs, reward, done, info

    def _perform_hold(self) -> None:
        """Execute the hold swap without placing — mutates game state."""
        old_current = self.game.current_piece
        if self.game.held_piece is None:
            # No held piece: store current, pull next from bag
            self.game.held_piece = old_current
            self.game.current_piece = self.game.next_piece
            self.game.next_piece = self.game._next_from_bag()
        else:
            # Swap current and held
            self.game.current_piece, self.game.held_piece = (
                self.game.held_piece, self.game.current_piece
            )
        self.game.can_hold = False

    def _build_observation(self) -> np.ndarray:
        """Build 4-channel observation cropped to visible rows.

        Returns:
            Float32 array of shape (4, 20, 10).
        """
        h = self.VISIBLE_HEIGHT
        w = self.board_width
        buf = self._buffer_rows
        obs = np.zeros((self.NUM_CHANNELS, h, w), dtype=np.float32)

        # Channel 0: Board grid
        grid = self.game.board.get_grid()
        obs[0] = (grid[buf:buf + h] != 0).astype(np.float32)

        # Channel 1: Current piece shape (top-left corner)
        piece = self.game.current_piece
        if piece is not None:
            shape = piece["rotations"][0]  # Spawn rotation shape for identification
            self._render_piece_shape(obs[1], shape)

        # Channel 2: Next piece shape
        if self.game.next_piece is not None:
            shape = self.game.next_piece["rotations"][0]
            self._render_piece_shape(obs[2], shape)

        # Channel 3: Held piece shape
        if self.game.held_piece is not None:
            shape = self.game.held_piece["rotations"][0]
            self._render_piece_shape(obs[3], shape)

        return obs

    def _render_piece_shape(self, channel: np.ndarray, shape: np.ndarray) -> None:
        """Render a piece shape in top-left corner of a channel."""
        rows, cols = shape.shape
        channel[:rows, :cols] = (shape != 0).astype(np.float32)

    def _get_board_metrics(self) -> tuple[int, int, int]:
        """Read current board metrics."""
        return (
            self.game.board.get_holes(),
            self.game.board.get_aggregate_height(),
            self.game.board.get_bumpiness(),
        )

    def _build_info(self, lines_cleared: int, valid_mask: np.ndarray) -> dict[str, Any]:
        """Build info dict including valid_mask for next state."""
        return {
            "score": self.game.score,
            "level": self.game.level,
            "lines_cleared": lines_cleared,
            "total_lines": self.game.total_lines,
            "holes": self._prev_holes,
            "height": self._prev_height,
            "bumpiness": self._prev_bumpiness,
            "valid_mask": valid_mask,
        }

    # ── Afterstate Methods ────────────────────────────────────────────────

    def get_current_afterstate_obs(self) -> np.ndarray:
        """Build afterstate obs from current board state (after step / after reset)."""
        return self._build_afterstate_obs(self.game.board.grid, self.game.held_piece)

    def get_afterstates(self) -> list[tuple[int, np.ndarray, float]]:
        """Enumerate all valid placements, simulate each, return afterstate info.

        For each valid action:
          1. Copy the board grid
          2. Simulate piece placement (hard drop + lock + line clear)
          3. Compute immediate reward from delta metrics
          4. Build 2-channel afterstate observation

        Returns:
            List of (action_idx, afterstate_obs, immediate_reward).
        """
        mask = self.get_valid_mask()
        pre_holes, pre_height, pre_bump = self._get_board_metrics()
        results = []

        for action_idx in np.where(mask)[0]:
            temp_board, lines, new_holes, held_after = self._simulate_placement(
                int(action_idx)
            )

            post_holes = temp_board.get_holes()
            post_height = temp_board.get_aggregate_height()
            post_bump = temp_board.get_bumpiness()

            reward = calculate_reward(
                lines_cleared=lines,
                holes_delta=post_holes - pre_holes,
                height_delta=post_height - pre_height,
                bumpiness_delta=post_bump - pre_bump,
                game_over=False,
                piece_locked=True,
                new_holes_from_lock=new_holes,
                **self._reward_params,
            )

            obs = self._build_afterstate_obs(temp_board.grid, held_after)
            results.append((int(action_idx), obs, reward))

        return results

    def _simulate_placement(
        self, action_idx: int
    ) -> tuple[Board, int, int, dict | None]:
        """Simulate a single placement without mutating env/game state.

        Creates a temporary Board copy, places the appropriate piece
        (handling hold logic), hard drops, locks, and clears lines.

        Args:
            action_idx: Action index 0-79.

        Returns:
            (temp_board, lines_cleared, new_holes, held_piece_after)
        """
        temp_board = Board(self.board_width, self.board_height)
        temp_board.grid = self.game.board.grid.copy()

        is_hold = action_idx >= self._place_actions
        local_action = action_idx % self._place_actions
        rotation = local_action // self.board_width
        column = local_action % self.board_width

        if is_hold:
            if self.game.held_piece is not None:
                piece = self.game.held_piece
            else:
                piece = self.game.next_piece
            held_after = self.game.current_piece
        else:
            piece = self.game.current_piece
            held_after = self.game.held_piece

        # Hard drop: find lowest valid y
        shape = piece["rotations"][rotation]
        piece_height = shape.shape[0]
        y = self._buffer_rows - piece_height + 1
        while temp_board.is_valid_position(piece, column, y + 1, rotation):
            y += 1

        # Lock piece and clear lines
        holes_before = temp_board.get_holes()
        temp_board.place_piece(piece, column, y, rotation)
        lines = temp_board.clear_lines()
        new_holes = max(0, temp_board.get_holes() - holes_before)

        return temp_board, lines, new_holes, held_after

    def _build_afterstate_obs(
        self, board_grid: np.ndarray, held_piece: dict | None
    ) -> np.ndarray:
        """Build 2-channel afterstate observation.

        Ch0: Board grid after placement (binary, visible rows only)
        Ch1: Held piece shape (top-left corner, zeros if None)

        Args:
            board_grid: Full board grid (height x width).
            held_piece: Held piece dict or None.

        Returns:
            Float32 array of shape (2, 20, 10).
        """
        h = self.VISIBLE_HEIGHT
        w = self.board_width
        buf = self._buffer_rows
        obs = np.zeros((2, h, w), dtype=np.float32)

        obs[0] = (board_grid[buf : buf + h] != 0).astype(np.float32)

        if held_piece is not None:
            shape = held_piece["rotations"][0]
            rows, cols = shape.shape
            obs[1, :rows, :cols] = (shape != 0).astype(np.float32)

        return obs
