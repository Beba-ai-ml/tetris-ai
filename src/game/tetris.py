"""
Game orchestrator — game loop, scoring, levels, and 7-bag randomizer.

This module ties together the Board and Piece definitions into a full
Tetris game with proper scoring (NES-style), hold mechanics, SRS rotation
with wall kicks, gravity, lock delay, and level progression.
"""

from __future__ import annotations

import enum
import random
from typing import Any

import numpy as np

from src.game.board import Board
from src.game.pieces import PIECE_TYPES, get_kick_offsets


class Action(enum.IntEnum):
    """Discrete action space for the Tetris game."""
    LEFT = 0
    RIGHT = 1
    ROTATE_CW = 2
    ROTATE_CCW = 3
    SOFT_DROP = 4
    HARD_DROP = 5
    HOLD = 6
    NOOP = 7


# NES-style scoring table: index = lines cleared (1-4)
SCORE_TABLE: dict[int, int] = {
    1: 40,
    2: 100,
    3: 300,
    4: 1200,
}

# Gravity speed table: frames per drop at each level (decreases with level)
# Based on NES Tetris. Levels beyond table use minimum of 1 frame.
GRAVITY_TABLE: list[int] = [
    48, 43, 38, 33, 28,   # levels 0-4
    23, 18, 13,  8,  6,   # levels 5-9
     5,  5,  5,  4,  4,   # levels 10-14
     4,  3,  3,  3,  2,   # levels 15-19
     2,  2,  2,  2,  2,   # levels 20-24
     2,  2,  2,  2,  1,   # levels 25-29
]


class TetrisGame:
    """Full Tetris game with SRS rotation, hold, scoring, and 7-bag randomizer.

    Attributes:
        board: The game board.
        score: Current score.
        level: Current level (starts at 0).
        total_lines: Total lines cleared since game start.
        current_piece: Currently active piece dict.
        current_x: Column position of active piece's top-left corner.
        current_y: Row position of active piece's top-left corner.
        current_rotation: Rotation state of active piece (0-3).
        next_piece: Next piece dict to be spawned.
        held_piece: Currently held piece dict, or None.
        can_hold: Whether the player can hold (resets each new piece).
        game_over: Whether the game has ended.
    """

    def __init__(self, board_width: int = 10, board_height: int = 30) -> None:
        """Initialize a new Tetris game.

        Args:
            board_width: Board width in columns.
            board_height: Board height in rows (including buffer zone).
        """
        self.board = Board(board_width, board_height)
        self.score: int = 0
        self.level: int = 0
        self.total_lines: int = 0
        self.current_piece: dict | None = None
        self.current_x: int = 0
        self.current_y: int = 0
        self.current_rotation: int = 0
        self.next_piece: dict | None = None
        self.held_piece: dict | None = None
        self.can_hold: bool = True
        self.game_over: bool = False

        # Internal state
        self._bag: list[dict] = []
        self._lock_delay_counter: int = 0
        self._gravity_counter: int = 0

    def reset(self) -> dict[str, Any]:
        """Reset the game to its initial state.

        Clears the board, resets score/level/lines, refills the bag,
        and spawns the first piece.

        Returns:
            Initial game state dict (same format as get_state()).
        """
        self.board.reset()
        self.score = 0
        self.level = 0
        self.total_lines = 0
        self.current_piece = None
        self.current_x = 0
        self.current_y = 0
        self.current_rotation = 0
        self.next_piece = None
        self.held_piece = None
        self.can_hold = True
        self.game_over = False
        self._bag = []
        self._lock_delay_counter = 0
        self._gravity_counter = 0
        self._lines_cleared_last = 0

        # Fill bag and set up next piece, then spawn
        self.next_piece = self._next_from_bag()
        self._spawn_piece()
        return self.get_state()

    def step(self, action: int) -> tuple[dict[str, Any], float, bool]:
        """Execute one game step with the given action.

        Processes the action, applies gravity, handles lock delay, clears
        lines, updates score/level, and checks for game over.

        Args:
            action: An Action enum value (0-6).

        Returns:
            A tuple of (state_info, reward, done):
              - state_info: dict from get_state()
              - reward: float reward for this step (raw game reward, NOT
                shaped — reward shaping is done in env.py)
              - done: True if the game is over.
        """
        if self.game_over:
            return self.get_state(), 0.0, True

        self._lines_cleared_last = 0
        reward = 0.0
        action_moved_or_rotated = False

        # Process the action
        if action == Action.LEFT:
            if self._move(-1, 0):
                action_moved_or_rotated = True
        elif action == Action.RIGHT:
            if self._move(1, 0):
                action_moved_or_rotated = True
        elif action == Action.ROTATE_CW:
            if self._rotate(1):
                action_moved_or_rotated = True
        elif action == Action.ROTATE_CCW:
            if self._rotate(-1):
                action_moved_or_rotated = True
        elif action == Action.SOFT_DROP:
            if self._move(0, 1):
                reward += 1.0
                self.score += 1
        elif action == Action.HARD_DROP:
            rows_dropped = self._hard_drop()
            reward += 2.0 * rows_dropped
            self.score += 2 * rows_dropped
            # Hard drop immediately locks the piece
            lines = self._lock_piece()
            self._lines_cleared_last = lines
            score_gained = self._calculate_score(lines)
            self.score += score_gained
            reward += score_gained
            if lines > 0:
                self.total_lines += lines
                self._update_level()
            if not self.game_over:
                if not self._spawn_piece():
                    self.game_over = True
            return self.get_state(), reward, self.game_over
        elif action == Action.HOLD:
            self._hold()

        # Apply gravity
        self._apply_gravity()

        # Handle lock delay: check if piece can move down
        if self.current_piece is not None:
            can_move_down = self.board.is_valid_position(
                self.current_piece,
                self.current_x,
                self.current_y + 1,
                self.current_rotation,
            )
            if not can_move_down:
                # Piece is resting on something
                if action_moved_or_rotated:
                    # Reset lock delay on successful move/rotation
                    self._lock_delay_counter = 0
                else:
                    self._lock_delay_counter += 1

                if self._lock_delay_counter >= 30:
                    # Lock the piece
                    lines = self._lock_piece()
                    self._lines_cleared_last = lines
                    score_gained = self._calculate_score(lines)
                    self.score += score_gained
                    reward += score_gained
                    if lines > 0:
                        self.total_lines += lines
                        self._update_level()
                    if not self.game_over:
                        if not self._spawn_piece():
                            self.game_over = True
            else:
                # Piece can still fall, reset lock delay
                self._lock_delay_counter = 0

        return self.get_state(), reward, self.game_over

    def get_state(self) -> dict[str, Any]:
        """Return a dict describing the full observable game state.

        Returns:
            Dict with keys:
              - board_grid: np.ndarray (height x width, int8)
              - current_piece: piece dict or None
              - current_x: int
              - current_y: int
              - current_rotation: int
              - next_piece: piece dict or None
              - held_piece: piece dict or None
              - can_hold: bool
              - score: int
              - level: int
              - lines_cleared: int (lines cleared in LAST step)
              - total_lines: int
        """
        return {
            "board_grid": self.board.get_grid(),
            "current_piece": self.current_piece,
            "current_x": self.current_x,
            "current_y": self.current_y,
            "current_rotation": self.current_rotation,
            "next_piece": self.next_piece,
            "held_piece": self.held_piece,
            "can_hold": self.can_hold,
            "score": self.score,
            "level": self.level,
            "lines_cleared": getattr(self, "_lines_cleared_last", 0),
            "total_lines": self.total_lines,
        }

    def _fill_bag(self) -> None:
        """Refill the 7-bag with a shuffled copy of all 7 piece types.

        The 7-bag randomizer ensures each piece appears exactly once per
        batch of 7, creating a balanced distribution.
        """
        bag = list(PIECE_TYPES)
        random.shuffle(bag)
        self._bag = bag

    def _next_from_bag(self) -> dict:
        """Pop the next piece from the bag, refilling if empty.

        Returns:
            A piece dict from PIECE_TYPES.
        """
        if not self._bag:
            self._fill_bag()
        return self._bag.pop()

    def _spawn_piece(self) -> bool:
        """Spawn the next piece at the top of the board.

        Sets current_piece, current_x (centered), current_y (just above
        visible area), current_rotation (0), and resets can_hold.

        Returns:
            True if the piece was successfully spawned (position is valid),
            False if the spawn overlaps existing blocks (game over).
        """
        self.current_piece = self.next_piece
        self.next_piece = self._next_from_bag()
        self.current_rotation = 0

        # Center the piece horizontally
        shape = self.current_piece["rotations"][0]
        piece_width = shape.shape[1]
        self.current_x = (self.board.width - piece_width) // 2

        # Spawn at the top of the buffer zone so the piece appears
        # at the top of the visible area. For a 30-row board with
        # 10-row buffer, the visible area starts at row 10.
        # Spawn so piece's visible cells are at/near the top of visible area.
        # Buffer rows = height - 20. Spawn at buffer_rows - piece_height rows up.
        buffer_rows = self.board.height - 20  # 10 for standard
        # Place so the piece sits just entering the visible area
        # For 3-row pieces (most): y=8 means rows 8,9,10 -> just entering visible
        # For 4-row pieces (I): y=7 means rows 7,8,9,10
        piece_height = shape.shape[0]
        self.current_y = buffer_rows - piece_height + 1

        self.can_hold = True
        self._lock_delay_counter = 0
        self._gravity_counter = 0

        # Check if spawn position is valid
        if not self.board.is_valid_position(
            self.current_piece, self.current_x, self.current_y, self.current_rotation
        ):
            self.game_over = True
            return False
        return True

    def _move(self, dx: int, dy: int) -> bool:
        """Try to move the current piece by (dx, dy).

        Args:
            dx: Column offset (positive = right).
            dy: Row offset (positive = down).

        Returns:
            True if the move succeeded, False if blocked.
        """
        if self.current_piece is None:
            return False
        new_x = self.current_x + dx
        new_y = self.current_y + dy
        if self.board.is_valid_position(
            self.current_piece, new_x, new_y, self.current_rotation
        ):
            self.current_x = new_x
            self.current_y = new_y
            return True
        return False

    def _rotate(self, direction: int) -> bool:
        """Try to rotate the current piece with SRS wall kicks.

        Attempts each kick offset in order. The first valid position is used.

        Args:
            direction: +1 for clockwise, -1 for counter-clockwise.

        Returns:
            True if the rotation succeeded (possibly with a kick),
            False if all kick tests failed.
        """
        if self.current_piece is None:
            return False

        from_rot = self.current_rotation
        to_rot = (from_rot + direction) % 4

        kicks = get_kick_offsets(self.current_piece, from_rot, to_rot)

        for dx, dy in kicks:
            # dy in SRS: positive = up on screen, which means subtract from row
            new_x = self.current_x + dx
            new_y = self.current_y - dy  # SRS convention: positive dy = up = -row
            if self.board.is_valid_position(
                self.current_piece, new_x, new_y, to_rot
            ):
                self.current_x = new_x
                self.current_y = new_y
                self.current_rotation = to_rot
                return True
        return False

    def _hard_drop(self) -> int:
        """Instantly drop the current piece to its landing position.

        Returns:
            Number of rows dropped.
        """
        if self.current_piece is None:
            return 0
        rows = 0
        while self.board.is_valid_position(
            self.current_piece,
            self.current_x,
            self.current_y + 1,
            self.current_rotation,
        ):
            self.current_y += 1
            rows += 1
        return rows

    def _hold(self) -> bool:
        """Swap the current piece with the held piece.

        If no piece is held, the current piece goes to hold and the next
        piece is spawned. The player cannot hold again until a new piece
        is spawned.

        Returns:
            True if hold was performed, False if can_hold is False.
        """
        if not self.can_hold or self.current_piece is None:
            return False

        if self.held_piece is None:
            # No held piece yet: store current, spawn next
            self.held_piece = self.current_piece
            self.current_piece = None
            self._spawn_piece()
            # Set can_hold AFTER spawn (which resets it to True)
            self.can_hold = False
        else:
            # Swap current with held
            old_held = self.held_piece
            self.held_piece = self.current_piece
            self.current_piece = old_held
            self.current_rotation = 0

            # Re-center the swapped-in piece
            shape = self.current_piece["rotations"][0]
            piece_width = shape.shape[1]
            self.current_x = (self.board.width - piece_width) // 2

            buffer_rows = self.board.height - 20
            piece_height = shape.shape[0]
            self.current_y = buffer_rows - piece_height + 1

            self._lock_delay_counter = 0
            self._gravity_counter = 0
            self.can_hold = False

            if not self.board.is_valid_position(
                self.current_piece, self.current_x, self.current_y, self.current_rotation
            ):
                self.game_over = True
                return True

        return True

    def _lock_piece(self) -> int:
        """Lock the current piece onto the board and clear lines.

        Returns:
            Number of lines cleared after locking.
        """
        if self.current_piece is None:
            return 0

        self.board.place_piece(
            self.current_piece,
            self.current_x,
            self.current_y,
            self.current_rotation,
        )
        lines = self.board.clear_lines()

        self.current_piece = None
        return lines

    def _calculate_score(self, lines_cleared: int) -> int:
        """Calculate the score for lines cleared at the current level.

        Uses NES-style scoring:
          1 line  = 40  * (level + 1)
          2 lines = 100 * (level + 1)
          3 lines = 300 * (level + 1)
          4 lines = 1200 * (level + 1)

        Args:
            lines_cleared: Number of lines cleared (0-4).

        Returns:
            Points earned.
        """
        if lines_cleared <= 0:
            return 0
        base = SCORE_TABLE.get(lines_cleared, 0)
        return base * (self.level + 1)

    def _update_level(self) -> None:
        """Update the level based on total lines cleared.

        Level increases by 1 for every 10 lines cleared.
        """
        self.level = self.total_lines // 10

    def _get_gravity_frames(self) -> int:
        """Return the number of frames per gravity drop at the current level.

        Returns:
            Frames between automatic downward moves.
        """
        if self.level < len(GRAVITY_TABLE):
            return GRAVITY_TABLE[self.level]
        return 1

    def _apply_gravity(self) -> None:
        """Apply gravity: move the piece down if the gravity counter expires.

        Increments the gravity counter each step. When it reaches the
        threshold for the current level, the piece moves down one row.
        """
        if self.current_piece is None:
            return

        self._gravity_counter += 1
        gravity_frames = self._get_gravity_frames()

        if self._gravity_counter >= gravity_frames:
            self._gravity_counter = 0
            # Try to move piece down
            self._move(0, 1)
