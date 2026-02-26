"""
Board logic for a 10x30 Tetris grid.

The board is a 2D numpy array (height x width) of int8 values:
  - 0 = empty cell
  - 1-7 = piece type ID (used for coloring)

The top 10 rows (indices 0-9) are a hidden buffer zone.
The visible play area is rows 10-29 (20 visible rows).
"""

from __future__ import annotations

import numpy as np


class Board:
    """Tetris board with collision detection, line clearing, and board metrics.

    Attributes:
        width: Number of columns (default 10).
        height: Number of rows including buffer zone (default 30).
        grid: 2D numpy array of shape (height, width), dtype int8.
    """

    def __init__(self, width: int = 10, height: int = 30) -> None:
        """Initialize an empty board.

        Args:
            width: Number of columns.
            height: Total number of rows (including hidden buffer zone).
        """
        self.width = width
        self.height = height
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)

    def is_valid_position(self, piece: dict, x: int, y: int, rotation: int) -> bool:
        """Check whether a piece at (x, y) with given rotation fits on the board.

        A position is valid if every filled cell of the piece:
          - Is within the board boundaries (0 <= col < width, 0 <= row < height).
          - Does not overlap a filled cell on the board grid.

        Args:
            piece: Piece dict from pieces.py (must have 'rotations' key).
            x: Column offset of the piece's top-left corner.
            y: Row offset of the piece's top-left corner.
            rotation: Rotation state index (0-3).

        Returns:
            True if the position is valid, False otherwise.
        """
        shape = piece["rotations"][rotation]
        rows, cols = shape.shape
        for r in range(rows):
            for c in range(cols):
                if shape[r, c] != 0:
                    board_row = y + r
                    board_col = x + c
                    # Check boundaries
                    if board_col < 0 or board_col >= self.width:
                        return False
                    if board_row < 0 or board_row >= self.height:
                        return False
                    # Check collision with existing blocks
                    if self.grid[board_row, board_col] != 0:
                        return False
        return True

    def place_piece(self, piece: dict, x: int, y: int, rotation: int) -> None:
        """Lock a piece onto the board at the given position.

        Writes the piece's ID into the board grid at each filled cell.
        Does NOT check validity first — caller must ensure position is valid.

        Args:
            piece: Piece dict (must have 'id' and 'rotations' keys).
            x: Column offset of the piece's top-left corner.
            y: Row offset of the piece's top-left corner.
            rotation: Rotation state index (0-3).
        """
        shape = piece["rotations"][rotation]
        piece_id = piece["id"]
        rows, cols = shape.shape
        for r in range(rows):
            for c in range(cols):
                if shape[r, c] != 0:
                    self.grid[y + r, x + c] = piece_id

    def clear_lines(self) -> int:
        """Remove all fully filled rows and shift everything above them down.

        Returns:
            The number of lines cleared (0-4).
        """
        # Find rows where every cell is filled (nonzero)
        full_rows = []
        for r in range(self.height):
            if np.all(self.grid[r] != 0):
                full_rows.append(r)

        if not full_rows:
            return 0

        lines_cleared = len(full_rows)
        # Remove full rows and prepend empty rows at the top
        mask = np.ones(self.height, dtype=bool)
        for r in full_rows:
            mask[r] = False
        remaining = self.grid[mask]
        empty_rows = np.zeros((lines_cleared, self.width), dtype=np.int8)
        self.grid = np.vstack([empty_rows, remaining])
        return lines_cleared

    def get_grid(self) -> np.ndarray:
        """Return a copy of the board grid.

        Returns:
            A numpy array of shape (height, width), dtype int8.
        """
        return self.grid.copy()

    def get_holes(self) -> int:
        """Count the number of holes on the board.

        A hole is any empty cell that has at least one filled cell above it
        in the same column.

        Returns:
            Total number of holes across all columns.
        """
        filled = self.grid != 0
        block_above = np.maximum.accumulate(filled, axis=0)
        holes = block_above & ~filled
        return int(holes.sum())

    def get_aggregate_height(self) -> int:
        """Calculate the sum of the heights of all columns.

        A column's height is the distance from the bottom of the board to the
        highest filled cell in that column (0 if the column is empty).

        Returns:
            Sum of all column heights.
        """
        return int(self.get_column_heights().sum())

    def get_bumpiness(self) -> int:
        """Calculate the bumpiness of the board surface.

        Bumpiness is the sum of absolute differences between adjacent column
        heights: sum(|h[i] - h[i+1]|) for i in 0..width-2.

        Returns:
            Total bumpiness value.
        """
        heights = self.get_column_heights()
        return int(np.abs(np.diff(heights)).sum())

    def get_column_heights(self) -> np.ndarray:
        """Get the height of every column (vectorized).

        A column's height is measured from the bottom of the board (row
        height-1) up to the topmost filled cell. An empty column has height 0.

        Returns:
            Numpy array of ints with length equal to board width.
        """
        filled = self.grid != 0
        has_block = filled.any(axis=0)
        first_block = np.argmax(filled, axis=0)
        return np.where(has_block, self.height - first_block, 0)

    def is_game_over(self) -> bool:
        """Determine whether the game is over.

        The game is over when any cell in the visible spawn area (rows 10-11
        for standard 30-row board) is occupied, indicating a newly spawned
        piece would immediately overlap.

        Returns:
            True if the game-over condition is met.
        """
        # The buffer zone is rows 0-9. The visible spawn area is rows 10-11.
        # Check if any cell in the top two visible rows is filled.
        buffer_rows = self.height - 20  # typically 10
        spawn_rows = self.grid[buffer_rows:buffer_rows + 2]
        return bool(np.any(spawn_rows != 0))

    def reset(self) -> None:
        """Clear the entire board, setting all cells to 0."""
        self.grid = np.zeros((self.height, self.width), dtype=np.int8)
