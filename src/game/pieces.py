"""
Tetromino definitions with all 4 rotation states and SRS kick tables.

All piece data follows the official Super Rotation System (SRS) from the
Tetris Guideline. Each piece has 4 rotation states (0=spawn, 1=CW, 2=180,
3=CCW) and corresponding wall-kick offset data.

Coordinate convention:
  - Rotations are stored as lists of (row, col) offsets relative to the
    piece's origin (top-left corner of the bounding box).
  - On the board, row 0 is the top and row increases downward.
  - Column 0 is the left edge and column increases rightward.
"""

from __future__ import annotations

import numpy as np

# =============================================================================
# Piece Colors — standard Tetris guideline colors (RGB)
# =============================================================================

COLOR_CYAN   = (0, 255, 255)    # I
COLOR_YELLOW = (255, 255, 0)    # O
COLOR_PURPLE = (128, 0, 128)    # T
COLOR_GREEN  = (0, 255, 0)      # S
COLOR_RED    = (255, 0, 0)      # Z
COLOR_BLUE   = (0, 0, 255)     # J
COLOR_ORANGE = (255, 165, 0)    # L

# =============================================================================
# Tetromino Definitions
# =============================================================================
# Each rotation state is a 2D numpy array where 1 marks a filled cell.
# The arrays use the smallest bounding box that fits the piece.
# Rotation order: [0=spawn, 1=CW (R), 2=180 (2), 3=CCW (L)]

I_PIECE: dict = {
    "id": 1,
    "name": "I",
    "color": COLOR_CYAN,
    "rotations": [
        # Rotation 0 (spawn)
        np.array([
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ], dtype=np.int8),
        # Rotation 1 (CW / R)
        np.array([
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
        ], dtype=np.int8),
        # Rotation 2 (180)
        np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [0, 0, 0, 0],
        ], dtype=np.int8),
        # Rotation 3 (CCW / L)
        np.array([
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.int8),
    ],
}

O_PIECE: dict = {
    "id": 2,
    "name": "O",
    "color": COLOR_YELLOW,
    "rotations": [
        # All 4 rotations are identical for O-piece
        np.array([
            [1, 1],
            [1, 1],
        ], dtype=np.int8),
        np.array([
            [1, 1],
            [1, 1],
        ], dtype=np.int8),
        np.array([
            [1, 1],
            [1, 1],
        ], dtype=np.int8),
        np.array([
            [1, 1],
            [1, 1],
        ], dtype=np.int8),
    ],
}

T_PIECE: dict = {
    "id": 3,
    "name": "T",
    "color": COLOR_PURPLE,
    "rotations": [
        # Rotation 0 (spawn)
        np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        # Rotation 1 (CW / R)
        np.array([
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 0],
        ], dtype=np.int8),
        # Rotation 2 (180)
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 0],
        ], dtype=np.int8),
        # Rotation 3 (CCW / L)
        np.array([
            [0, 1, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], dtype=np.int8),
    ],
}

S_PIECE: dict = {
    "id": 4,
    "name": "S",
    "color": COLOR_GREEN,
    "rotations": [
        # Rotation 0 (spawn)
        np.array([
            [0, 1, 1],
            [1, 1, 0],
            [0, 0, 0],
        ], dtype=np.int8),
        # Rotation 1 (CW / R)
        np.array([
            [0, 1, 0],
            [0, 1, 1],
            [0, 0, 1],
        ], dtype=np.int8),
        # Rotation 2 (180)
        np.array([
            [0, 0, 0],
            [0, 1, 1],
            [1, 1, 0],
        ], dtype=np.int8),
        # Rotation 3 (CCW / L)
        np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
        ], dtype=np.int8),
    ],
}

Z_PIECE: dict = {
    "id": 5,
    "name": "Z",
    "color": COLOR_RED,
    "rotations": [
        # Rotation 0 (spawn)
        np.array([
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        # Rotation 1 (CW / R)
        np.array([
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
        ], dtype=np.int8),
        # Rotation 2 (180)
        np.array([
            [0, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
        ], dtype=np.int8),
        # Rotation 3 (CCW / L)
        np.array([
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 0],
        ], dtype=np.int8),
    ],
}

J_PIECE: dict = {
    "id": 6,
    "name": "J",
    "color": COLOR_BLUE,
    "rotations": [
        # Rotation 0 (spawn)
        np.array([
            [1, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        # Rotation 1 (CW / R)
        np.array([
            [0, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ], dtype=np.int8),
        # Rotation 2 (180)
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 1],
        ], dtype=np.int8),
        # Rotation 3 (CCW / L)
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [1, 1, 0],
        ], dtype=np.int8),
    ],
}

L_PIECE: dict = {
    "id": 7,
    "name": "L",
    "color": COLOR_ORANGE,
    "rotations": [
        # Rotation 0 (spawn)
        np.array([
            [0, 0, 1],
            [1, 1, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        # Rotation 1 (CW / R)
        np.array([
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ], dtype=np.int8),
        # Rotation 2 (180)
        np.array([
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ], dtype=np.int8),
        # Rotation 3 (CCW / L)
        np.array([
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
        ], dtype=np.int8),
    ],
}

# =============================================================================
# Ordered list of all piece types
# =============================================================================

PIECE_TYPES: list[dict] = [I_PIECE, O_PIECE, T_PIECE, S_PIECE, Z_PIECE, J_PIECE, L_PIECE]

# =============================================================================
# SRS Wall-Kick Offset Data
# =============================================================================
#
# When a rotation is attempted and the piece overlaps existing blocks or walls,
# the game tries up to 5 alternative positions (kicks). The offsets below are
# (dx, dy) where dx is column offset (positive = right) and dy is row offset
# (positive = up, i.e., SUBTRACTED from the row index on the board).
#
# The kick tests are tried in order; the first valid position is used. If none
# succeed, the rotation fails.
#
# Source: https://tetris.wiki/Super_Rotation_System
#
# Key format: KICK_TABLE[from_rotation][to_rotation] = list of (dx, dy) offsets.
# Rotation states: 0=spawn, 1=R (CW), 2=180 (2), 3=L (CCW)
#
# NOTE: dy follows "screen up = negative row" convention used in SRS docs,
#       so positive dy means UP on screen (subtract from row index).
# =============================================================================

# ---------------------------------------------------------------------------
# Kick table for J, L, S, T, Z pieces (standard / "JLSTZ" table)
# ---------------------------------------------------------------------------

KICK_TABLE_JLSTZ: dict[int, dict[int, list[tuple[int, int]]]] = {
    # 0 -> 1  (spawn -> CW)
    0: {
        1: [( 0, 0), (-1, 0), (-1,  1), ( 0, -2), (-1, -2)],
        3: [( 0, 0), ( 1, 0), ( 1,  1), ( 0, -2), ( 1, -2)],
    },
    # 1 -> 2  (CW -> 180)   and   1 -> 0  (CW -> spawn)
    1: {
        2: [( 0, 0), ( 1, 0), ( 1, -1), ( 0,  2), ( 1,  2)],
        0: [( 0, 0), ( 1, 0), ( 1, -1), ( 0,  2), ( 1,  2)],
    },
    # 2 -> 3  (180 -> CCW)  and   2 -> 1  (180 -> CW)
    2: {
        3: [( 0, 0), ( 1, 0), ( 1,  1), ( 0, -2), ( 1, -2)],
        1: [( 0, 0), (-1, 0), (-1,  1), ( 0, -2), (-1, -2)],
    },
    # 3 -> 0  (CCW -> spawn) and  3 -> 2  (CCW -> 180)
    3: {
        0: [( 0, 0), (-1, 0), (-1, -1), ( 0,  2), (-1,  2)],
        2: [( 0, 0), (-1, 0), (-1, -1), ( 0,  2), (-1,  2)],
    },
}

# ---------------------------------------------------------------------------
# Kick table for I piece (separate offsets)
# ---------------------------------------------------------------------------

KICK_TABLE_I: dict[int, dict[int, list[tuple[int, int]]]] = {
    # 0 -> 1  and  0 -> 3
    0: {
        1: [( 0, 0), (-2, 0), ( 1, 0), (-2, -1), ( 1,  2)],
        3: [( 0, 0), (-1, 0), ( 2, 0), (-1,  2), ( 2, -1)],
    },
    # 1 -> 2  and  1 -> 0
    1: {
        2: [( 0, 0), (-1, 0), ( 2, 0), (-1,  2), ( 2, -1)],
        0: [( 0, 0), ( 2, 0), (-1, 0), ( 2,  1), (-1, -2)],
    },
    # 2 -> 3  and  2 -> 1
    2: {
        3: [( 0, 0), ( 2, 0), (-1, 0), ( 2,  1), (-1, -2)],
        1: [( 0, 0), ( 1, 0), (-2, 0), ( 1, -2), (-2,  1)],
    },
    # 3 -> 0  and  3 -> 2
    3: {
        0: [( 0, 0), ( 1, 0), (-2, 0), ( 1, -2), (-2,  1)],
        2: [( 0, 0), (-2, 0), ( 1, 0), (-2, -1), ( 1,  2)],
    },
}


def get_kick_offsets(piece: dict, from_rot: int, to_rot: int) -> list[tuple[int, int]]:
    """Return the list of SRS kick offsets for a rotation transition.

    Args:
        piece: Piece dict (must have 'name' key).
        from_rot: Current rotation state (0-3).
        to_rot: Target rotation state (0-3).

    Returns:
        List of (dx, dy) kick offsets to try in order.
        For O-piece, returns [(0, 0)] (no kicks).
    """
    if piece["name"] == "O":
        return [(0, 0)]
    if piece["name"] == "I":
        return KICK_TABLE_I[from_rot][to_rot]
    return KICK_TABLE_JLSTZ[from_rot][to_rot]
