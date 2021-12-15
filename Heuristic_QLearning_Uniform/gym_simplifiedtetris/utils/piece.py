from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import numpy as np

PieceCoord = List[Tuple[int, int]]
Rotation = int
PieceCoords = Dict[Rotation, PieceCoord]
PieceInfo = Dict[str, Union[PieceCoords, str]]
PieceID = int
PiecesInfo = Dict[PieceID, PieceInfo]
PieceSize = int

# Create the piece dictionaries.
_MONOMINOS: PiecesInfo = dict()
_MONOMINOS[0] = {"coords": {0: [(0, 0)]}, "name": "O"}

_DOMINOS: PiecesInfo = dict()
_DOMINOS[0] = {"coords": {0: [(0, 0), (0, -1)], 90: [(0, 0), (1, 0)]}, "name": "I"}

_TROMINOS: PiecesInfo = dict()
_TROMINOS[0] = {
    "coords": {
        0: [(0, 0), (0, -1), (0, -2)],
        90: [(0, 0), (1, 0), (2, 0)],
        180: [(0, 0), (0, 1), (0, 2)],
        270: [(0, 0), (-1, 0), (-2, 0)],
    },
    "name": "I",
}
_TROMINOS[1] = {
    "coords": {
        0: [(0, 0), (1, 0), (0, -1)],
        90: [(0, 0), (0, 1), (1, 0)],
        180: [(0, 0), (-1, 0), (0, 1)],
        270: [(0, 0), (0, -1), (-1, 0)],
    },
    "name": "L",
}

_TETRIMINOS = dict()
_TETRIMINOS[0] = {
    "coords": {
        0: [(0, 0), (0, -1), (0, -2), (0, -3)],
        90: [(0, 0), (1, 0), (2, 0), (3, 0)],
        180: [(0, 0), (0, 1), (0, 2), (0, 3)],
        270: [(0, 0), (-1, 0), (-2, 0), (-3, 0)],
    },
    "name": "I",
}
_TETRIMINOS[1] = {
    "coords": {
        0: [(0, 0), (1, 0), (0, -1), (0, -2)],
        90: [(0, 0), (0, 1), (1, 0), (2, 0)],
        180: [(0, 0), (-1, 0), (0, 1), (0, 2)],
        270: [(0, 0), (0, -1), (-1, 0), (-2, 0)],
    },
    "name": "L",
}
_TETRIMINOS[2] = {
    "coords": {
        0: [(0, 0), (0, -1), (-1, 0), (-1, -1)],
        90: [(0, 0), (1, 0), (0, -1), (1, -1)],
        180: [(0, 0), (0, 1), (1, 0), (1, 1)],
        270: [(0, 0), (-1, 0), (0, 1), (-1, 1)],
    },
    "name": "O",
}
_TETRIMINOS[3] = {
    "coords": {
        0: [(0, 0), (-1, 0), (1, 0), (0, 1)],
        90: [(0, 0), (0, -1), (0, 1), (-1, 0)],
        180: [(0, 0), (1, 0), (-1, 0), (0, -1)],
        270: [(0, 0), (0, 1), (0, -1), (1, 0)],
    },
    "name": "T",
}
_TETRIMINOS[4] = {
    "coords": {
        0: [(0, 0), (-1, 0), (0, -1), (0, -2)],
        90: [(0, 0), (0, -1), (1, 0), (2, 0)],
        180: [(0, 0), (1, 0), (0, 1), (0, 2)],
        270: [(0, 0), (0, 1), (-1, 0), (-2, 0)],
    },
    "name": "J",
}
_TETRIMINOS[5] = {
    "coords": {
        0: [(0, 0), (-1, 0), (0, -1), (1, -1)],
        90: [(0, 0), (0, -1), (1, 0), (1, 1)],
        180: [(0, 0), (1, 0), (0, 1), (-1, 1)],
        270: [(0, 0), (0, 1), (-1, 0), (-1, -1)],
    },
    "name": "S",
}
_TETRIMINOS[6] = {
    "coords": {
        0: [(0, 0), (-1, -1), (0, -1), (1, 0)],
        90: [(0, 0), (1, -1), (1, 0), (0, 1)],
        180: [(0, 0), (1, 1), (0, 1), (-1, 0)],
        270: [(0, 0), (-1, 1), (-1, 0), (0, -1)],
    },
    "name": "Z",
}

_PIECES_DICT: Dict[PieceSize, PiecesInfo] = dict()
_PIECES_DICT[1] = _MONOMINOS
_PIECES_DICT[2] = _DOMINOS
_PIECES_DICT[3] = _TROMINOS
_PIECES_DICT[4] = _TETRIMINOS


def _generate_max_min(coord_string: str, coords: PieceCoords):
    """
    Return the max and min x and y coordinates for the coordinate string and coordinates provided.

    :param coord_string: the string specifying what to calculate.
    :param coords: the piece coordinates.
    :return: the max and min x and y coordinates for the coordinate string and coordinates provided.
    """
    coord_strings = {
        "max_y_coord": {"func": np.max, "index": 1},
        "min_y_coord": {"func": np.min, "index": 1},
        "max_x_coord": {"func": np.max, "index": 0},
        "min_x_coord": {"func": np.min, "index": 0},
    }

    return {
        rot: coord_strings[coord_string]["func"](
            [coord[coord_strings[coord_string]["index"]] for coord in coords]
        )
        for rot, coords in coords.items()
    }


@dataclass
class Piece(object):
    """
    Class for instantiating a Tetris piece.
    """

    _size: int
    _idx: int
    _rotation: int = 0
    _all_coords: PieceCoords = field(init=False)
    _coords: PieceCoord = field(init=False)
    _name: str = field(init=False)
    _max_y_coord: Dict[int, int] = field(init=False)
    _min_y_coord: Dict[int, int] = field(init=False)
    _max_x_coord: Dict[int, int] = field(init=False)
    _min_x_coord: Dict[int, int] = field(init=False)

    def __post_init__(self):
        self._all_coords = deepcopy(_PIECES_DICT[self._size][self._idx]["coords"])
        self._coords = self._all_coords[self._rotation]
        self._name = deepcopy(_PIECES_DICT[self._size][self._idx]["name"])
        self._max_y_coord = _generate_max_min("max_y_coord", self._all_coords)
        self._min_y_coord = _generate_max_min("min_y_coord", self._all_coords)
        self._max_x_coord = _generate_max_min("max_x_coord", self._all_coords)
        self._min_x_coord = _generate_max_min("min_x_coord", self._all_coords)
