from enum import Enum
from typing import Any, Tuple

from matplotlib import colors
import numpy as np


def get_bgr_code(colour_name: str) -> Tuple[float, float, float]:
    """
    Get the inverted RGB code corresponding to the arg provided.

    :param colour_name: a string of the colour name,
    :return: an inverted RGB code of the inputted colour name.
    """
    return tuple(np.array([255, 255, 255]) * colors.to_rgb(colour_name))[::-1]


class Colours(Enum):
    """
    Enumerate inverted RGB code.
    """

    WHITE = get_bgr_code("white")
    BLACK = get_bgr_code("black")
    CYAN = get_bgr_code("cyan")
    ORANGE = get_bgr_code("orange")
    YELLOW = get_bgr_code("yellow")
    PURPLE = get_bgr_code("purple")
    BLUE = get_bgr_code("blue")
    GREEN = get_bgr_code("green")
    RED = get_bgr_code("red")
