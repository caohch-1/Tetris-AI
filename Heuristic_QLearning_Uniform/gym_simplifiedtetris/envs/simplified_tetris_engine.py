import random
import time
from copy import deepcopy
from typing import Dict, List, Optional, Sequence, Tuple

import cv2.cv2 as cv
import numpy as np
from PIL import Image

# import imageio

from gym_simplifiedtetris.utils import Piece, Colours


class SimplifiedTetrisEngine(object):
    """
    TODO

    :param grid_dims: the grid dimensions (height and width).
    :param piece_size: the size of the pieces in use.
    :param num_pieces: the number of pieces in use.
    :param num_actions: the number of available actions in each state.
    """

    CELL_SIZE = 50

    BLOCK_COLOURS = {
        0: Colours.WHITE.value,
        1: Colours.CYAN.value,
        2: Colours.ORANGE.value,
        3: Colours.YELLOW.value,
        4: Colours.PURPLE.value,
        5: Colours.BLUE.value,
        6: Colours.GREEN.value,
        7: Colours.RED.value,
    }

    @staticmethod
    def _close() -> None:
        """Close the open windows."""
        cv.waitKey(1)
        cv.destroyAllWindows()
        cv.waitKey(1)

    def __init__(
        self,
        grid_dims: Sequence[int],
        piece_size: int,
        num_pieces: int,
        num_actions: int,
    ) -> None:

        self._height, self._width = grid_dims
        self._piece_size = piece_size
        self._num_pieces = num_pieces
        self._num_actions = num_actions

        self._grid = np.zeros((grid_dims[1], grid_dims[0]), dtype="bool")
        self._colour_grid = np.zeros((grid_dims[1], grid_dims[0]), dtype="int")
        self._anchor = [grid_dims[1] / 2 - 1, piece_size - 1]

        self._final_scores = np.array([], dtype=int)
        self._sleep_time = 500
        self._show_agent_playing = True

        self._img = np.array([])
        self._last_move_info = {}
        # self._image_lst = []

        self._initialise_pieces()
        self._update_coords_and_anchor()
        self._get_all_available_actions()
        self._reset()

    def _generate_id_randomly(self) -> int:
        """
        Randomly generate an id.

        :return: a randomly generated ID.
        """
        return random.randint(0, self._num_pieces - 1)

    def _initialise_pieces(self) -> None:
        """Create a dictionary containing the pieces."""
        self._pieces = {}
        for idx in range(self._num_pieces):
            self._pieces[idx] = Piece(self._piece_size, idx)

    def _reset(self) -> None:
        """Reset the score, grid, piece coords, piece id and anchor."""
        self._score = 0
        self._grid = np.zeros_like(self._grid, dtype="bool")
        self._colour_grid = np.zeros_like(self._colour_grid, dtype="int")
        self._update_coords_and_anchor()

    def _render(self, mode: Optional[str] = "human") -> np.ndarray:
        """
        Show an image of the current grid, having dropped the current piece.
        The human has the option to pause (SPACEBAR), speed up (RIGHT key),
        slow down (LEFT key) or quit (ESC) the window.

        :param mode: the render mode.
        :return: the image pixel values.
        """
        assert mode in ["human", "rgb_array"], "Mode should be 'human' or 'rgb_array'."

        grid = self._get_grid()
        self._resize_grid(grid)
        self._draw_separating_lines()
        self._add_img_left()
        self._draw_boundary()

        if mode == "human":
            if self._show_agent_playing:

                """frame_rgb = cv.cvtColor(self._img, cv.COLOR_BGR2RGB)
                self._image_lst.append(frame_rgb)

                if len(self._final_scores) == 4:  # self._score == 20:
                    imageio.mimsave(
                        f"assets/{self._height}x{self._width}_{self._piece_size}.gif",
                        self._image_lst,
                        fps=60,
                        duration=0.5,
                    )
                    self._save_frame = False"""

                cv.imshow(f"Simplified Tetris", self._img)
                k = cv.waitKey(self._sleep_time)

                if k == 3:  # Right arrow has been pressed.
                    self._sleep_time -= 100

                    if self._sleep_time < 100:
                        self._sleep_time = 1

                    time.sleep(self._sleep_time / 1000)
                elif k == 2:  # Left arrow has been pressed.
                    self._sleep_time += 100
                    time.sleep(self._sleep_time / 1000)
                elif k == 27:  # Esc has been pressed.
                    self._show_agent_playing = False
                    self._close()
                elif k == 32:  # Spacebar has been pressed.
                    while True:
                        j = cv.waitKey(30)

                        if j == 32:  # Spacebar has been pressed.
                            break
                        elif j == 27:  # Esc has been pressed.
                            self._show_agent_playing = False
                            self._close()
                            break
        else:
            return self._img

    def _draw_boundary(self) -> None:
        """Draw a horizontal red line to indicate the cut off point."""
        vertical_position = self._piece_size * self.CELL_SIZE
        self._img[
            vertical_position
            - int(self.CELL_SIZE / 40) : vertical_position
            + int(self.CELL_SIZE / 40)
            + 1,
            400:,
            :,
        ] = Colours.RED.value

    def _get_grid(self) -> np.ndarray:
        """
        Get the array of the current grid containing the colour tuples.

        :return: the array of the current grid.
        """
        grid = [
            [self.BLOCK_COLOURS[self._colour_grid[j][i]] for j in range(self._width)]
            for i in range(self._height)
        ]

        return np.array(grid)

    def _resize_grid(self, grid: np.ndarray) -> None:
        """
        Reshape the grid, convert it to an Image and resize it, then convert it
        to an array.

        :param grid: the grid to be resized.
        """
        self._img = grid.reshape((self._height, self._width, 3)).astype(np.uint8)
        self._img = Image.fromarray(self._img, "RGB")
        self._img = self._img.resize(
            (self._width * self.CELL_SIZE, self._height * self.CELL_SIZE)
        )
        self._img = np.array(self._img)

    def _draw_separating_lines(self) -> None:
        """
        Draw the horizontal and vertical _black lines to separate the grid's cells.
        """
        for j in range(-int(self.CELL_SIZE / 40), int(self.CELL_SIZE / 40) + 1):
            self._img[
                [i * self.CELL_SIZE + j for i in range(self._height)], :, :
            ] = Colours.BLACK.value
            self._img[
                :, [i * self.CELL_SIZE + j for i in range(self._width)], :
            ] = Colours.BLACK.value

    def _add_img_left(self) -> None:
        """
        Add the image that will appear to the left of the grid.
        """
        img_array = np.zeros((self._height * self.CELL_SIZE, 400, 3)).astype(np.uint8)
        mean_score = (
            0.0 if len(self._final_scores) == 0 else np.mean(self._final_scores)
        )

        self._add_statistics(
            img_array=img_array,
            items=[
                [
                    "Height",
                    "Width",
                    "",
                    "Current score",
                    "Mean score",
                ],
                [
                    f"{self._height}",
                    f"{self._width}",
                    "",
                    f"{self._score}",
                    f"{mean_score:.1f}",
                ],
            ],
            x_offsets=[50, 300],
        )
        self._img = np.concatenate((img_array, self._img), axis=1)

    @staticmethod
    def _add_statistics(
        img_array: np.ndarray, items: List[List[str]], x_offsets: List[int]
    ) -> None:
        """
        Add statistics to the array provided.

        :param img_array: the array to be edited.
        :param items: the lists to be added to the array.
        :param x_offsets: the horizontal positions where the statistics should be added.
        """
        for i, item in enumerate(items):
            for count, j in enumerate(item):
                cv.putText(
                    img_array,
                    j,
                    (x_offsets[i], 60 * (count + 1)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    1,
                    Colours.WHITE.value,
                    2,
                    cv.LINE_AA,
                )

    def _update_coords_and_anchor(self) -> None:
        """Update the current piece, and reset the anchor."""
        self._piece = self._pieces[self._generate_id_randomly()]
        self._anchor = [self._width / 2 - 1, self._piece_size - 1]

    def _is_illegal(self) -> bool:
        """
        Check if the piece's current position is illegal by looping over each
        of its square blocks.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/42e11e98573edf0c5270d0cc33f1cf1bae3d9d49/src/engine.py#L23

        :return: whether the piece's current position is illegal.
        """
        # Loop over each of the piece's blocks.
        for i, j in self._piece._coords:
            x_pos, y_pos = self._anchor[0] + i, self._anchor[1] + j

            # Don't check if the move is illegal when the block is too high.
            if y_pos < 0:
                continue

            # Check if the move is illegal. The last condition must come after
            # the previous conditions.
            if (
                x_pos < 0
                or x_pos >= self._width
                or y_pos >= self._height
                or self._grid[x_pos, y_pos] > 0
            ):

                return True

        return False

    def _hard_drop(self) -> None:
        """
        Find the position to place the piece (the anchor) by hard dropping the current piece.
        """
        while True:
            # Keep going until current piece occupies a full cell, then
            # backtrack once.
            if not self._is_illegal():
                self._anchor[1] += 1
            else:
                self._anchor[1] -= 1
                break

    def _clear_rows(self) -> int:
        """
        Remove blocks from every full row.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/42e11e98573edf0c5270d0cc33f1cf1bae3d9d49/src/engine.py#L83

        :return: the number of rows cleared.
        """
        can_clear = np.all(self._grid, axis=0)
        new_grid = np.zeros_like(self._grid)
        new_colour_grid = np.zeros_like(self._colour_grid)
        col = self._height - 1

        self._last_move_info["eliminated_num_blocks"] = 0

        for row_num in range(self._height - 1, -1, -1):

            if not can_clear[row_num]:
                new_grid[:, col] = self._grid[:, row_num]
                new_colour_grid[:, col] = self._colour_grid[:, row_num]
                col -= 1
            else:
                self._last_move_info["eliminated_num_blocks"] += self._last_move_info[
                    "rows_added_to"
                ][row_num]

        self._grid = new_grid
        self._colour_grid = new_colour_grid

        num_rows_cleared = sum(can_clear)
        self._last_move_info["num_rows_cleared"] = num_rows_cleared

        return num_rows_cleared

    def _update_grid(self, set_piece: bool) -> None:
        """
        Set the current piece using the anchor.

        :param set_piece: whether to set the piece.
        """
        self._last_move_info["rows_added_to"] = {
            row_num: 0 for row_num in range(self._height)
        }
        # Loop over each block.
        for piece_x_coord, piece_y_coord in self._piece._coords:
            x_coord, y_coord = (
                piece_x_coord + self._anchor[0],
                piece_y_coord + self._anchor[1],
            )
            if set_piece:
                self._last_move_info["rows_added_to"][y_coord] += 1
                self._grid[x_coord, y_coord] = 1
                self._colour_grid[x_coord, y_coord] = self._piece._idx + 1
            else:
                self._grid[x_coord, y_coord] = 0
                self._colour_grid[x_coord, y_coord] = 0

        anchor_height = self._height - self._anchor[1]
        max_y_coord = self._piece._max_y_coord[self._piece._rotation]
        min_y_coord = self._piece._min_y_coord[self._piece._rotation]
        self._last_move_info["landing_height"] = anchor_height - 0.5 * (
            min_y_coord + max_y_coord
        )

    def _get_reward(self) -> Tuple[float, int]:
        """
        Return the reward, which is the number of rows cleared.

        :return: the reward and the number of rows cleared.
        """
        num_rows_cleared = self._clear_rows()

        return float(num_rows_cleared), num_rows_cleared

    def _get_all_available_actions(self) -> None:
        """Get the actions available for each of the pieces in use."""
        self._all_available_actions = {}
        for idx, piece in self._pieces.items():
            self._piece = piece
            self._all_available_actions[idx] = self._compute_available_actions()

    def _compute_available_actions(self) -> Dict[int, Tuple[int, int]]:
        """
        Compute the actions available with the current piece.

        Author: Andrean Lay
        Source: https://github.com/andreanlay/tetris-ai-deep-reinforcement-learning/blob/42e11e98573edf0c5270d0cc33f1cf1bae3d9d49/src/engine.py#L196

        :return: the available actions.
        """
        available_actions: Dict[int, Tuple[int, int]] = {}
        count = 0

        for rotation in self._piece._all_coords.keys():
            self._rotate_piece(rotation)

            max_x_coord = self._piece._max_x_coord[rotation]
            min_x_coord = self._piece._min_x_coord[rotation]

            for translation in range(abs(min_x_coord), self._width - max_x_coord):

                if count == self._num_actions:

                    return available_actions

                self._anchor = [translation, 0]
                self._hard_drop()

                self._update_grid(True)
                available_actions[count] = (translation, rotation)
                self._update_grid(False)

                count += 1

        return available_actions

    def _get_dellacherie_scores(self) -> np.array:
        """
        Get the Dellacherie feature values.

        :return: a list of the Dellacherie feature values.
        """
        weights = np.array([-1, 1, -1, -1, -4, -1], dtype="double")
        ratings = np.empty((self._num_actions), dtype="double")

        for action, (translation, rotation) in self._all_available_actions[
            self._piece._idx
        ].items():
            old_grid = deepcopy(self._grid)
            old_anchor = deepcopy(self._anchor)
            old_colour_grid = deepcopy(self._colour_grid)

            self._rotate_piece(rotation)
            self._anchor = [translation, 0]
            self._hard_drop()
            self._update_grid(True)
            self._clear_rows()

            feature_values = np.empty((6), dtype="double")
            for count, feature_func in enumerate(self._get_dellacherie_funcs()):
                feature_values[count] = feature_func()

            ratings[action] = np.dot(feature_values, weights)
            self._update_grid(False)
            self._anchor = deepcopy(old_anchor)
            self._grid = deepcopy(old_grid)
            self._colour_grid = deepcopy(old_colour_grid)

        max_indices = np.argwhere(ratings == np.amax(ratings)).flatten()

        if len(max_indices) == 1:

            return ratings

        return self._get_priorities(max_indices)

    def _get_priorities(self, max_indices: np.array) -> np.array:
        """
        Calculate the priorities of the available actions.

        :param max_indices: the actions with the maximum ratings.
        :return: the priorities.
        """
        priorities = np.zeros((self._num_actions), dtype="double")

        for action in max_indices:
            translation, rotation = self._all_available_actions[self._piece._idx][
                action
            ]
            x_spawn_pos = self._width / 2 + 1
            priorities[action] += 100 * abs(translation - x_spawn_pos)

            if translation < x_spawn_pos:
                priorities[action] += 10

            priorities[action] -= rotation / 90

            # Ensure that the priority of the best actions is never negative.
            priorities[action] += 5  # 5 is always greater than rotation / 90.

        return priorities

    def _get_dellacherie_funcs(self) -> list:
        """
        Get the Dellacherie feature functions.

        :return: a list of the Dellacherie feature functions.
        """
        return [
            self._get_landing_height,
            self._get_eroded_cells,
            self._get_row_transitions,
            self._get_column_transitions,
            self._get_holes,
            self._get_cumulative_wells,
        ]

    def _get_landing_height(self) -> int:
        """
        Get the landing height. Landing height = the midpoint of the last piece to be placed.

        :return: landing height.
        """
        if "landing_height" in self._last_move_info:

            return self._last_move_info["landing_height"]

        return 0

    def _get_eroded_cells(self) -> int:
        """
        Return the eroded cells value. Num. eroded cells = number of rows cleared x number of blocks removed that were added to the grid by the last action.

        :return: eroded cells.
        """
        if "num_rows_cleared" in self._last_move_info:

            return (
                self._last_move_info["num_rows_cleared"]
                * self._last_move_info["eliminated_num_blocks"]
            )

        return 0

    def _get_row_transitions(self) -> float:
        """
        Return the row transitions value. Row transitions = Number of transitions from empty to full cells (or vice versa), examining each row one at a time.

        Author: Ben Schofield
        Source: https://github.com/Benjscho/gym-mdptetris/blob/1a47edc33330deb638a03275e484c3e26932d802/gym_mdptetris/envs/feature_functions.py#L45

        :return: row transitions.
        """
        # A full column should be added either side.
        grid = np.ones((self._width + 2, self._height), dtype="bool")
        grid[1:-1, :] = self._grid

        return np.diff(grid.T).sum()

    def _get_column_transitions(self) -> float:
        """
        Return the column transitions value. Column transitions = Number of transitions from empty to full (or vice versa), examining each column one at a time.

        Author: Ben Schofield
        Source: https://github.com/Benjscho/gym-mdptetris/blob/1a47edc33330deb638a03275e484c3e26932d802/gym_mdptetris/envs/feature_functions.py#L60

        :return: column transitions.
        """
        # A full row should be added to the bottom.
        grid = np.ones((self._width, self._height + 1), dtype="bool")
        grid[:, :-1] = self._grid

        return np.diff(grid).sum()

    def _get_holes(self) -> int:
        """
        Get the number of holes present in the current grid. A hole is an empty cell with at least one full cell above it in the same column.

        :return: holes.
        """
        return np.count_nonzero((self._grid).cumsum(axis=1) * ~self._grid)

    def _get_cumulative_wells(self) -> int:
        """
        Get the cumulative wells value. Cumulative wells is defined here:
        https://arxiv.org/abs/1905.01652.  For each well, find the depth of
        the well, d(w), then calculate the sum from i=1 to d(w) of i.  Lastly,
        sum the well sums.  A block is part of a well if the cells directly on
        either side are full, and the block can be reached from above (there
        are no full cells directly above it).

        :return: cumulative wells.
        """
        cumulative_wells = 0

        new_grid = np.ones((self._width + 2, self._height + 1), dtype="bool")
        new_grid[1:-1, :-1] = self._grid

        for col in range(1, self._width + 1):

            depth = 1
            well_complete = False

            for row in range(self._height):

                cell_mid = new_grid[col][row]
                cell_right = new_grid[col + 1][row]
                cell_left = new_grid[col - 1][row]

                if cell_mid >= 1:
                    well_complete = True

                # Check either side to see if the cells are occupied.
                if not well_complete and cell_left > 0 and cell_right > 0:
                    cumulative_wells += depth
                    depth += 1

        return cumulative_wells

    def _rotate_piece(self, rotation: int) -> None:
        """
        Set the piece's rotation and rotate the current piece.

        :param rotation: the piece's rotation.
        """
        self._piece._rotation = rotation
        self._piece._coords = self._piece._all_coords[self._piece._rotation]

    def _get_translation_rotation(self, action: int) -> Tuple[int, int]:
        """
        Return the translation and rotation associated with the action provided.

        :param action: the action.
        :return: the translation and rotation associated with the action provided.
        """
        return self._all_available_actions[self._piece._idx][action]
