#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unittest

import numpy as np

from gym_simplifiedtetris.envs import SimplifiedTetrisEngine as Engine
from gym_simplifiedtetris.utils import Piece


class SimplifiedTetrisEngineStandardTetrisTest(unittest.TestCase):
    def setUp(self) -> None:
        height = 20
        width = 10
        self.piece_size = 4

        num_actions, num_pieces = {
            1: (width, 1),
            2: (2 * width - 1, 1),
            3: (4 * width - 4, 2),
            4: (4 * width - 6, 7),
        }[self.piece_size]

        self.engine = Engine(
            grid_dims=(height, width),
            piece_size=self.piece_size,
            num_pieces=num_pieces,
            num_actions=num_actions,
        )

        self.engine._reset()

    def tearDown(self) -> None:
        self.engine._close()
        del self.engine

    def test__is_illegal_non_empty_overlapping(self) -> None:
        self.engine._piece = Piece(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        self.engine._grid[0, self.engine._height - 1] = 1
        self.assertEqual(self.engine._is_illegal(), True)

    def test__hard_drop_empty_grid(self) -> None:
        self.engine._piece = Piece(self.piece_size, 0)
        self.engine._anchor = [0, 0]
        self.engine._hard_drop()
        self.assertEqual(self.engine._anchor, [0, self.engine._height - 1])

    def test__hard_drop_non_empty_grid(self) -> None:
        self.engine._piece = Piece(self.piece_size, 0)
        self.engine._anchor = [0, 0]
        self.engine._grid[0, self.engine._height - 1] = 1
        self.engine._hard_drop()
        self.assertEqual(self.engine._anchor, [0, self.engine._height - 2])

    def test__clear_rows_output_with_empty_grid(self) -> None:
        self.assertEqual(self.engine._clear_rows(), 0)

    def test__clear_rows_empty_grid_after(self) -> None:
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")
        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test__clear_rows_output_one_full_row(self) -> None:
        self.engine._grid[:, self.engine._height - 1 :] = 1
        self.assertEqual(self.engine._clear_rows(), 1)

    def test__clear_rows_one_full_row_grid_after(self) -> None:
        self.engine._grid[:, self.engine._height - 1 :] = 1
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")
        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test__clear_rows_output_two_full_rows(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.assertEqual(self.engine._clear_rows(), 2)

    def test__clear_rows_two_full_rows_grid_after(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")
        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test__clear_rows_output_two_full_rows_full_cell_above(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.engine._grid[3, self.engine._height - 3] = 1
        self.assertEqual(self.engine._clear_rows(), 2)

    def test__clear_rows_two_full_rows_full_cell_above_grid_after(self) -> None:
        self.engine._grid[:, self.engine._height - 2 :] = 1
        self.engine._grid[3, self.engine._height - 3] = 1
        self.engine._clear_rows()
        grid_after = np.zeros((self.engine._width, self.engine._height), dtype="bool")
        grid_after[3, self.engine._height - 1] = 1
        np.testing.assert_array_equal(self.engine._grid, grid_after)

    def test__update_grid_simple(self) -> None:
        self.engine._piece = Piece(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        grid_to_compare = np.zeros(
            (self.engine._width, self.engine._height), dtype="bool"
        )
        grid_to_compare[0, self.engine._height - self.piece_size :] = 1
        self.engine._update_grid(True)
        np.testing.assert_array_equal(self.engine._grid, grid_to_compare)

    def test__update_grid_empty(self) -> None:
        self.engine._piece = Piece(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        grid_to_compare = np.zeros(
            (self.engine._width, self.engine._height), dtype="bool"
        )
        self.engine._update_grid(False)
        np.testing.assert_array_equal(self.engine._grid, grid_to_compare)

    def test__update_grid_populated(self) -> None:
        self.engine._piece = Piece(self.piece_size, 0)
        self.engine._grid[0, self.engine._height - self.piece_size :] = 1
        self.engine._anchor = [0, self.engine._height - 1]
        grid_to_compare = np.zeros(
            (self.engine._width, self.engine._height), dtype="bool"
        )
        self.engine._update_grid(False)
        np.testing.assert_array_equal(self.engine._grid, grid_to_compare)

    def test__get_all_available_actions(self) -> None:
        self.engine._get_all_available_actions()
        for value in self.engine._all_available_actions.values():
            self.assertEqual(self.engine._num_actions, len(value))

    def test__get_dellacherie_scores_empty_grid(self) -> None:
        self.engine._piece = Piece(self.piece_size, 0)
        print(self.engine._all_available_actions[self.engine._piece._idx])
        array_to_compare = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                614.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                4.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                312.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                302.0,
            ]
        )
        array_to_compare = np.zeros(self.engine._num_actions)
        array_to_compare[10] = abs(0 - 6) * 100 + 10 - (90 / 90) + 5
        array_to_compare[16] = abs(6 - 6) * 100 + 0 - (90 / 90) + 5
        array_to_compare[27] = abs(3 - 6) * 100 + 10 - (270 / 90) + 5
        array_to_compare[33] = abs(9 - 6) * 100 + 0 - (270 / 90) + 5
        np.testing.assert_array_equal(
            self.engine._get_dellacherie_scores(),
            array_to_compare,
        )

    def test__get_dellacherie_funcs_populated_grid(self) -> None:
        self.engine._grid[:, -5:] = True
        self.engine._grid[1, self.engine._height - 5 : self.engine._height - 1] = False
        self.engine._grid[self.engine._width - 1, self.engine._height - 2] = False
        self.engine._grid[self.engine._width - 2, self.engine._height - 1] = False
        self.engine._grid[self.engine._width - 3, self.engine._height - 3] = False
        self.engine._grid[self.engine._width - 1, self.engine._height - 6] = True
        self.engine._grid[3, self.engine._height - 3 : self.engine._height - 1] = False
        self.engine._piece = Piece(self.piece_size, 0)
        self.engine._anchor = [0, 0]
        self.engine._hard_drop()
        self.engine._update_grid(True)
        self.engine._clear_rows()
        array_to_compare = np.array(
            [func() for func in self.engine._get_dellacherie_funcs()]
        )
        np.testing.assert_array_equal(
            array_to_compare,
            np.array([5.5 + 0.5 * self.piece_size, 0, 48, 18, 5, 10], dtype="double"),
        )

    def test__get_landing_height_I_piece_(self) -> None:
        self.engine._piece = Piece(self.piece_size, 0)
        self.engine._anchor = [0, self.engine._height - 1]
        self.engine._update_grid(True)
        self.assertEqual(self.engine._get_landing_height(), 0.5 * (1 + self.piece_size))

    def test__get_eroded_cells_empty(self) -> None:
        self.assertEqual(self.engine._get_eroded_cells(), 0)

    def test__get_eroded_cells_single(self) -> None:
        self.engine._grid[:, self.engine._height - 1 :] = True
        self.engine._grid[0, self.engine._height - 1] = False
        self.engine._piece = Piece(self.piece_size, 0)
        self.engine._anchor = [0, 0]
        self.engine._hard_drop()
        self.engine._update_grid(True)
        self.engine._clear_rows()
        self.assertEqual(self.engine._get_eroded_cells(), 1)

    def test__get_row_transitions_empty(self) -> None:
        self.assertEqual(self.engine._get_row_transitions(), 40)

    def test__get_row_transitions_populated(self) -> None:
        self.engine._grid[:, -2:] = True
        self.engine._grid[0, self.engine._height - 1] = False
        self.engine._grid[2, self.engine._height - 1] = False
        self.engine._grid[1, self.engine._height - 2] = False
        self.assertEqual(self.engine._get_row_transitions(), 42)

    def test__get_row_transitions_populated_more_row_transitions(self) -> None:
        self.engine._grid[:, -2:] = True
        self.engine._grid[0, self.engine._height - 2 :] = False
        self.engine._grid[2, self.engine._height - 2 :] = False
        self.engine._grid[4, self.engine._height - 1] = False
        np.testing.assert_array_equal(self.engine._get_row_transitions(), 46)

    def test__get_column_transitions_empty(self) -> None:
        self.assertEqual(self.engine._get_column_transitions(), 10)

    def test__get_column_transitions_populated(self) -> None:
        self.engine._grid[:, -2:] = True
        self.engine._grid[0, self.engine._height - 1] = False
        self.engine._grid[2, self.engine._height - 1] = False
        self.engine._grid[1, self.engine._height - 2] = False
        self.assertEqual(self.engine._get_column_transitions(), 14)

    def test__get_column_transitions_populated_less_column_transitions(self) -> None:
        self.engine._grid[:, -2:] = True
        self.engine._grid[0, self.engine._height - 2 :] = False
        self.engine._grid[2, self.engine._height - 2 :] = False
        self.engine._grid[4, self.engine._height - 1] = False
        np.testing.assert_array_equal(self.engine._get_column_transitions(), 12)

    def test__get_holes_empty(self) -> None:
        self.assertEqual(self.engine._get_holes(), 0)

    def test__get_holes_populated_two_holes(self) -> None:
        self.engine._grid[:, -2:] = True
        self.engine._grid[0, self.engine._height - 1] = False
        self.engine._grid[2, self.engine._height - 1] = False
        self.assertEqual(self.engine._get_holes(), 2)

    def test__get_holes_populated_no_holes(self) -> None:
        self.engine._grid[:, -2:] = True
        self.engine._grid[0, self.engine._height - 2 :] = False
        self.assertEqual(self.engine._get_holes(), 0)

    def test__get_holes_populated_one_hole(self) -> None:
        self.engine._grid[:, -2:] = True
        self.engine._grid[0, self.engine._height - 2 :] = False
        self.engine._grid[2, self.engine._height - 2 :] = False
        self.engine._grid[4, self.engine._height - 1] = False
        np.testing.assert_array_equal(self.engine._get_holes(), 1)

    def test__get_cumulative_wells_empty(self) -> None:
        np.testing.assert_array_equal(self.engine._get_cumulative_wells(), 0)

    def test__get_cumulative_wells_populated(self) -> None:
        self.engine._grid[:, -2:] = True
        self.engine._grid[0, self.engine._height - 2 :] = False
        np.testing.assert_array_equal(self.engine._get_cumulative_wells(), 3)

    def test__get_cumulative_wells_populated_deeper_well(self) -> None:
        self.engine._grid[:, -2:] = True
        self.engine._grid[0, self.engine._height - 2 :] = False
        self.engine._grid[2, self.engine._height - 2 :] = False
        self.engine._grid[4, self.engine._height - 1] = False
        np.testing.assert_array_equal(self.engine._get_cumulative_wells(), 6)


if __name__ == "__main__":
    unittest.main()
