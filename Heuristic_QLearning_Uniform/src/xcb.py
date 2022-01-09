import argparse
import torch
import cv2
from tetris import Tetris

env = Tetris()
env.board = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 2, 0, 0, 0, 0, 0, 0],
            [0, 6, 6, 2, 2, 0, 0, 0, 0, 0],
            [0, 4, 6, 2, 7, 7, 7, 0, 0, 0],
            [4, 4, 6, 6, 0, 2, 7, 3, 0, 0],
            [4, 6, 6, 6, 2, 2, 2, 3, 3, 0],
            [1, 1, 5, 5, 5, 5, 4, 4, 3, 0],
            [1, 1, 3, 3, 4, 4, 0, 4, 4, 0],
            [0, 3, 3, 4, 4, 4, 4, 1, 1, 0],
            [6, 6, 6, 2, 4, 4, 7, 1, 1, 0],
            [5, 3, 2, 0, 4, 1, 1, 4, 4, 6],
            [5, 2, 2, 4, 4, 1, 1, 4, 5, 0],
            [0, 2, 3, 3, 6, 2, 2, 2, 6, 5],
            [5, 0, 4, 4, 6, 6, 2, 2, 2, 5],
            [2, 2, 0, 4, 4, 4, 4, 3, 6, 5],
            [5, 0, 3, 3, 6, 6, 2, 2, 1, 1],
            [4, 7, 7, 0, 2, 2, 2, 6, 6, 6]]
env.ind = 4
pos = {'x': 1, 'y': 0}
env.piece = [row[:] for row in env.pieces[env.ind]]
curr_piece = [row[:] for row in env.piece]
piece = [row[:] for row in curr_piece]
piece = [[5], [5], [5], [5]]
#print(env.check_collision(piece, pos))
shit = env.get_next_states()
print(shit)
next_actions, next_states = zip(*shit.items())
print(next_actions)
print(next_states)
next_states = torch.stack(next_states)
print(next_states)
for row in env.board:
    for item in row:
        print(item, end=" ")
    print(" ", end="\n")