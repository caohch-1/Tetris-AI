import torch
from src.tetris import Tetris
from src.arg_parser import get_args

import numpy as np



def test(opt):
    """Set up Env"""
    env = Tetris(width=10, height=20, block_size=30)
    env.reset()

    while True:
        # Get all possible next states
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # get random action
        action = env.get_best_action()

        # Excute action
        _, done = env.step(action, render=opt.gui)

        if done:
            return env.score, env.tetrominoes, env.cleared_lines


if __name__ == "__main__":
    opt = get_args(train=False)
    print("Test Start with test_num={}, saved_path={}, gui={}".format(opt.test_num, opt.saved_path, bool(opt.gui)))
    total = [0, 0, 0]
    for i in range(opt.test_num):
        temp = test(opt)
        total[0] += temp[0]
        total[1] += temp[1]
        total[2] += temp[2]
        print(i, np.array(total) / (i + 1))