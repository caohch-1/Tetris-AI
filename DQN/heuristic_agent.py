import torch
from src.tetris import Tetris
from src.arg_parser_heur import get_args

import numpy as np

opt = get_args(train=False)
env = Tetris(width=10, height=20, block_size=30)

def heuristic_func():
    hole_num = env.get_holes(env.board)
    maxHeight = env.yaq_state()[0]
    avgHeight = env.yaq_state()[2]
    diff = env.yaq_state()[3]

    return opt.w1 * maxHeight + opt.w2 * avgHeight + opt.w3 * hole_num + opt.w4 * diff


def get_best_action():
    possible_actions = env.get_next_states().keys()
    best_action = None
    max_h = -99999

    for action in possible_actions:
        current_h = heuristic_func()
        temp = env.get_next_states()[action]
        current_h += (opt.w5 * temp[0] + opt.w6 * temp[1] + opt.w7 * temp[2] + opt.w8 * temp[3])  # clear,holenum,bumpiness,maxheight

        if current_h > max_h:
            max_h = current_h
            best_action = action

    '''
    for action in possible_actions:
        temp = self.get_next_states()[action]
        temp1 += (40*temp[0]-1*temp[1]-1*temp[2]-1*temp[3])#clear,holenum,bumpiness,maxheight
    end = time.time()
    print(end - start)
    '''
    return best_action

def test(opt):
    """Set up Env"""

    env.reset()

    while env.cleared_lines < 5000:
        # Get all possible next states
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # get random action
        action = get_best_action()

        # Excute action
        _, done = env.step(action, render=opt.gui)

        if done:
            return env.score, env.tetrominoes, env.cleared_lines


if __name__ == "__main__":

    print("Test Start with test_num={}, saved_path={}, gui={}".format(opt.test_num, opt.saved_path, bool(opt.gui)))
    total = [0, 0, 0]
    for i in range(opt.test_num):
        temp = test(opt)
        total[0] += temp[0]
        total[1] += temp[1]
        total[2] += temp[2]
        print(i, np.array(total) / (i + 1))