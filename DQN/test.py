import torch
import torch.backends.cudnn as cudnn
from src.tetris import Tetris
from src.nets import ConvNet, MLP
from src.arg_parser import get_args

import numpy as np


def test(opt):
    """Load Net"""
    if torch.cuda.is_available():
        model = torch.load("{}/tetris".format(opt.saved_path))
    else:
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net.to('cuda')
    model.eval()

    """Set up Env"""
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()

    while True:
        # Get all possible next states
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # Choose action by DQN (i.e., choose best next state)
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]

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
        if i % (opt.test_num / 5) == 0:
            print(i, np.array(total) / (i + 1))
