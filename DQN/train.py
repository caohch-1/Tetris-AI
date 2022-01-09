from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim

from src.tetris import Tetris
from src.deep_q_network import MLP
from src.arg_parser import get_args
from collections import deque


def train(opt):
    """Preparation for Network and Tetris Env."""
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    state = env.reset()
    net = MLP(input_num=opt.input_features, hidden_num=opt.model)
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    epoch = 0
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net.to('cuda')
        state = state.cuda()

    execution_record = deque(maxlen=30000)

    print("Train start with epochs={}, decay_epochs={}, lr={}, model={}, saved_path={}, gui={}".format(opt.num_epochs, opt.num_decay_epochs, opt.lr, opt.model, opt.saved_path, bool(opt.gui)))
    while epoch < opt.num_epochs:
        # Exploration Function
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)

        # Get all possible states
        next_steps = None
        if opt.input_features == 200:
            next_steps = env.get_next_state_img()
        elif opt.input_features == 4:
            next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()

        # Evaluate every state, predictions is a list contains scores for each state
        net.eval()
        with torch.no_grad():
            predictions = net(next_states)[:, 0]
        net.train()

        # Choose action (i.e., choose best next state)
        index = randint(0, len(next_steps) - 1) if random() <= epsilon else torch.argmax(predictions).item()

        # Run the chosen action and get reward
        next_state = next_states[index, :]
        action = next_actions[index]
        reward, done = env.step(action, render=opt.gui)
        if torch.cuda.is_available():
            next_state = next_state.cuda()

        # If game-over and replay_memory (i.e., training data) is sufficient, we train the net
        if state.shape[0] == opt.input_features:
            execution_record.append([state, reward, next_state, done])
        if done:
            final_score = env.score
            final_tetrominoes = env.tetrominoes
            final_cleared_lines = env.cleared_lines
            state = env.reset()
            if torch.cuda.is_available():
                state = state.cuda()
        else:
            state = next_state
            continue

        if len(execution_record) < 3000:
            continue

        # Prepare data for training (in batch size)
        batch = sample(execution_record, min(len(execution_record), opt.batch_size))
        state_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        state_batch = torch.stack(tuple(state for state in state_batch))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.stack(tuple(state for state in next_state_batch))
        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()

        # Update Q_value (i.e., train net), here |Q - (R + gamma * Q')|
        q_values = net(state_batch)

        net.eval()
        with torch.no_grad():
            next_prediction_batch = net(next_state_batch)
        net.train()

        y_batch = torch.cat(
            tuple(reward if done else reward + opt.gamma * prediction for reward, done, prediction in
                  zip(reward_batch, done_batch, next_prediction_batch)))[:, None]
        optimizer.zero_grad()
        loss = criterion(q_values, y_batch)
        loss.backward()
        optimizer.step()

        epoch += 1
        print("Epoch: {}/{}, Score: {}, Tetrominoes {}, Cleared lines: {}".format(
            epoch,
            opt.num_epochs,
            final_score,
            final_tetrominoes,
            final_cleared_lines))

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(net, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(net, "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args(train=True)
    train(opt)
