"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter

from src.tetris import Tetris
from collections import deque


# Todo: Change to fit new feature vector (i.e. 204 input dim.)
class DeepQNetwork(nn.Module):
    def __init__(self):
        super(DeepQNetwork, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(4, 256), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(inplace=True))
        self.fc3 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(inplace=True))
        self.out = nn.Sequential(nn.Linear(64, 1))

        self._create_weights()

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.out(x)

        return x


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    # Every block is a square, block_size means len of its side
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--batch_size", type=int, default=512, help="The number of replays per batch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=1)
    parser.add_argument("--final_epsilon", type=float, default=1e-3)
    parser.add_argument("--num_decay_epochs", type=float, default=2000)
    parser.add_argument("--num_epochs", type=int, default=3000)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--replay_memory_size", type=int, default=30000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--gui", type=int, default=1)

    args = parser.parse_args()
    return args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(opt):
    """Preparation for Logging"""
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)

    """Preparation for Network and Tetris Env."""
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    state = env.reset()
    net = DeepQNetwork()
    optimizer = optim.Adam(net.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    epoch = 0
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
        net.to('cuda')
        state = state.cuda()

    replay_memory = deque(maxlen=opt.replay_memory_size)

    while epoch < opt.num_epochs:
        # Exploration Function
        epsilon = opt.final_epsilon + (max(opt.num_decay_epochs - epoch, 0) * (
                opt.initial_epsilon - opt.final_epsilon) / opt.num_decay_epochs)

        # Get all possible states
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
        if random() <= epsilon:
            index = randint(0, len(next_steps) - 1)
        else:
            index = torch.argmax(predictions).item()

        # Run the chosen action and get reward
        next_state = next_states[index, :]
        action = next_actions[index]
        reward, done = env.step(action, render=opt.gui)
        if torch.cuda.is_available():
            next_state = next_state.cuda()

        # If game-over and replay_memory (i.e., training data) is sufficient, we train the net
        replay_memory.append([state, reward, next_state, done])
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

        if len(replay_memory) < opt.replay_memory_size / 10:
            continue

        # Prepare data for training (in batch size)
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
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
        writer.add_scalar('Train/Score', final_score, epoch - 1)
        writer.add_scalar('Train/Tetrominoes', final_tetrominoes, epoch - 1)
        writer.add_scalar('Train/Cleared lines', final_cleared_lines, epoch - 1)

        if epoch > 0 and epoch % opt.save_interval == 0:
            torch.save(net, "{}/tetris_{}".format(opt.saved_path, epoch))

    torch.save(net, "{}/tetris".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
