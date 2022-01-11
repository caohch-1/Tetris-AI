import argparse


def get_args(train: bool):
    parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")
    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models128")
    parser.add_argument("--output", type=str, default="output.mp4")
    parser.add_argument("--gui", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=3000)
    parser.add_argument("--model", type=str, default='DQN64')
    parser.add_argument("--w1",type=int,default=-5)
    parser.add_argument("--w2", type=int, default=-7.5)
    parser.add_argument("--w3", type=int, default=5)
    parser.add_argument("--w4", type=int, default=-1)
    parser.add_argument("--w5", type=int, default=40)
    parser.add_argument("--w6", type=int, default=-1)
    parser.add_argument("--w7", type=int, default=-1)
    parser.add_argument("--w8", type=int, default=-1)


    args = parser.parse_args()
    return args
