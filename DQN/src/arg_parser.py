import argparse


def get_args(train: bool):
    if train:
        parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")
        parser.add_argument("--width", type=int, default=10)
        parser.add_argument("--height", type=int, default=20)
        # Every block is a square, block_size means len of its side
        parser.add_argument("--block_size", type=int, default=30)

        parser.add_argument("--model", type=int, default=64)
        parser.add_argument("--input_features", type=int, default=4)
        parser.add_argument("--batch_size", type=int, default=512)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--gamma", type=float, default=0.99)
        parser.add_argument("--initial_epsilon", type=float, default=0.7)
        parser.add_argument("--final_epsilon", type=float, default=1e-3)
        parser.add_argument("--num_decay_epochs", type=float, default=2000)
        parser.add_argument("--num_epochs", type=int, default=3000)

        parser.add_argument("--save_interval", type=int, default=1000)
        parser.add_argument("--saved_path", type=str, default="trained_models64")
        parser.add_argument("--gui", type=int, default=1)

        args = parser.parse_args()
        return args
    else:
        parser = argparse.ArgumentParser("""Implementation of Deep Q Network to play Tetris""")
        parser.add_argument("--width", type=int, default=10, help="The common width for all images")
        parser.add_argument("--height", type=int, default=20, help="The common height for all images")
        parser.add_argument("--block_size", type=int, default=30, help="Size of a block")

        parser.add_argument("--fps", type=int, default=300, help="frames per second")
        parser.add_argument("--saved_path", type=str, default="trained_models64")
        parser.add_argument("--gui", type=int, default=1)

        parser.add_argument("--test_num", type=int, default=3000)

        args = parser.parse_args()
        return args
