import argparse


def get_args(train: bool):
    parser = argparse.ArgumentParser("""Implementation of Genetic Beam Search to play Tetris""")
    parser.add_argument("--gui", type=int, default=1)
    parser.add_argument("--test_num", type=int, default=300)
    parser.add_argument("--time", type=int, default=300)
    parser.add_argument("--alpha",type=int,default=0.5)
    parser.add_argument("--beta", type=int, default=0.2)
    parser.add_argument("--gamma", type=int, default=0.1)



    args = parser.parse_args()
    return args
