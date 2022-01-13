# _*_ coding: utf-8 _*_
# @Time    :   2022/01/09 18:40:23
# @FileName:   train.py
# @Author  :   Jiawen Yang

from genetic import GeneticProcessor
from src.arg_genetic import get_args

if __name__ == '__main__':
    opt = get_args(train=False)
    gene = GeneticProcessor(opt.test_num)
    gene.train()