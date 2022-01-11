# _*_ coding: utf-8 _*_
# @Time    :   2022/01/09 18:40:23
# @FileName:   train.py
# @Author  :   Jiawen Yang

from genetic import GeneticProcessor

if __name__ == '__main__':
    gene = GeneticProcessor(10)
    gene.train()