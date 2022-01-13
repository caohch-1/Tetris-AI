# _*_ coding: utf-8 _*_
# @Time    :   2022/01/10 18:27:28
# @FileName:   test.py
# @Author  :   Jiawen Yang

import ast
import time

from torch._C import Block
from genetic import *


from src.arg_genetic import get_args

from src.tetris import Tetris

filename_CGBP = "BestChromosome_CGBP.txt"
filename_CFP = "BestChromosome_CFP.txt"

def loadFromFile(fileName):
    population = list()
    File = open(fileName, "r")
    for line in File:
        chromosome = ast.literal_eval(line)
        print(chromosome)
        population.append(chromosome)
    File.close()
    return population

def test(t,Best,env:Tetris):
    population = loadFromFile(Best)
    if len(population) == 1:
        BestChromosome = population[0]
        BestGene = Genetic(BestChromosome,env)
        Best = BestGene.run_in_time(t)
        return Best
    else:
        print('Please run train.py first.')
        return None
    
if __name__ == '__main__':
    opt = get_args(train=False)
    Score = list()
    Lines = list()
    Blocks = list()
    for i in range(opt.test_num):  # ROUND
        env = Tetris()
        Best = test(opt.time,filename_CFP,env)  # TIME 5MIN
        Score.append(Best[0])
        Lines.append(Best[1])
        Blocks.append(Best[2])
    AvgScore = sum(Score)/len(Score)
    AvgLines = sum(Lines)/len(Lines)
    AvgBlocks = sum(Blocks)/len(Blocks)
    print('AvgScore = ',AvgScore)
    print('AvgLines = ',AvgLines)
    print('AvgBlocks = ',AvgBlocks)