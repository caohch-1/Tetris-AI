# _*_ coding: utf-8 _*_
# @Time    :   2022/01/10 07:53:36
# @FileName:   best.py
# @Author  :   Jiawen Yang

import ast
from genetic import *

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

def BestGeneticRunner(Best,env:Tetris):
    population = loadFromFile(Best)
    if len(population) == 1:
        BestChromosome = population[0]
        BestGene = Genetic(BestChromosome,env)
        BestScore = BestGene.run()
        print('Best score is ',BestScore)
        print('Best chromosome is ',BestChromosome)
    else:
        print('Please run train.py first.')
        return None
    
if __name__ == '__main__':
    env = Tetris()
    BestGeneticRunner(filename_CFP,env)