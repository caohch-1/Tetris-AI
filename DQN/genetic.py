# _*_ coding: utf-8 _*_
# @Time    :   2022/01/09 14:13:23
# @FileName:   genetic.py
# @Author  :   Jiawen Yang

from networkx.classes import ordered
from operator import itemgetter
import numpy as np
import random
import copy
import time

filename = "BestChromosome.txt"

class Genetic:
    def __init__(self, chromosome):
        self.chromosome = chromosome


    def Simulation(test_board,test_piece,move):
        rot,y = move[0],move[1]

        if test_piece is None:
            return None
        
        for i in range(0,rot):
            rotate(test_piece)  # need to modify. func: rotate test_piece to match the desired move

        # need to modify. func: 碰撞判断 到底部更新x坐标 更新board

        return test_board


    def GetExpextedScore(self,board):
        # need to modify
        # fulllines: 已经消除的总行数
        # holes: 洞数
        # blocks: 下落方块数
        # max_x: 当前最高x
        # std_x: 标准差
        # avg_x: 均值
        # max_diff_x: 最大高度差
        return float(fulllines * self.chromosome[0] -
                     holes * self.chromosome[1] -
                     blocks * self.chromosome[2] -
                     max_x * self.chromosome[3] -
                     std_x * self.chromosome[4] -
                     avg_x * self.chromosome[5] -
                     max_diff_x * self.chromosome[6]  )


    def GetBestMove(self, board, piece, next_piece): # the piece contains num_rotations,id,x,y,color...
        best_rot = 0
        best_y = 0
        best_score = -float('inf')
        next_score = [0,0,-float('inf')]

        for rot in range(0, piece.num_rotations): # need to modify num_rotation
            for y in range(10):
                move = [rot, y]
                test_board = copy.deepcopy(board)
                test_piece = copy.deepcopy(piece)
                test_board = self.Simulation(test_board,test_piece,move)

                if test_board is not None:
                    # choose the best after next_piece
                    for rot2 in range(0, next_piece.num_rotations): # need to modify num_rotation
                        for y2 in range(10):
                            move2 = [rot2, y2]
                            test_board2 = copy.deepcopy(test_board)
                            test_piece2 = copy.deepcopy(next_piece)
                            test_board2 = self.Simulation(test_board2,test_piece2,move2)
                            if test_board2 is not None:
                                test_score2 = self.GetExpextedScore(test_board2)
                                if next_score[2] < test_score2:
                                    next_score = [rot2, y2, test_score2]
                    if best_score < next_score[2]:
                        best_score = next_score[2]
                        best_rot = rot
                        best_y = y

        return [best_rot,best_y]











class GeneticProcessor:
    def __init__(self, numGen, dimChromomsome = 7):
        self.numGen = int(numGen)
        self.numRun = 3
        self.dimChromomsome = dimChromomsome
        self.generation = list()
        self.population = list()

    def getNewChromosome(self):
        # return new list of random Chromosome
        return [round(random.uniform(0.0, 1.0), 3) for i in range(7)]


    def AVGFitness(self, chromosome):
        avgFitness = 0
        for i in range(self.numRun):
            g = Genetic(chromosome)
            start = time.time()
            score = run() # need to modify
            finish = time.time()
            avgFitness += score + round(finish-start)
        return avgFitness/self.numRun


    def mutation(self, a):
        if random.randint(1, 10) == 10:  # 10% of mutation
            if 4.5 <= a <= 5.0:
                return -0.1 * random.randint(1, 5)
            else:
                return 0.1 * random.randint(1, 5)
        else:
            return 0  # 90% of not mutation


    def CrossingChromosome(self,a,b):
        newchromosome = [0]* self.dimChromomsome
        for x in range(self.dimChromomsome):
            if random.randint(0,9) == 1:
                newchromosome[x] = self.mutation(a[x])
            elif random.randint(0,9) == 2:
                newchromosome[x] = self.mutation(b[x])
            else:
                newchromosome[x] = (a[x] + b[x]) / 2
            newchromosome[x] += self.mutation(newchromosome[x])
        return newchromosome



    def CrossingTournmentPopulation(self, population, k):
        new_population = list()
        if len(population) == 2:
            new_population.append(population[0])
        else:
            for x in range(0,int(k),2):
                chromosome1,_ = population[x]
                chromosome2,_ = population[x+1]
                new_population.append(self.CrossingChromosome(chromosome1,chromosome2))
                new_population.append(self.CrossingChromosome(chromosome2,chromosome1))
        return new_population


    def CrossingGeneticBeamPopulation(self,population, numGen):
        new_population = list()
        ordered_population =sorted(population, key=itemgetter(1), reverse=True)
        if numGen == self.numGen:
            chromosome, _ = ordered_population[0]
            new_population.append(chromosome)
        else:
            cross = self.CrossingTournmentPopulation(ordered_population, round(len(ordered_population)/2))
            for x in range(round(len(ordered_population)/2)):
                chromosome,_ = ordered_population[x]
                new_population.append(chromosome)
            for x in range(round(len(cross))):
                new_population.append(cross[x])
        return new_population


    def chromToStr(c, dim):
        _str = "["
        for i in range(dim):
            _str += str((c[i]))
            if i < dim - 1:
                _str += ","
        _str += "]"
        return _str


    def save(self, filename, population):
        file = open(filename, "a+")
        y = 0
        for x in range(1, len(population) + 1, 1):
            writethis = str(self.chromToStr(population[y], len(population[y]))) + "\n"
            y += 1
            file.write(writethis)
        file.close()


    def train(self):
        print("Training starts!")
        self.numGen0 = 8
        self.generation = [self.getNewChromosome() for i in range(self.numGen0)]
        game_index_array = list()
        scoreArray = list()
        gene0Array = list()
        gene1Array = list()
        gene2Array = list()
        gene3Array = list()
        gene4Array = list()
        gene5Array = list()
        gene6Array = list()
        i = 0
        n_run = 0
        while True:
            population = list()
            print("Gen "+str(i+1)+" starts!")
            for x in range(len(self.generation)):
                n_run += 1
                avgScoreChromosome = self.AVGFitness(self.generation[x])
                scoreArray.append(avgScoreChromosome)
                population.append((self.generation[x], avgScoreChromosome))
                gene0Array.append(self.generation[x][0])
                gene1Array.append(self.generation[x][1])
                gene2Array.append(self.generation[x][2])
                gene3Array.append(self.generation[x][3])
                gene4Array.append(self.generation[x][4])
                gene5Array.append(self.generation[x][5])
                gene6Array.append(self.generation[x][6])
                game_index_array.append(n_run)
                print("Gen: ", i, " Run: ", x, " AvgScore: ", str(avgScoreChromosome))
            print('Full Generation = ', self.generation)
            print('Gen ends! ',i+1)
            i+=1
            self.generation = self.CrossingGeneticBeamPopulation(population, i)
            for x in range(len(population)):
                self.population.append(population[x])
            if len(self.population) == 1:
                break
            else:
                continue
        file = open(filename, 'w')
        file.close()
        self.save(filename, self.generation)
        print("Training finished!")