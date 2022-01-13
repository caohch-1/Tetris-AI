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
from src.tetris import Tetris

filename_CGBP = "BestChromosome_CGBP.txt"
filename_CFP = "BestChromosome_CFP.txt"


class Genetic:
    def __init__(self, chromosome, env: Tetris):
        self.chromosome = chromosome
        self.env = env


    def GetExpectedScore(self,env:Tetris,state):
        max_y,std_y,mean_y,max_diff_y = env.yaq_state()
        fulllines = state[0]
        holes = state[1]
        blocks = env.tetrominoes
        abs_y = state[2]
        return float(self.chromosome[0]*fulllines -
                     self.chromosome[1]*holes-
                     self.chromosome[2]*blocks-
                     self.chromosome[3]*max_y-
                     self.chromosome[4]*std_y-
                     self.chromosome[5]*abs_y-
                     self.chromosome[6]*max_diff_y)


    def GetBestMove(self,next_actions,next_states):
        best_y = 0
        best_rot = 0
        best_score = -float('inf')
        next_score = [0,0,-float('inf')]
        for i,action in enumerate(next_actions):
            y = action[0]
            rot = action[1]
            test_env = copy.deepcopy(self.env)
            test_env.step(action,render = False,name = 'GeneticProcessor')
            if not test_env.gameover:
                next_step2 = test_env.get_next_states()
                next_action2, next_state2 = zip(*next_step2.items())
                for k,action2 in enumerate(next_action2):
                    y2 = action2[0]
                    rot2 = action2[1]
                    test_env2 = copy.deepcopy(test_env)
                    test_env2.step(action2,render = False,name = 'GeneticProcessor')
                    if not test_env2.gameover:
                        score = self.GetExpectedScore(test_env2,next_state2[k])
                        # print('current score = ',score)
                        if next_score[2]<score:
                            next_score = [y2,rot2,score]
            if best_score<next_score[2]:
                best_score = next_score[2]
                best_y = y
                best_rot = rot
        # print("----------------------------------------")
        # print("Best score = ",best_score)
        # print('Best move = ',y,best_rot)
        # print("----------------------------------------")
        return [best_y,best_rot]



    def run(self,i = 0,gen = 0):
        # return score
        while True:
            print("--------------------------------")
            print('current chromosome: ',self.chromosome)
            next_steps = self.env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            best_action = self.GetBestMove(next_actions, next_states)
            score,stop = self.env.step(best_action,name = 'Gen = '+str(gen)+' Sample# '+str(i))
            print("Current score = ",self.env.score)
            if stop:
                print('GeneticProcessor Generation = '+str(gen)+' Sample# '+str(i)+" Score = " + str(self.env.score))
                return self.env.score


    def run_in_time(self,t,i = 0,gen = 0):
        t1 = time.time()
        while True:
            print("--------------------------------")
            # print('current chromosome: ',self.chromosome)
            next_steps = self.env.get_next_states()
            next_actions, next_states = zip(*next_steps.items())
            best_action = self.GetBestMove(next_actions, next_states)
            score,stop = self.env.step(best_action, render = False,name = 'Gen = '+str(gen)+' Sample# '+str(i))
            print("Current score = ",self.env.score)
            print("Current lines = ",self.env.cleared_lines)
            print("Current blocks = ",self.env.tetrominoes)
            t2 = time.time()
            duration = t2 - t1
            print('Duration time = ',duration)
            if duration >= t:
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                print('Duration time = ',duration)
                print('Finished score = ',self.env.score)
                print("Finished lines = ",self.env.cleared_lines)
                print("Finished blocks = ",self.env.tetrominoes)
                print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
                return self.env.score,self.env.cleared_lines,self.env.tetrominoes
            if stop:
                print('GeneticProcessor Generation = '+str(gen)+' Sample# '+str(i)+" Score = " + str(score))
                return self.env.score,self.env.cleared_lines,self.env.tetrominoes








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


    def AVGFitness(self, chromosome,gen):
        avgFitness = 0
        for i in range(self.numRun):
            print('current numRun = ',i)
            env = Tetris()
            env.reset()
            g = Genetic(chromosome, env)
            start = time.time()
            score = g.run(i,gen)
            print('Final score = ',score)
            print('-----------------------------------')
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


    def bestChromosomeSearch(self, population, k):
        bestK = list()
        OrderedChromosome = sorted(population, key=itemgetter(1), reverse=True)

        for x in range(k):
            chromosome, _ = OrderedChromosome[x]
            bestK.append(chromosome)
            print(" - BestK - ", self.chromToStr(chromosome, self.dimChromomsome) + " --- WScore Of: ", str(_))
        return bestK


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


    def CrossingTournmentPopulationHalf(self, population, k):
        new_population = list()
        if len(population) == 2:
            new_population.append(population[0])
        else:
            for x in range(0,int(k),2):
                new_population.append(self.CrossingChromosome(population[x], population[x + 1]))
        return new_population


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


    def CrossingFixedPopulation(self, population, numGen):
        new_population = list()
        ordered_population =sorted(population, key=itemgetter(1), reverse=True)
        if numGen == self.numGen:
            new_population.append(ordered_population[0])
        else:
            cross = self.CrossingTournmentPopulationHalf(ordered_population, round(len(ordered_population)/2))
            for x in range(round(len(ordered_population)/2)):
                new_population.append(ordered_population[x])
            for x in range(round(len(cross))):
                new_population.append(cross[x])
            new = round(len(population) - len(new_population))
            for x in range(new):                                # 1/4 new
                new_population.append(self.getNewChromosome())


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
        self.numGen0 = 2 ** self.numGen
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
                avgScoreChromosome = self.AVGFitness(self.generation[x],x)
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
            k = len(self.generation)
            self.generation = self.CrossingFixedPopulation(self.bestChromosomeSearch(population, k), k, i)
            # self.generation = self.CrossingFixedPopulation(self.generation, i)
            # self.generation = self.CrossingGeneticBeamPopulation(population, i)
            for x in range(len(population)):
                self.population.append(population[x])
            if len(self.population) == 1:
                break
            else:
                continue
        file = open(filename_CGBP, 'w')
        file.close()
        self.save(filename_CGBP, self.generation)
        print("Training finished!")