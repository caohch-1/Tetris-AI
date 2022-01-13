# AI for Tetris

The code for project "AI for Tetris" in CS181, ShanghaiTech University.

## Introduction

Tetris is one of the most popular video games. It is difficult to quantify and model for a satisfying strategy for human players, but the AI agent can be designed in multiple ways for this game.

In this work,we explore three Tetris AI agents:

Heuistic Agent: As a classic method dealing with search problem, it provide us with a basic but effective idea to get a higher score in Tetris

Genetic-Beam Agent: An algorithm simulates the natural evolution process to find the optimal solution.

Q-Learning Agent: An reinforcement learning based algorithm using Deep Q-Network (DQN).

### **What task does our code (method) solve?**

we design three AI agents for Tetris, including heuristics algorithm, genetic-beam search,and deep Q-learning, and conduct a comprehensive evaluation. The Comparison experiments show that the Q-Learning Agent has the best performance in general, which arrives 21887.584 for average scores, and 934.67 for average cleared lines by fixing the number of Tetrominoes to 2500.



## Code Structure

```
|--TetrisAI/
|   |--src/
|      |--arg_parser.py		/* Files for the implementation of arguments used to test parameters in Q-Learning agent
|      |--arg_genetic.py		/* Files for the implementation of arguments used to test parameters in Genentic-Beam agent
|      |--tetris.py     /* Files for the implementation of Tetris game
|      |--arg_parser_heur.py    /*Files for the implementation of arguments used to test parameters in heuristic agent
|   |--trained_models{K}/
|      |--tetris		/* Torch Model of MLP{K} as defined in our report
|   |--img/      /* Folder for images
|   |--genetic.py                /* Files for applying Genetic-Beam Search algorithm
|   |--GeneticTrain.py           /* Files for training best chromosomes for Tetris game
|   |--GeneticTest.py            /*  Files for testing scores for Tetris game
|   |--GeneticBestRunner.py             /* Files for demo to run Tetris game with the best chromosomes trained.
|   |--BestChromosome_CFP.txt    /* The trained chromosomes generated by GeneticTrain.py
|   |--BestChromosome_CGBP.txt   /* The trained chromosomes generated by GeneticTrain.py
|   |--heuristic_agent.py    /*File for heuristic agent
|   |--QLearningTrain.py    /* Train DQN Agent 
|   |--QLearningTest.py    /* Test DQN Agent
```



### Notations and Definitions

The notations used throughtout following sections are shown in Table.Ⅰ.

![table1](https://github.com/caohch-1/AI-Project/blob/main/img/table1.png)

Before diving into the detail of three AI agents, we briefly describe three definition used in the following algorithms.

__Holes__. We define an empty position as a hole if it is surrounded by non-zero numbers or the border of the board.

__Block height__. The distance from the lowest non-zero number to the highest non-zero number of each column is the height of the column.

__Bumpiness.__ The list of difference between the column heights of two adjacent columns.

![picture1](https://github.com/caohch-1/AI-Project/blob/main/img/picture1.png)



### Heuristic Agent

This agent uses 8 features to calculate a score for each action and choose the best action with the highest score to execute.

- Usage

```
python heuristic_agent.py --gui=1 --test_num=300 ## For detail See src/arg_parser_heur.py
```
If you want to change the weight of 8 features, add --w1=-1 --w2=-2... --w8=-5 to the command to change weights of each feature.
e.g. 
```
python heuristic_agent.py --gui=1 --test_num=300 --w1=-1 --w2=5
```


### Genetic-Beam Agent

The Genetic-Beam agent converts the solution process into a process similar to the crossover and mutation of the chromosomes in biological evolution. 

- Usage

To train the model,
```
python GeneticTrain.py --test_num=20       # For detail see src/arg_genetic.py
```
To test the model,
```
python GeneticTest.py --test_num=300 --time=300       # For detail see src/arg_genetic.py
```



### Q-Learning Agent

An reinforcement learning Agent using Deep Q-Network (DQN

- Usage

To train the model,
```
# Make sure saved_path is a existing directory
python QLearningTrain.py --model=32 --save_interval=500 --saved_path=trained_models32 --gui=1 # For detail See src/arg_parser.py
```

To test the model,
```
# Make sure saved_path is a existing directory
python QLearningTest.py --saved_path=trained_models32 --gui=1 --test_num=500 # # For detail See src/arg_parser.py
```
Notice that the usages above are for reference. To check the performances under different parameters, please see src/arg_parser.py for detail.


## Evaluation

**Experiment settings.**

To cater for the needs of our methods, we modify  and integrate some Tetris implementations on GitHub. To evaluate the three methods of Tetris agent, we set N=10, M=20 (i.e., a 10×20 grid) as our test environment.



In order to evaluate the efficiency of methods implementing Tetris agent, we choose three scoring criteria to quantify the performance of different methods. To see the full evaluation, please read our report article.

## Conclusion

We design three AI agents for Tetris, including heuristics algorithm, genetic-beam search,
and deep Q-learning, and conduct a comprehensive evaluation for the hyper-paramaters of each agent. We also explore the potential reasons for the effects of the hyper-parameters. Among there agents, both Genetic-Beam Agent and DQN Agent can achieve a great performance that hardly meet game over with proper hyper-parameters. The Comparison experiments show that the Q-Learning Agent has the best performance in general, which arrives 21887.584 for average scores, and 934.67 for average cleared lines by fixing the number of Tetrominoes to 2500.

