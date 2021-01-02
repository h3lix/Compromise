import numpy as np
import nn as nn
import CompromiseGame as cg
import random

class NNPlayer(cg.AbstractPlayer):
    def __init__(self):
        self.brain = nn.NeuralNetwork([59,64,64,3], output_activation_func=nn.sigmoid)
        self.fitness = 0

    def play(self, my_state, opp_state, my_score, opp_score, turn, length, num_pips):
        nn_inputs = list(np.array(my_state).flatten()) + list(np.array(opp_state).flatten())
        nn_inputs.extend((my_score, opp_score, turn, length, num_pips))
        return np.floor(self.brain.forward(nn_inputs) * 3).tolist()[0]

if __name__ == "__main__":
    population_size = 10
    population = [NNPlayer() for _ in range(population_size)]

    while len(population) > 1:
        played = []
        random.shuffle(population)


        pA = population.pop()
        pB = population.pop()
        g = cg.CompromiseGame(pA, pB, 30, 10)
        g.play()

        played.extend((pA, pB))
        print(g.greenScore, g.redScore)