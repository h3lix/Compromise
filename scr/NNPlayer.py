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

    def calculate_fitness(self, my_score, opp_score):
        self.fitness = (my_score * 2) - opp_score
        if my_score > opp_score:
            self.fitness += 5

class GeneticAlgorithm:
    def __init__(self, population_size):
        self.population = [NNPlayer() for _ in range(population_size)]

    def play_random(self, num_of_games):
        games = [cg.CompromiseGame(pA, pB, 30, 10) for _ in range(num_of_games)]

        while len(self.population) > 1:
            random.shuffle(self.population)

            player_A = self.population.pop()
            player_B = self.population.pop()


if __name__ == "__main__":
    pA = NNPlayer()
    pB = NNPlayer()

    g = cg.CompromiseGame(pA, pB, 30, 10)

    print(g.play(), g.greenPlayer, g.redPlayer)

    g.newPlayers(cg.GreedyPlayer(), cg.RandomPlayer())

    print(g.play(), g.greenPlayer, g.redPlayer)

    '''
    population_size = 20
    population = [NNPlayer() for _ in range(population_size)]

    played = []

    while len(population) > 1:
        random.shuffle(population)


        pA = population.pop()
        pB = population.pop()
        g = cg.CompromiseGame(pA, pB, 30, 10)
        scores = g.play()

        pA.calculate_fitness(scores[0], scores[1])
        pB.calculate_fitness(scores[1], scores[0])

        played.extend((pA, pB))
        print(g.greenScore, g.redScore)

    fitnesses = np.array([p.fitness for p in played])
    norm_fitnesses = fitnesses / sum(fitnesses)
    print(fitnesses, norm_fitnesses)
    parents = np.random.choice(played, len(played)//2, replace=False, p=norm_fitnesses)

    for parent in parents:
        print(parent.fitness)

    #sorted_population = sorted(played, key=lambda player: -player.fitness)
    #for player in sorted_population:
    #    print(player.fitness)
    '''