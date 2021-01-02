import numpy as np
import nn as nn
import CompromiseGame as cg
import random

class NNPlayer(cg.AbstractPlayer):
    def __init__(self):
        self.brain = nn.NeuralNetwork([58,64,64,3], output_activation=nn.sigmoid)
        self.fitness = 0

    def play(self, my_state, opp_state, my_score, opp_score, turn, length, num_pips):
        nn_inputs = list(np.array(my_state).flatten()) + list(np.array(opp_state).flatten())
        nn_inputs.extend((my_score, opp_score, turn, length))
        return np.floor(self.brain.forward(nn_inputs) * 3).tolist()[0]

    def calculate_fitness(self, my_score, opp_score):
        self.fitness = (my_score * 2) - opp_score
        if my_score > opp_score:
            self.fitness += 5

class GeneticAlgorithm:
    def __init__(self, population_size=10, num_of_games=12):
        self.population = [NNPlayer() for _ in range(population_size)]
        self.games = [cg.CompromiseGame(cg.RandomPlayer(), cg.RandomPlayer(), 30, 10) for _ in range(num_of_games)]

    def play_random(self):
        random.shuffle(self.population)

        for player_a, player_b in zip(*[iter(self.population)]*2): # *[iter(s)*n] explanation here: https://www.reddit.com/r/learnpython/comments/bpsyjt/how_to_i_iterate_through_a_list_two_at_a_time/enxda01/
            scores = []

            for game in self.games:
                game.newPlayers(player_a, player_b)
                scores.append(game.play())

            player_a.fitness, player_b.fitness = self.calculate_fitness(scores)
            print(player_a.fitness, player_b.fitness)

    def _fitness_func(self, my_score, opp_score):
        change = (my_score * 2) - opp_score
        if my_score > opp_score:
            change += 5
        return change

    def calculate_fitness(self, scores):
        pA_fitness = 0
        pB_fitness = 0
        for score in scores:
            pA_fitness += self._fitness_func(score[0], score[1])
            pB_fitness += self._fitness_func(score[1], score[0])
        return pA_fitness, pB_fitness

    def choose_parents(self):
        fitnesses = np.array([p.fitness for p in self.population])
        norm_fitnesses = fitnesses / sum(fitnesses)
        return np.random.choice(self.population, len(self.population)//2, replace=False, p=norm_fitnesses)

if __name__ == "__main__":
    ga = GeneticAlgorithm(20)
    ga.play_random()
    parents = ga.choose_parents()

    for parent in parents:
        print(parent, parent.fitness)