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
    def __init__(self, population_size=10, num_of_games=12, mutation_rate=0.01):
        self.population = [NNPlayer() for _ in range(population_size)]
        self.games = [cg.CompromiseGame(cg.RandomPlayer(), cg.RandomPlayer(), 30, 10) for _ in range(num_of_games)]
        self.mutation_rate = mutation_rate

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

    def crossover(self, parent_a, parent_b):
        pa_weights = parent_a.brain.get_weights()
        pb_weights = parent_b.brain.get_weights()
        shape = parent_a.brain.shape

        split = random.randint(0, len(pa_weights)-1)

        child1_weights = np.append(pa_weights[:split,], pb_weights[split:,])
        child2_weights = np.append(pb_weights[:split,], pa_weights[split:,])

        self.mutate(child1_weights)
        self.mutate(child2_weights)

        child1 = nn.NeuralNetwork(shape)
        child1.set_weights(child1_weights)

        child2 = nn.NeuralNetwork(shape)
        child2.set_weights(child2_weights)

    def mutate(self, weights):
        # Mask code found here: https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
        mask = np.random.choice([0, 1], size=weights.shape, p=((1 - self.mutation_rate), self.mutation_rate)).astype(np.bool)
        random_weights = 0.1 * np.random.randn(*weights.shape)
        weights[mask] = random_weights[mask]
        return weights

if __name__ == "__main__":
    ga = GeneticAlgorithm(20)
    ga.play_random()
    parents = ga.choose_parents()
    ga.crossover(parents[0], parents[1])

    for parent in parents:
        print(parent, parent.fitness)