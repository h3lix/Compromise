import numpy as np
import nn as nn
import ga as ga
import CompromiseGame as cg
import random

class NNPlayer(cg.AbstractPlayer):
    
    possible_moves = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
    
    def __init__(self, shape=[54,32,32,32,27], output_activation=nn.softmax):
        self.shape = shape
        self.output_activation = output_activation
        self.brain = nn.NeuralNetwork(shape, output_activation=output_activation)
        self.fitness = 0
        self.games_won = 0
        self.scores = []

    def copy(self):
        obj = NNPlayer(self.shape, self.output_activation)
        obj.brain.set_weights(self.brain.get_weights())
        obj.brain.set_biases(self.brain.get_biases())
        return obj

    def play(self, my_state, opp_state, my_score, opp_score, turn, length, num_pips):
        nn_inputs = list(np.array(my_state).flatten()) + list(np.array(opp_state).flatten())
        #nn_inputs.extend((my_score, opp_score, turn, length))
        return self.possible_moves[np.argmax(self.brain.forward(nn_inputs))]

    def _fitness_func(self, my_score, opp_score):
        change = my_score - opp_score
        if my_score > opp_score:
            self.games_won += 1
            return change + 10
        return change

    def calc_fitness(self):
        self.games_won = 0
        self.fitness = 0

        for score in self.scores:
            for game in score:
                self.fitness += self._fitness_func(game[0], game[1])
        return self.fitness/len(self.scores)

    def add_score(self, score):
        self.scores.append(score)
        
def self_play(population, num_games, num_opponents):
    game = cg.CompromiseGame(cg.RandomPlayer(), cg.RandomPlayer(), 30, 10)
    for _ in range(num_opponents):
        random.shuffle(population)
        for player_a, player_b in zip(*[iter(population)]*2):
            scores = []

            game.newPlayers(player_a, player_b)

            for g in range(num_games):
                game.resetGame()
                scores.append(game.play())

            player_a.add_score(scores)
            player_b.add_score(np.flip(scores))

    for player in population:
        player.calc_fitness()

    return population

def play_against(population, opponent, num_games):
    game = cg.CompromiseGame(cg.RandomPlayer(), cg.RandomPlayer(), 30, 10)
    for player in population:
        scores = []

        game.newPlayers(player, opponent)

        for g in range(num_games):
            game.resetGame()
            scores.append(game.play())

        player.calc_fitness(scores)

    return population

if __name__ == "__main__":
    generations = 100
    population_size = 100
    num_games = 11
    num_opponents = 10
    mutation_rate = 0.05

    population = ga.generate_population(population_size, NNPlayer)

    game = cg.CompromiseGame(cg.RandomPlayer(), cg.RandomPlayer(), 30, 10)
    sgp = cg.RandomPlayer()
    
    for gen in range(generations):

        random.shuffle(population)

        population = self_play(population, num_games, num_opponents)
        population = sorted(population, key=lambda player: player.fitness, reverse=True)

        avg_fitness = 0
        avg_games = 0

        for player in population:
            avg_fitness += player.fitness
            avg_games += player.games_won
            player.scores = []

        avg_fitness = avg_fitness/len(population)
        avg_games = (avg_games/len(population))/num_opponents
        
        print(f"Generation: {gen}, Avg Fitness: {avg_fitness}, Avg Games Won: {avg_games}, Max Fitness: {population[0].fitness}, Max Games Won: {population[0].games_won}")

        parents = ga.select_mating_pool(population, len(population)//5)
        children = ga.crossover(parents, population_size-len(parents), mutation_rate)

        population = np.append(parents, children)

    population = sorted(population, key=lambda player: player.fitness, reverse=True)
    population[0].brain.save("best-model.npz")
