"""A basic Neural Network player to be used to play the game "Compromise"
This is a project for the Data Science, Algorithms and Complexity module at the University of Warwick

This player contains some basic structures:
    NNPlayer - A player that can be trained to play "Compromise"
"""

import CompromiseGame as cg
import numpy as np
import nn as nn
import ga as ga
import random
import multiprocessing
import itertools

class NNPlayer(cg.AbstractPlayer):
    """A player with a Neural Network brain to play the game "Compromise"
    """
    
    possible_moves = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
    
    def __init__(self, shape=[54,128,27], hidden_activation=nn.relu, output_activation=nn.softmax):
        """
        Initialise a NNPlayer with a Neural Network brain

            Parameters:
                shape (list): The shape of the internal neural network
                hidden_activation (function): The function to be used for the hidden layers
                output_activation (function): The function to be used for the output layer
            Returns:
                None
        """
        self.brain = nn.NeuralNetwork(shape, hidden_activation, output_activation)
        self.fitness = 0
        self.games_won = 0
        self.scores = []

    def copy(self):
        """
        A method to create a copy of this NNPlayer

            Parameters:
                None
            Returns:
                obj (NNPlayer): An exact copy of this NNPlayer object
        """
        obj = NNPlayer(self.brain.shape, self.brain.hidden_activation, self.brain.output_activation)
        obj.brain.set_weights(self.brain.get_weights())
        obj.brain.set_biases(self.brain.get_biases())
        return obj

    def play(self, my_state, opp_state, my_score, opp_score, turn, length, num_pips):
        """
        A method to get a move for any state in the Compromise game

            Parameters:
                my_state (3D-list): The current state of this players pips in a 3D list
                opp_state (3D-list): The current state of the opponents pips in a 3D list
                my_score (int): This player's current score
                opp_score (int): The opponent's current score
                turn (int): The turn number
                length (int): Total number of turns in the game
                num_pips (int): The number of pips to be placed each round
            Returns:
                move (list): The move that this player would like to take in the format [1,1,2]
        """
        nn_inputs = list(np.array(my_state).flatten()) + list(np.array(opp_state).flatten())
        #nn_inputs.extend((my_score, opp_score, turn, length))
        return self.possible_moves[np.argmax(self.brain.forward(nn_inputs))]

    def _fitness_func(self, my_score, opp_score):
        """
        The fitness function for calculating the added fitness for a given score

            Parameters:
                my_score (int): This player's final score for a singular game of Compromise
                opp_score (int): The opponent's final score for a singular game of Compromise
            Returns:
                change (int): The difference between my_score and opp_score
        """
        # Find the difference in the scores (i.e. a player that wins by 30 points is better than one who wins by 1)
        change = my_score - opp_score
        if my_score > opp_score:
            self.games_won += 1
        return change

    def calc_fitness(self):
        """
        A method to calculate the fitness of an individual

            Parameters:
                None
            Returns:
                fitness (float): The fitness of this individual
        """
        self.games_won = 0
        self.fitness = 0

        for score in self.scores:
            for game in score:
                self.fitness += self._fitness_func(game[0], game[1])
        
        # The final fitness is the sum of all differences through all games, plus 10 * number of games won
        # divided by the number of opponents played (len(self.scores)) to get an average fitness
        return (self.fitness + (10 * games_won)) / len(self.scores)

    def add_score(self, score):
        """A method for appending a score to this player's scores
        """
        self.scores.append(score)

def play_game(player_a, player_b, num_games=11):
    """
    A method for playing a number of games between two players

        Parameters:
            player_a (NNPlayer): A player to play Compromise
            player_b (AbstractPlayer): Any player that can play Compromise
            num_games (int): The number of games for these two players to play
        Returns:
            player_a (NNPlayer): The player after it has played many games and added it's scores
            'CONDITIONAL' player_b (NNPlayer): If player_b was a NNPlayer then return it as well
    """
    game = cg.CompromiseGame(player_a, player_b, 30, 10)
    scores = []

    for _ in range(num_games):
        game.resetGame()
        scores.append(game.play())

    player_a.add_score(scores)

    if isinstance(player_b, NNPlayer):
        player_b.add_score(np.flip(scores).tolist())

        return player_a, player_b
    
    return player_a
        
def self_play(population, num_games, num_opponents):
    """
    A method to play a population of players against themselves

        Parameters:
            population (list): A list of NNPlayer's to play
            num_games (int): The number of games to play against each opponent
            num_oppopnents (int): The number of opponents to play against
        Returns:
            population (list): A final list of NNPlayers
    """
    with multiprocessing.Pool() as pool:
        for _ in range(num_opponents):
            random.shuffle(population)
            
            population = np.array(pool.starmap(play_game, zip(*[iter(population)]*2))).flatten()

    return population

def play_against(population, opponent, num_games, num_opponents):
    """
    A method to play a population against a specified opponent

        Parameters:
            population (list): A list of NNPlayer's to play
            opponents (AbstractPlayer): The opponent to play against
            num_games (int): The number of games to play against each opponent
            num_oppopnents (int): The number of opponents to play against
        Returns:
            population (list): A final list of NNPlayers
    """
    with multiprocessing.Pool() as pool:
        for _ in range(num_opponents):
            arguments = map(lambda player: (player, opponent, num_games), population)

            population = np.array(pool.starmap(play_game, arguments)).flatten()

    return population

if __name__ == "__main__":
    generations = 10
    population_size = 10
    num_games = 11
    num_opponents = 5
    mutation_rate = 0.05
    mutation_delta = 0.1

    population = ga.generate_population(population_size, NNPlayer)

    opponents = [cg.RandomPlayer(), cg.GreedyPlayer(), cg.SmartGreedyPlayer()]

    #game = cg.CompromiseGame(cg.RandomPlayer(), cg.RandomPlayer(), 30, 10)
    #sgp = cg.RandomPlayer()
    
    for gen in range(generations):
        avg_fitness = 0
        avg_games = 0

        random.shuffle(population)

        if random.randint(0,100) <= 70:
            population = self_play(population, num_games, num_opponents)
        else:
            population = play_against(population, random.choice(opponents), num_games, num_opponents)

        for player in population:
            player.calc_fitness()
            avg_fitness += player.fitness
            avg_games += player.games_won
            player.scores = []

        avg_fitness = avg_fitness/len(population)
        avg_games = avg_games/len(population)

        population = sorted(population, key=lambda player: player.fitness, reverse=True)
        
        print(f"Generation: {gen}, Avg Fitness: {avg_fitness}, Avg Games Won: {avg_games}, Max Fitness: {population[0].fitness}, Max Games Won: {population[0].games_won}")

        parents = ga.select_mating_pool(population, len(population)//2)
        children = ga.crossover(parents, population_size-len(parents), mutation_rate, mutation_delta)

        population = np.append(parents, children)

    population = sorted(population, key=lambda player: player.fitness, reverse=True)
    population[0].brain.save("model-test.npz")
