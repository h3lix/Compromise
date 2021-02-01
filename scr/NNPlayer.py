"""A basic Neural Network player to be used to play the game "Compromise"
This is a project for the Data Science, Algorithms and Complexity module at the University of Warwick

This player contains some basic structures:
    NNPlayer - A player that can be trained to play "Compromise"
"""

import CompromiseGame as cg
import numpy as np
import nn as nn

class NNPlayer(cg.AbstractPlayer):
    """A player with a Neural Network brain to play the game "Compromise"
    """
    
    possible_moves = [[0, 0, 0], [0, 0, 1], [0, 0, 2], [0, 1, 0], [0, 1, 1], [0, 1, 2], [0, 2, 0], [0, 2, 1], [0, 2, 2], [1, 0, 0], [1, 0, 1], [1, 0, 2], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 2, 0], [1, 2, 1], [1, 2, 2], [2, 0, 0], [2, 0, 1], [2, 0, 2], [2, 1, 0], [2, 1, 1], [2, 1, 2], [2, 2, 0], [2, 2, 1], [2, 2, 2]]
    
    def __init__(self, shape=[27,64,64,27], hidden_activation=nn.relu, output_activation=nn.softmax):
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
        #nn_inputs = list(np.array(my_state).flatten()/3) + list(np.array(opp_state).flatten()/3)
        nn_inputs = np.array(my_state).flatten() - np.array(opp_state).flatten()
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
        return change / 2

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

        self.fitness = (self.fitness + (10 * self.games_won)) / len(self.scores)
        
        # The final fitness is the sum of all differences through all games, plus 10 * number of games won
        # divided by the number of opponents played (len(self.scores)) to get an average fitness
        return self.fitness

    def add_score(self, score):
        """A method for appending a score to this player's scores
        """
        self.scores.append(score)


