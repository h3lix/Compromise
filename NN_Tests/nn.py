"""A basic Neural Network library to be used to play the game "Compromise"
This is a project for the Data Science, Algorithms and Complexity module at the University of Warwick

This library contains some basic structures:
    Layer - A single layer within a larger NN
    NeuralNetwork - The NN as a whole, made up of many Layers
"""

import numpy as np

class Layer:
    """A layer within a larger neural network
    """
    def __init__(self, num_inputs, num_neurons):
        """
        Initialises a layer of neurons with random weights and biases

            Parameters:
                num_inputs (int): The number of inputs each neuron within this layer should expect
                num_neurons (int): The number of neurons in this layer
            Returns:
                None
        """
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    def forward(self, inputs):
        """
        A method to perform the forward pass of this layer within a neural network

            Parameters:
                inputs (list): A list of inputs to be processed by the neurons within thie layer
            Returns:
                dot_product (matrix): The dot/matrix product of the inputs with this layer's weights, then added biases
        """
        return np.dot(inputs, self.weights) + self.biases

class NeuralNetwork:
    """A neural network, built from many layers to solve some problem
    """
    def __init__(self, shape):
        """
        Initialises a neural network

            Parameters:
                shape (list): The shape of the neural network e.g. [2,4,3]
                    [2,4,3] will create a three layer network with 2 input nodes, 4 nodes in the hidden layer and 3 output nodes.
                    The shape should be of at least length 2 otherwise only input nodes will be created
            Returns:
                None
        """
        self.layers = []
        for inputs, neurons in zip(shape, shape[1:]):
            self.layers.append(Layer(inputs, neurons))

    def forward(self, inputs):
        """
        Performs the forward pass of the network

            Parameters:
                inputs (list): A list of inputs to be processed by the network. Must be of the same size as the input layer
                    e.g. in a [2,4,3] network, inputs must be of length 2
                    This network can handle batches, for example [[1,2], [3,4], [5,6]] would be accepted as 3 different inputs in a [2,4,3] network
            Returns:
                previous_output (matrix): The outputs of the network
        """
        previous_output = inputs
        for layer in self.layers:
            previous_output = layer.forward(previous_output)
        return previous_output


# X is usually used to denote the input data to a network
X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

nn = NeuralNetwork([4,5,2])

print(nn.forward(X))