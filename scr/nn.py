"""A basic Neural Network library to be used to play the game "Compromise"
This is a project for the Data Science, Algorithms and Complexity module at the University of Warwick

This library contains some basic structures:
    Layer - A single layer within a larger NN
    NeuralNetwork - The NN as a whole, made up of many Layers

The code from Neural Networks from Scratch (NNFS) part 6 was used as a base to build further from:
https://www.youtube.com/watch?v=omz_NdFgWyU
"""

import numpy as np

def sigmoid(inputs):
    """Execute the sigmoid ( 1 / (1+e^-x) ) function element-wise on a numpy array"""
    ### Sigmoid tends not to be used much now, however is implemented here for experimentation purposes anyway
    return 1 / (1 + np.exp(-inputs))

def relu(inputs):
    """Execute the rectified linear (ReLU) function on a numpy array"""
    return np.maximum(0, inputs)
    
def softmax(inputs):
    """Execute the softmax function on a numpy array to convert values to probabilities"""
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)

class Layer:
    """A layer within a larger neural network
    """
    def __init__(self, num_inputs, num_neurons, activation_func=relu):
        """
        Initialises a layer of neurons with random weights and biases

            Parameters:
                num_inputs (int): The number of inputs each neuron within this layer should expect
                num_neurons (int): The number of neurons in this layer
                activation_func (function): The function to be used for activation
                    default: relu
            Returns:
                None
        """
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.set_activation(activation_func)

    def set_activation(self, activation_func):
        """Sets the activation function for this layer"""
        self.activate = activation_func

    def forward(self, inputs):
        """
        A method to perform the forward pass of this layer within a neural network

            Parameters:
                inputs (list): A list of inputs to be processed by the neurons within this layer
            Returns:
                dot_product (matrix): The dot/matrix product of the inputs with this layer's weights, then added biases
        """
        return self.activate(np.dot(inputs, self.weights) + self.biases)


class NeuralNetwork:
    """A neural network, built from many layers to solve some problem
    """
    def __init__(self, shape, hidden_activation=relu, output_activation=softmax):
        """
        Initialises a neural network

            Parameters:
                shape (list): The shape of the neural network e.g. [2,4,3]
                    [2,4,3] will create a three layer network with 2 input nodes, 4 nodes in the hidden layer and 3 output nodes.
                    The shape should be of at least length 2 otherwise only input nodes will be created
                hidden_activation (function): The activation function to use throughout the hidden layers
                    default: relu
                output_activation (function): The activation function to use for the output layer
                    default: softmax
            Returns:
                None
        """
        self.layers = []
        self.initialise_shape(shape, hidden_activation, output_activation)

    def initialise_shape(self, shape, hidden_activation=relu, output_activation=softmax):
        """
        Initialises a set of layers in the specified shape

            Parameters:
                shape (list): The shape of the neural network e.g. [2,4,3] creates 2 input nodes, 4 hidden nodes and 3 output nodes
                hidden_activation (function): The activation function to be used throughout the network
                    default: relu
                output_activation(function): The activation function to be used for the output
                    default: softmax
            Returns:
                None
        """
        self.shape = shape
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layers = []

        for inputs, neurons in zip(shape, shape[1:]):
            self.layers.append(Layer(inputs, neurons, hidden_activation))

        if output_activation is not None:
            self.layers[-1].set_activation(output_activation)

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

    def get_weights(self):
        """
        Flattens the networks weights into a 1 dimensional array

                Parameters:
                    None
                Returns:
                    flattened_weights (array): The weights of the network in a 1D array
        """
        flattened_weights = np.array([])
        for layer in self.layers:
            flattened_weights = np.append(flattened_weights, layer.weights.flatten())
        return flattened_weights

    def set_weights(self, weights):
        """
        Unflattens a 1 dimensional array of weights into the network

            Parameters:
                weights (array): A 1D array of weights to be set in the network
            Returns:
                None
        """
        len_expected = len(self.get_weights())
        if len(weights) != len_expected:
            print("Input weights do not match expected length")
            return

        index = 0
        for layer in self.layers:
            shape = layer.weights.shape
            size = np.product(shape)
            layer.weights = weights[index:index+size].reshape(shape)
            index += size

    def get_biases(self):
        """
        Flattens the networks biases into a 1 dimensional array

                Parameters:
                    None
                Returns:
                    flattened_biases (array): The biases of the network in a 1D array
        """
        flattened_biases = np.array([])
        for layer in self.layers:
            flattened_biases = np.append(flattened_biases, layer.biases.flatten())
        return flattened_biases

    def set_biases(self, biases):
        """
        Unflattens a 1 dimensional array of biases into the network

            Parameters:
                biases (array): A 1D array of biases to be set in the network
            Returns:
                None
        """
        len_expected = len(self.get_biases())
        if len(biases) != len_expected:
            print("Input biases do not match expected length")
            return

        index = 0
        for layer in self.layers:
            shape = layer.biases.shape
            size = np.product(shape)
            layer.biases = biases[index:index+size].reshape(shape)
            index += size

    def save(self, filename):
        """
        Saves the neural networks shape, weights and biases to a file

            Parameters:
                filename (string): Name of file for the model to be saved to
            Returns:
                None
        """
        np.savez(filename, shape=self.shape,
                           weights=self.get_weights(), 
                           biases=self.get_biases())

    def load(self, filename):
        """
        Loads a saved neural network from a file, including shape, weights and biases
        
            Parameters:
                filename (string): Name of file to load the model from
            Returns:
                None
        """
        with np.load(filename) as model:
            self.initialise_shape(model['shape'])
            self.set_weights(model['weights'])
            self.set_biases(model['biases'])

if __name__ == "__main__":
    # X is usually used to denote the input data to a network
    X = [[1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]

    nn = NeuralNetwork([4,4,4,4])

    print(nn.forward(X))