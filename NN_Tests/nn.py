import numpy as np

class Layer:
    """A layer that should lie within a larger neural network
    """
    def __init__(self, num_inputs, num_neurons):
        """
        Initialises a layer of neurons with random weights and biases

        Parameters:
            num_inputs (int): The number of inputs this layer should expect
            num_neurons (int): The number of neurons in this layer
        Returns:
            None
        """
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
    def forward(self, inputs):
        return np.dot(inputs, self.weights) + self.biases

class NeuralNetwork:
    def __init__(self, shape):
        self.layers = []
        for inputs, neurons in zip(shape, shape[1:]):
            self.layers.append(Layer(inputs, neurons))

    def forward(self, inputs):
        previous_output = inputs
        for layer in self.layers:
            previous_output = layer.forward(previous_output)
        return previous_output


X = [[1.0, 2.0, 3.0, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)

nn = NeuralNetwork([4,5,2])

print(nn.forward(X))