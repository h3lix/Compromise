import numpy as np

file_to_load = "200-random-player.npz"

with np.load(file_to_load) as model:
    np.savetxt("shape.txt", model['shape'], delimiter=',')
    np.savetxt("weights.txt", model['weights'], delimiter=',')
    np.savetxt("biases.txt", model['biases'], delimiter=',')