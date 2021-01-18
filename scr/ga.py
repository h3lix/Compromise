"""A basic Genetic Algorithm library to be used to play the game "Compromise"
This is a project for the Data Science, Algorithms and Complexity module at the University of Warwick

This contains some basic functions that *should* be generalisable to other neural networks
"""

import numpy as np
import random

def generate_population(population_size, pop_obj):
    """
    A method to generate a population of objects

        Parameters:
            population_size (int): The size of the population
            pop_obj (Class): The class to instantiate throughout the population
        Returns:
            population (list): A list of pop_obj()
    """
    return [pop_obj() for _ in range(population_size)]

def get_normalised_fitnesses(population):
    """
    A method to get fitness values as probabilities

        Paramaters:
            population (list): A list of objects that have a fitness value
        Returns:
            normalised_fitnesses (list): A list of fitnesses that have been converted to probabilties
    """
    # Normalise fitness values using the softmax 
    fitnesses = np.array([p.fitness for p in population])
    exp_values = np.exp(fitnesses - np.max(fitnesses))
    return exp_values / np.sum(exp_values)

def select_mating_pool(population, num_parents):
    """
    A method for selecting parents in a population

        Parameters:
            population (list): A list of objects to be selected from based on fitness
            num_parents (int): The number of objects to keep and breed from
        Returns:
            parents (list): A list of remaining parents for breeding
    """
    # Select only the top parents based on fitness
    sorted_parents = sorted(population, key=lambda parent: parent.fitness, reverse=True)
    return sorted_parents[:num_parents]

    # Randomly select parents based on fitness, higher = more likely to be selected
    # This can result in lower fitness values being picked for gene diversity
    # This sometimes fails due to too many fitness values being negative therefore their probability for being selected is 0
    #fitnesses = get_normalised_fitnesses(population)
    #return np.random.choice(population, size=num_parents, replace=False, p=fitnesses)

def crossover(parents, num_children, mutation_rate, mutation_delta):
    """
    A method for breeding children from a set of parents by crossing over weights and biases

        Parameters:
            parents (list): A list of potential candidate parents, with a fitness value
            num_children (int): The numver of children to create
            mutation_rate (float): The amount of mutation that should occur to child's weights and biases
            mutation_delta (float): The intensity of mutations
        Returns
            children (list): A list of children generated from parents
    """
    children = []
    probabilities = get_normalised_fitnesses(parents)
    for _ in range(num_children):
        # Choose two random parents based on their fitness values, higher values = more likely to be picked
        parent_a, parent_b = np.random.choice(parents, size=2, p=probabilities)
        
        # Get the parents weights and biases as a 1D array
        pa_weights = parent_a.brain.get_weights()
        pb_weights = parent_b.brain.get_weights()

        pa_biases = parent_a.brain.get_biases()
        pb_biases = parent_b.brain.get_biases()

        shape = parent_a.brain.shape

        # Create split points for the weights and biases
        weight_split = random.randint(0, len(pa_weights)-1)
        biases_split = random.randint(0, len(pa_biases)-1)

        # Combine parent weights and biases at the split points as well as mutate them
        child_weights = mutate(np.append(pa_weights[:weight_split,], pb_weights[weight_split:,]), mutation_rate, mutation_delta)
        child_biases = mutate(np.append(pa_biases[:biases_split,], pb_biases[biases_split:,]), mutation_rate, mutation_delta)

        # Create a new child and set its weight and biases and add it to a children list
        child = parent_a.copy()

        child.brain.set_weights(child_weights)
        child.brain.set_biases(child_biases)

        children.append(child)

    return children

def mutate(weights, mutation_rate, mutation_delta):
    """
    A method to mutate a set of weights

        Parameters:
            weights (list): The weights to be mutated
            mutation_rate (float): The amount of mutation that should occur to child's weights and biases
            mutation_delta (float): The intensity of mutations
        Returns:
            weights (list): A mutated list of weights
    """
    # Mask code found here: https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
    mask = np.random.choice([0, 1], size=weights.shape, p=((1 - mutation_rate), mutation_rate)).astype(np.bool)
    random_weights = mutation_delta * np.random.randn(*weights.shape)
    weights[mask] += random_weights[mask]
    return weights