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

def create_mask(weights, rate):
    # Mask code found here: https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
    return np.random.choice([0, 1], size=weights.shape, p=((1 - rate), rate)).astype(np.bool)

def scale_fitness(fitness, scale_factor, offset):
    """
    A method to scale fitness values in order to even the playing field for parent selection

        Parameters:
            fitness (float): The fitness value to scale
            scale_factor (float): The number to scale the fitness by
            offset (float): The number to offset fitnesses by
        Returns
            scaled_fitness (float): The fitness value after scaling
    """
    return scale_factor * fitness + abs(offset)

def get_normalised_fitnesses(population):
    """
    A method to get fitness values as probabilities

        Paramaters:
            population (list): A list of objects that have a fitness value
        Returns:
            normalised_fitnesses (list): A list of fitnesses that have been converted to probabilties
    """
    # Normalise fitness values using the softmax 
    minimum = min(population, key=lambda player: player.fitness).fitness
    fitnesses = np.array([scale_fitness(p.fitness, 1/2, minimum) for p in population])
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
    #sorted_parents = sorted(population, key=lambda parent: parent.fitness, reverse=True)
    #return sorted_parents[:num_parents]

    # Randomly select parents based on fitness, higher = more likely to be selected
    # This can result in lower fitness values being picked for gene diversity
    # This sometimes fails due to too many fitness values being negative therefore their probability for being selected is 0
    # This is negated by fitness scaling to ensure that the range between the highest and lowest fitness is smaller
    fitnesses = get_normalised_fitnesses(population)
    return np.random.choice(population, size=num_parents, replace=False, p=fitnesses)

def select_parents(parents, probabilities):
    """
    A method for selecting a number of parents based on a list of probabilities

        Parameters:
            parents (array): An array of parents to choose from
            probabilities (array): An array of probabilities, must be the same length as parents and add to 1
        Returns:
            parent_a (NNPlayer): An instance of NNPlayer 
            parent_b (NNPlayer): A second instance of NNPlayer
    """
    # Select two parents based on the list of probabilities (Scaled fitnesses)
    # replace=False ensures that parents picked will be different
    parent_a, parent_b = np.random.choice(parents, size=2, replace=False, p=probabilities)
    return parent_a, parent_b

def get_weights_and_biases(parent):
    """
    A method to quickly retrieve a parents weights and biases

        Parameters:
            parent (NNPlayer): The NNPlayer to retrieve the weights and biases from
        Returns:
            weights (array): The weights of the parent
            biases (array): The biases of the parent
    """
    return parent.brain.get_weights(), parent.brain.get_biases()

def create_child(base, weights, biases):
    """
    A method to quickly create a child object

        Parameters:
            base (Class): The base class to copy from
            weights (array): The weights to set the child to have
            biases (array): The biases to set the child to have
        Returns:
            child (Class): An instantiation of the 'base' class with the weights and biases parsed
    """
    child = base.copy()

    child.brain.set_weights(weights)
    child.brain.set_biases(biases)

    return child

def create_random_children(base_parent, num_children):
    """
    A method to create a set of random children

        Parameters:
            base_parent (Object): The base object to randomise
            num_children (int): The amount of children to create
        Returns:
            children (array): An array of random children
    """
    children = []
    for _ in range(num_children):
        random_child = base_parent.copy()
        random_child.brain.initialise_shape(random_child.brain.shape, 
                                            random_child.brain.hidden_activation, 
                                            random_child.brain.output_activation)
        children.append(random_child)
    return children

def crossover(parents, num_children, amount_to_randomise, crossover_func, **kwargs):
    """
    A cookie cutter crossover method

        Parameters:
            parents (array): The parents to crossover into offspring
            num_children (int): The number of children to create
            pct_to_randomise (int): Percentage of children that should be completely random
            crossover_func (function): The function for crossing over weights and biases
            crossover_rate (float): The rate at which crossover should occur
        Returns:
            children (array): An array of children to be added to the population
    """
    children = create_random_children(parents[0], amount_to_randomise)
    probabilities = get_normalised_fitnesses(parents)

    for _ in range(num_children - len(children)):
        parent_a, parent_b = select_parents(parents, probabilities)

        pa_weights, pa_biases = get_weights_and_biases(parent_a)
        pb_weights, pb_biases = get_weights_and_biases(parent_b)

        child_weights = crossover_func(pa_weights, pb_weights, **kwargs)
        child_biases = crossover_func(pa_biases, pb_biases, **kwargs)

        child = create_child(parent_a, child_weights, child_biases)

        children.append(child)

    return children

def single_point_crossover(pa_weights, pb_weights, **kwargs):
    """
    A method to perform single point crossover on a set of weights

        Parameters:
            pa_weights (array): The weights of parent_a
            pb_weights (array): The weights of parent_b
        Returns:
            weights (array): The crossed over weights of pa_weights and pb_weights
    """
    # Create split points for the weights
    split = random.randint(0, len(pa_weights)-1)

    # Combine parent weights at the split point
    return np.append(pa_weights[:split,], pb_weights[split:,])

def uniform_crossover(pa_weights, pb_weights, crossover_rate=0.5, **kwargs):
    """
    A method to perform uniform crossover on a set of weights

        Parameters:
            pa_weights (array): The weights of parent_a
            pb_weights (array): The weights of parent_b
            crossover_rate (float): The rate at which weights should be swapped (can bias towards one parent if wanted)
                default: 0.5
        Returns:
            weights (array): The crossed over weights of pa_weights and pb_weights
    """
    # Create an array of zeros and ones to act as a mask to switch weights
    mask = create_mask(pa_weights, crossover_rate)

    # Swap pa_weights and pb_weights based on mask
    pa_weights[mask] = pb_weights[mask]
    return pa_weights

def intermediate_crossover(pa_weights, pb_weights, crossover_rate=0.5, extra_range=0.25, **kwargs):
    """
    A method to perform intermediate crossover on a set of weights

        Parameters:
            pa_weights (array): The weights of parent_a
            pb_weights (array): The weights of parent_b
            crossover_rate (float): The rate at which weights will be crossed over as well as extended
            crossover_intensity (float): The extra range of weights that can be generated
        Returns:
            weights (array): The crossed over weights of pa_weights and pb_weights
    """
    # Intermediate Recombination found here: https://cs.gmu.edu/~sean/book/metaheuristics/Essentials.pdf
    # Create an array of zeros and ones to act as a mask
    mask = create_mask(pa_weights, crossover_rate)

    for i in range(len(pa_weights)):
        # Skip any that the mask disallows
        if mask[i] == 0:
            continue

        # Create an alpha value between -p and 1+p where p is the crossover rate
        alpha = random.uniform(-extra_range, 1 + extra_range)
        pa_weights[i] = alpha * pa_weights[i] + (1 - alpha) * pb_weights[i]

    return pa_weights

def mutate(children, mutation_rate, mutation_delta):
    """
    A method to mutate a set of weights

        Parameters:
            children (array): The children to be mutated
            mutation_rate (float): The amount of mutation that should occur to child's weights and biases
            mutation_delta (float): The intensity of mutations
        Returns:
            children (array): A mutated list of children
    """
    for child in children:
        weights, biases = get_weights_and_biases(child)
    
        weight_mask = create_mask(weights, mutation_rate)
        biases_mask = create_mask(biases, mutation_rate)

        # Create an array of random numbers between -1 and 1 in the shape of weights & biases
        random_weights = mutation_delta * np.random.uniform(-1, 1, *weights.shape)
        random_biases = mutation_delta * np.random.uniform(-1, 1, *biases.shape)

        weights[weight_mask] += random_weights[weight_mask]
        biases[biases_mask] += random_biases[biases_mask]

        child.brain.set_weights(weights)
        child.brain.set_biases(biases)

    return children