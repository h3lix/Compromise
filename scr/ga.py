import numpy as np
import random

def generate_population(population_size, pop_obj):
    return [pop_obj() for _ in range(population_size)]

def get_normalised_fitnesses(population):
    fitnesses = np.array([p.fitness for p in population])
    exp_values = np.exp(fitnesses - np.max(fitnesses))
    return exp_values / np.sum(exp_values)

def select_mating_pool(population, num_parents):
    sorted_parents = sorted(population, key=lambda parent: parent.fitness, reverse=True)
    return sorted_parents[:num_parents]
    #fitnesses = get_normalised_fitnesses(population)
    #return np.random.choice(population, size=num_parents, replace=False, p=fitnesses)

def crossover(parents, num_children, mutation_rate, mutation_delta):
    children = []
    for _ in range(num_children):
        parent_a, parent_b = np.random.choice(parents, size=2, p=get_normalised_fitnesses(parents))
        
        pa_weights = parent_a.brain.get_weights()
        pb_weights = parent_b.brain.get_weights()

        pa_biases = parent_a.brain.get_biases()
        pb_biases = parent_b.brain.get_biases()

        shape = parent_a.brain.shape

        weight_split = random.randint(0, len(pa_weights)-1)
        biases_split = random.randint(0, len(pa_biases)-1)

        child_weights = mutate(np.append(pa_weights[:weight_split,], pb_weights[weight_split:,]), mutation_rate, mutation_delta)
        child_biases = mutate(np.append(pa_biases[:biases_split,], pb_biases[biases_split:,]), mutation_rate, mutation_delta)

        child = parent_a.copy()

        child.brain.set_weights(child_weights)
        child.brain.set_biases(child_biases)

        children.append(child)

    return children

def mutate(weights, mutation_rate, mutation_delta):
    # Mask code found here: https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
    mask = np.random.choice([0, 1], size=weights.shape, p=((1 - mutation_rate), mutation_rate)).astype(np.bool)
    random_weights = mutation_delta * np.random.randn(*weights.shape)
    weights[mask] += random_weights[mask]
    return weights