from NNPlayer import NNPlayer
import CompromiseGame as cg
import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, population_size=10, num_of_games=12, mutation_rate=0.01):
        self.population = [NNPlayer() for _ in range(population_size)]
        self.games = [cg.CompromiseGame(cg.RandomPlayer(), cg.RandomPlayer(), 30, 10) for _ in range(num_of_games)]
        self.mutation_rate = mutation_rate
        self.max_fitness = -99999
        self.avg_fitness = 0

    def do_many_generations(self, num_of_generations):
        for i in range(num_of_generations):
            print(f"Generation: {i}, Avg Fitness: {self.avg_fitness/len(self.population)}, Max Fitness: {self.max_fitness}")
            self.max_fitness = -99999
            self.avg_fitness = 0
            self.do_generation()

    def do_generation(self):
        random.shuffle(self.population)

        self.play_greedy()

        self.population = self.choose_parents()
        random.shuffle(self.population)
        children = self.crossover(self.population, len(self.population))

        self.population += children

    def play_game(self):
        for player_a, player_b in zip(*[iter(self.population)]*2): # *[iter(s)*n] explanation here: https://www.reddit.com/r/learnpython/comments/bpsyjt/how_to_i_iterate_through_a_list_two_at_a_time/enxda01/
            scores = []

            for game in self.games:
                game.newPlayers(player_a, player_b)
                scores.append(game.play())

            player_a.fitness, player_b.fitness = self.calculate_fitness(scores)

    def play_greedy(self):
        for player in self.population:
            scores = []

            for game in self.games:
                game.newPlayers(player, cg.SmartGreedyPlayer())
                score = game.play()
                #if score[0] > score[1]:
                    #print(score)
                scores.append(score)

            player.fitness = self.calculate_fitness(scores)
            #print(player.fitness)

    def _fitness_func(self, my_score, opp_score):
        change = my_score - opp_score
        if my_score > opp_score:
            change += 20
        return change

    def calculate_fitness(self, scores):
        pA_fitness = 0
        #pB_fitness = 0
        for score in scores:
            pA_fitness += self._fitness_func(score[0], score[1])
            #pB_fitness += self._fitness_func(score[1], score[0])
        if pA_fitness > self.max_fitness:
            self.max_fitness = pA_fitness
        self.avg_fitness += pA_fitness
        #if pB_fitness > self.max_fitness:
        #    self.max_fitness = pB_fitness
        return pA_fitness#, pB_fitness

    def _get_normalised_fitnesses(self, population):
        fitnesses = np.exp(np.array([p.fitness for p in population]))
        return fitnesses / np.sum(fitnesses)

    def choose_parents(self):
        norm_fitnesses = self._get_normalised_fitnesses(self.population)
        return list(np.random.choice(self.population, size=len(self.population)//2, replace=False, p=norm_fitnesses))

    def crossover(self, parents, num_of_children):
        children = []
        for _ in range(num_of_children):
            parent_a, parent_b = np.random.choice(parents, size=2, replace=False, p=self._get_normalised_fitnesses(self.population))
        
            pa_weights = parent_a.brain.get_weights()
            pb_weights = parent_b.brain.get_weights()

            pa_biases = parent_a.brain.get_biases()
            pb_biases = parent_b.brain.get_biases()

            shape = parent_a.brain.shape

            weight_split = random.randint(0, len(pa_weights)-1)
            biases_split = random.randint(0, len(pa_biases)-1)

            child_weights = self.mutate(np.append(pa_weights[:weight_split,], pb_weights[weight_split:,]))
            child_biases = self.mutate(np.append(pa_biases[:biases_split,], pb_biases[biases_split:,]))

            child = NNPlayer(shape)
            child.brain.set_weights(child_weights)
            child.brain.set_biases(child_biases)

            children.append(child)

        return children

    def mutate(self, weights):
        # Mask code found here: https://stackoverflow.com/questions/31389481/numpy-replace-random-elements-in-an-array
        mask = np.random.choice([0, 1], size=weights.shape, p=((1 - self.mutation_rate), self.mutation_rate)).astype(np.bool)
        random_weights = 0.3 * np.random.randn(*weights.shape)
        weights[mask] *= random_weights[mask]
        return weights


if __name__ == "__main__":
    ga = GeneticAlgorithm(200, 12)
    ga.do_many_generations(1000)