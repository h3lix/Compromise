import CompromiseGame as cg
from NNPlayer import NNPlayer
import ga
import numpy as np
import random
import multiprocessing
import itertools

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
    generations = 1000
    population_size = 100
    num_games = 11
    num_opponents = 10
    max_games = num_games * num_opponents
    mutation_rate = 0.05
    mutation_delta = 1

    population = ga.generate_population(population_size, NNPlayer)

    opponents = [cg.RandomPlayer(), cg.GreedyPlayer(), cg.SmartGreedyPlayer()]

    #game = cg.CompromiseGame(cg.RandomPlayer(), cg.RandomPlayer(), 30, 10)
    #sgp = cg.RandomPlayer()
    
    for gen in range(generations):
        avg_fitness = 0
        avg_games = 0

        random.shuffle(population)

        #if random.randint(0,100) <= 40:
        population = self_play(population, num_games, num_opponents)
        #else:
        #population = play_against(population, cg.SmartGreedyPlayer(), num_games, num_opponents)

        for player in population:
            player.calc_fitness()
            avg_fitness += player.fitness
            avg_games += player.games_won
            player.scores = []

        avg_fitness = avg_fitness/len(population)
        avg_games = avg_games/len(population)

        population = sorted(population, key=lambda player: player.fitness, reverse=True)
        max_games_won = population[0].games_won
        
        print(f"Generation: {gen}, Avg Fitness: {avg_fitness}, Avg Games Won: {avg_games}, Max Fitness: {population[0].fitness}, Max Games Won: {max_games_won}")

        # Alter mutation intensity based on closeness to maximum solution (i.e. big mutations at start, small when nearing solution)
        #current_mutation_delta = mutation_delta * (1 - (max_games_won / max_games))
        current_mutation_delta = mutation_delta * (1 - (gen / generations))

        parents = ga.select_mating_pool(population, len(population)//2)
        children = ga.crossover(parents, population_size-len(parents), 10, ga.uniform_crossover, crossover_rate=0.5, extra_range=0.25)
        children = ga.mutate(children, mutation_rate, current_mutation_delta)

        population = np.append(parents, children)

    population = sorted(population, key=lambda player: player.fitness, reverse=True)
    population[0].brain.save("best-nn.npz")
    population[-1].brain.save("worst-nn.npz")