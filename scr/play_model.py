from NNPlayer import NNPlayer
import CompromiseGame as cg

if __name__ == "__main__":
    player = NNPlayer(filename="200-random-player.npz")

    player2 = NNPlayer(filename="worst-nn.npz")

    game = cg.CompromiseGame(player, player2, 30, 10)
    #game = cg.CompromiseGame(player, cg.RandomPlayer(), 30, 10)
    #game = cg.CompromiseGame(player, cg.GreedyPlayer(), 30, 10)
    #game = cg.CompromiseGame(player, cg.SmartGreedyPlayer(), 30, 10)
   
    wins = 0
    games_to_play = 1000

    for _ in range(games_to_play):
        game.resetGame()
        score = game.play()
        if score[0] > score[1]:
            wins += 1
        #print(score)

    print(wins/games_to_play * 100)