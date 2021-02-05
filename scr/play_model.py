from NNPlayer import NNPlayer
import Player1817374
import CompromiseGame as cg

if __name__ == "__main__":
    player = NNPlayer(filename="1000-self-play-improved.npz")

    player2 = Player1817374.NNPlayer()

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