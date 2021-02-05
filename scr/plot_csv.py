import matplotlib.pyplot as plt
import csv

generations=[]
average_fitness=[]
average_games=[]
max_fitness=[]
max_games=[]


with open('200-generations.csv', 'r') as csvfile:
    plots= csv.reader(csvfile, delimiter=',')
    for row in plots:
        generations.append(int(row[0]))
        average_fitness.append(float(row[1]))
        average_games.append(float(row[2]))
        max_fitness.append(float(row[3]))
        max_games.append(float(row[4]))


plt.plot(generations,average_games, label = "Average Games Won")
plt.plot(generations,max_games, label = "Maximum Games Won")

plt.title('Data from training against RandomPlayer() for 200 generations.')

plt.xlabel('Generation')
plt.ylabel('Games Won')

plt.legend()
plt.show()
