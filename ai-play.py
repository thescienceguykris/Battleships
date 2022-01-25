import sys
import matplotlib.pyplot as plt
import numpy as np

from GameObjects import Game
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

mapSize = {"x": 10, "y": 10}

def demonstrationGameLoop(playerAI):
    game = Game(1)
    fig = plt
    player = game.players[0]

    while game.playersStillPlaying():
        if player.stillPlaying() == False:
            continue
        
        df = [player.getShotMatrix().flatten()]
        outputs = playerAI.predict(np.array(df, dtype=float))[0]
        
        rankedList = np.flip(np.argsort( outputs ))

        # [[1,2,3],[4,5,6]] -> P(x=0,y=1) = 2...
        # -> [1,2,3,4,5,6], say index 4 is the largest
        # x = highestIndex (4) / mapSize (3) = 1.333 -> int -> 1
        # y = highestIndex (4) % mapSize (3) = 1

        for shotIndex in rankedList:
            x_pos = int(shotIndex / mapSize["x"])
            y_pos = shotIndex % mapSize["y"]

            successful, alreadyShot = player.shootAt( ( x_pos, y_pos ) )
            print("Shoot at:", x_pos, y_pos, "Hit" if successful else "Miss", "Repeat" if alreadyShot else "Unique")
            if alreadyShot == False:
                break
        
        fig.suptitle("Turn " + str(player.getNumberOfShotsFired()) )
        fig.subplot(131)
        fig.title("Player Shot Matrix")
        fig.imshow(player.getShotMatrix(), cmap='hot', interpolation='nearest')
        fig.subplot(132)
        fig.title("Predicted Shots")
        fig.imshow(np.array(outputs).reshape(mapSize["x"],mapSize["y"]), cmap='hot', interpolation='nearest')
        fig.subplot(133)
        fig.title("Actual Positions")
        fig.imshow(player.getShipMatrix(), cmap='hot', interpolation='nearest')
        fig.show()

def evaluationGameLoop(playerAI, numberOfPlayers=100):
    
    game = Game(numberOfPlayers)
    timeToWin = np.zeros(numberOfPlayers)
    round = 1

    print("Round", end=" ")
    while game.playersStillPlaying():
        print(round, end=" ", flush=True)
        round += 1
        
        for player_id, player in enumerate(game.players):
            if player.stillPlaying() == False:
                continue
            df = [player.getShotMatrix().flatten()]
            outputs = playerAI.predict(np.array(df, dtype=float))[0]
            
            rankedList = np.flip(np.argsort( outputs ))

            for shotIndex in rankedList:
                x_pos = int(shotIndex / mapSize["x"])
                y_pos = shotIndex % mapSize["y"]

                successful, alreadyShot = player.shootAt( ( x_pos, y_pos ) )
                if alreadyShot == False:    
                    break
            
            if not player.stillPlaying():
                timeToWin[player_id] = player.getNumberOfShotsFired()
                
    plt.hist(timeToWin)
    plt.show()
    print("Time to win:", np.mean(timeToWin))

if __name__ == "__main__":
    if not (len(sys.argv) >= 2):
        print("Incorrect usage of ai-play.py")
        print("ai-play.py [model name and extension] [optional number of games to play]")
        modelName = input("Model name: ")
        numberOfGames = int(input("(Optional) Number of games to play: [1]"))
    else:
        try:
            modelName = str(sys.argv[1])
            numberOfGames = int(sys.argv[2])
        except:
            print("Incorrect usage of ai-play.py")
            print("ai-play.py [model name and extension] [number of games to play]")

model = load_model(modelName, compile=True)
print("Model loaded")
demonstrationGameLoop(model)
#evaluationGameLoop(model, numberOfGames)