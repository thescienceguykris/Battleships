from random import Random
import sys
import matplotlib.pyplot as plt
import numpy as np

from GameObjects import Game
from AIModels import RandomAI, RandomSeekAI, SmartSeekAI
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

mapSize = {"x": 10, "y": 10}

def playerTurn(player, playerAI, draw=False, silent=True):
    if player.stillPlaying() == False:
        return
        
    # take the players current shot matrix and feed this into the
    # model we previously trained.
    # this is why it was important to have the model trained in various
    # states of play and not just the final state
    df = [player.getShotMatrix().flatten()]
    outputs = playerAI.predict(np.array(df, dtype=float))[0]

    # this prediction is outputted and saved to outputs. [0] is required as the predict
    # function returns an array of predictions
    
    # this gives the indexes of the predictions ranked highest to lowest
    rankedList = np.flip(np.argsort( outputs ))

    # this 1d index needs to be converted to a 2d rank

    # when flattened -> [1,2,3,4,5,0], say index 4 (so number 5) is the largest
    # this came from an array which looked like this [[1,2,3],[4,5,6]]
    # x_index = highestIndex (4) / mapSize (3) = 1.333 -> int -> 1
    # y_index = highestIndex (4) % mapSize (3) = 1

    # this loop prevents the AI from continuinly shooting in the same place
    # it propogates down the list until it finds somewhere that it thinks there's a ship
    # where it hasn't already shot.
    for shotIndex in rankedList:
        x_pos = int(shotIndex / mapSize["x"])
        y_pos = shotIndex % mapSize["y"]

        successful, alreadyShot = player.shootAt( ( x_pos, y_pos ) )
        
        if not silent:
            print("Shoot at:", x_pos, y_pos, "Hit" if successful else "Miss", "Repeat" if alreadyShot else "Unique")
        
        if alreadyShot == False:
            break
    
    # draw this turn
    if draw:
        plt.suptitle("Turn " + str(player.getNumberOfShotsFired()) )
        plt.subplot(131)
        plt.title("Player Shot Matrix")
        plt.imshow(player.getShotMatrix(), cmap='hot')
        plt.subplot(132)
        plt.title("Predicted Shots")
        plt.imshow(np.array(outputs).reshape(mapSize["x"],mapSize["y"]), cmap='hot')
        plt.subplot(133)
        plt.title("Actual Positions")
        plt.imshow(player.getShipMatrix(), cmap='hot')
        plt.show()


# play through a single game of battleship monitoring the AI's choices
def demonstrationGameLoop(playerAI):

    # generate a single player game
    game = Game(1)
    player = game.players[0]

    # run the game loop
    while game.playersStillPlaying():
        playerTurn( player, playerAI, draw=True, silent=False )
        
# this is useful for seeing how many turns the average model takes to win
def evaluationGameLoop(playerAI, numberOfPlayers=100, aiName="AI"):
    
    game = Game(numberOfPlayers)
    timeToWin = np.zeros(numberOfPlayers)
    round = 1

    print("Round", end=" ")
    
    # the game loop as before, but now looping over multiple players in the loop
    while game.playersStillPlaying():
        print(round, end=" ", flush=True)
        round += 1
        
        for player_id, player in enumerate(game.players):
            playerTurn(player, playerAI)
            
            # after the last shot, the player records how many times it took them to win
            if not player.stillPlaying():
                timeToWin[player_id] = player.getNumberOfShotsFired()

    print("")
    print("---")
    print( "Average Game Length:", np.mean(timeToWin), "+-", np.std(timeToWin))
    
    # a histogram is drawn showing the results
    fig, ax = plt.subplots(figsize=(8,6))
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.hist(timeToWin) 
    plt.style.use("ggplot")
    ax.grid(False)
    
    ax.set_xlabel("Average Game Length")

    ax.set_yticks([])

    # Overall #
    ax.set_title(aiName + " performance over " +  str(numberOfPlayers) + "\ngames of Battleship", pad = 10)
    # Remove ticks and spines
    ax.tick_params(left = False, bottom = False)
    for ax, spine in ax.spines.items():
        spine.set_visible(False)
    
    plt.savefig(aiName + ".png", dpi=400 )
        
if __name__ == "__main__":
    if not (len(sys.argv) >= 2):
        print("Incorrect usage of ai-play.py")
        print("ai-play.py [model name and extension] [optional number of games to play]")
        modelName = input("Model name: ")
        try:
            numberOfGames = int ( input("(Optional) Number of games to play: [1] ") )
        except:
            numberOfGames = 1
    else:
        try:
            modelName = str(sys.argv[1])
            numberOfGames = int(sys.argv[2])
        except:
            print("Incorrect usage of ai-play.py")
            print("ai-play.py [model name and extension] [number of games to play]")

model = load_model(modelName, compile=True)
randomAI = RandomAI()
randomSeekAI = RandomSeekAI(mapSize["x"], mapSize["y"])
smartSeekAI = SmartSeekAI(mapSize["x"], mapSize["y"], model)
print("Model loaded")
if numberOfGames == 1:
    demonstrationGameLoop(smartSeekAI)
    print("--")
    demonstrationGameLoop(randomAI)
else:
    #evaluationGameLoop(model, numberOfGames)
    #evaluationGameLoop(randomAI, numberOfGames, aiName="Random guess")
    #evaluationGameLoop(randomSeekAI, numberOfGames, aiName="Random Seek AI")
    evaluationGameLoop(smartSeekAI, numberOfGames, aiName="Smart Seek AI")
    