import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

from GameObjects import Game
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

mapSize = {"x": 10, "y": 10}

def generateMaps(numberOfMapsToGenerate):

    game = Game(numberOfMapsToGenerate)

    # this will be where the actual ship maps will be stored, in AI the Y variable.
    trueShipMatrix = []

    # this will be a subset of the ship maps given to the AI, the X variable.
    givenShipMatrix = []

    print ("Generating Maps", end=" ")
    for player in game.players:
        
        # generate a shots fired overlay which will reveal some ships and give some missed shots
        # this creates a realistic game state that the AI might find itself in
        shotsFiredOverlay = (np.random.normal(loc=0.5, scale=2.0, size=(mapSize["x"], mapSize["y"]))) > 0.5
        shotsFiredOverlay = np.array(shotsFiredOverlay, dtype=int)
        
        # the location of the ships for this player, the noShips is the NOT(trueShips)
        # and indicates where the player could shoot and miss
        trueShips = np.array(player.getShipMatrix(), dtype=int)
        noShips = np.logical_not(trueShips)
        
        # the location of the ships is added to the whole dataset
        # trueShipMatrix.append(np.subtract(trueShips, shipsHit))
        trueShipMatrix.append(trueShips)
        
        # the shipsHit and missedShots array compare the overlay of shots and returns what they would and wouldn't hit
        # missed shots are represented by a -1 and successful hits a +1
        
        missedShots = (-1)*np.multiply(noShips,shotsFiredOverlay)
        shipsHit = np.multiply(trueShips, shotsFiredOverlay)

        # add the combined matrix to the AI training set
        givenShipMatrix.append(np.add(missedShots, shipsHit))
        
        # it might be worth teaching that there's no point shooting where you know you've already hit
        # shotsToScorePoints = trueShips - shipsHit = [0 1 1 0] - [1 1 0 0] = [-1 0 1 0]
        # this will only show ship points that haven't been hit
        
        print (".", end="")

    # prepare the data to be exported
    output_data = []
    for i in range(0, numberOfMapsToGenerate):
        # flatten to a 1d array rather than 2d matrix
        data = np.concatenate((trueShipMatrix[i].flatten(), givenShipMatrix[i].flatten()))
        output_data.append(data)

    return pd.DataFrame(data=output_data)

# split the data according to the exported format above
def splitData(df):
    trainTestDataSplit = 0.8
    numberToTrainOn = int(trainTestDataSplit * len(df.values))

    sizeOfBoard = mapSize["x"]*mapSize["y"]
    X = df.iloc[:, sizeOfBoard:]
    Y = df.iloc[:, :sizeOfBoard]

    return np.asarray(X[:numberToTrainOn]), np.asarray(X[numberToTrainOn:]), np.asarray(Y[:numberToTrainOn]), np.asarray(Y[numberToTrainOn:])


# build model
def buildModel(X, Y):

    model = Sequential()
    model.add(Dense( units=64, activation='relu', input_dim=len(X[0])))
    model.add(Dense( units=128, activation='relu'))
    model.add(Dense( units=len(Y[0]), activation='sigmoid'))

    model.compile( loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])

    return model

# train model
def trainModel(x_train, y_train, model):
    model.fit(x_train, y_train, epochs=150, batch_size=32)

# make predictions based on model
def modelPredictions(model, x_test):
    return model.predict( x_test )

# accuracy classification needs to convert the probabilities into definitive hits
# this could probably be done in a better way based upon how far along in the game the AI is
# this will mean that shots aren't wasted
def correctPredictions(y_test_predictions):
    corrected_predictions = []

    for prediction in y_test_predictions:
        
        highest_predictions = np.flip(np.sort(prediction))
        topScores = highest_predictions[5+4+3+2+2]
        corrected_predictions.append(list(map( lambda score: 1 if score >= topScores else 0, prediction )))
    
    return corrected_predictions

def evaluatePredictions(predictions, y_test):
    acc_score = accuracy_score ( y_test.flatten(), np.array(predictions).flatten() )
    print("Model Score:", acc_score)

# this could be setup as some form of experiments class

if __name__ == '__main__':
    if not (len(sys.argv) == 3):
        print("Incorrect usage of create-data.py")
        print("create-model.py [number of player maps] [output model file name]")
        numberOfMaps = int(input("Number of player maps: "))
        modelName = input("Model name: ")
    else:
        try:
            numberOfMaps = int(sys.argv[1])
            modelName = str(sys.argv[2])
        except:
            print("Incorrect usage of create-data.py")
            sys.exit("create-model.py [number of player maps] [output model file name]")

    df = generateMaps(numberOfMaps)
    x_train, x_test, y_train, y_test = splitData(df)
    model = buildModel(x_train, y_train)
    trainModel(x_train, y_train, model)

    predictions = modelPredictions(model, x_test)
    corrected = correctPredictions(predictions)
    evaluatePredictions(corrected, y_test)

    model.save(modelName + '.model')
