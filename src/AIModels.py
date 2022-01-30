import numpy as np

class RandomAI():

    def predict(self, input):
        output = np.random.random( input.shape )
        return output

class SeekAI():
    def __init__(self, mapSizeX, mapSizeY):
        self.mapSizeX = mapSizeX
        self.mapSizeY = mapSizeY

    def predict(self, input):
        return self.seek(input)

    def seek(self, input):
        shotMap = input.reshape(self.mapSizeX, self.mapSizeY)
        successfulHitsMap = np.array(shotMap > 0, dtype=int)
        missedShotsMap = np.array(shotMap < 0, dtype=int)

        surroundingCellsOfShotMap = np.zeros( (self.mapSizeX, self.mapSizeY) , dtype=int)
        
        # find current shot positions
        for x_index, X in enumerate(shotMap):
            for y_index, Y in enumerate(X):
                if Y == 1:
                    # above
                    if (y_index - 1 >= 0):
                        surroundingCellsOfShotMap[x_index, y_index-1] = 1
                    # below
                    if (y_index + 1 < self.mapSizeY):
                        surroundingCellsOfShotMap[x_index, y_index+1] = 1
                    # left
                    if (x_index - 1 >= 0):
                        surroundingCellsOfShotMap[x_index-1, y_index] = 1
                    # below
                    if (x_index + 1 < self.mapSizeX):
                        surroundingCellsOfShotMap[x_index+1, y_index] = 1

        alreadyShotMap = np.add(missedShotsMap, successfulHitsMap)
        newTargets = np.multiply(surroundingCellsOfShotMap, np.logical_not(alreadyShotMap), dtype=int).flatten()

        return np.array([newTargets])



class RandomSeekAI(SeekAI):
    
    def predict(self, input):
        newTargets = self.seek(input)

        if np.sum(newTargets) <= 0:
            return np.random.random( input.shape )

        return np.array(newTargets)

class SmartSeekAI(SeekAI):
    
    def __init__(self, mapSizeX, mapSizeY, AIGuessModel):
        super(SmartSeekAI, self).__init__(mapSizeX, mapSizeY)
        self.AIGuessModel = AIGuessModel

    def predict(self, input):
        newTargets = self.seek(input)

        if np.sum(newTargets) <= 0:
            return self.AIGuessModel.predict(np.array(input, dtype=float))

        return np.array(newTargets)