import random
import matplotlib.pyplot as plt
import numpy as np

mapSize = {"x": 10, "y": 10}

class Game():
    
    # Initialise a game object with a certain number of players.
    # Each player will be initialised and added to the game's players array.
    def __init__(self, numberOfPlayers):
        self.players = []
        for i in range(0, numberOfPlayers):
            self.players.append(Player())
            
    # This makes it easier to check whether any of the players in the game still have battleships
    # to sink. This allows a game loop to finish after everyone has finished.
    def playersStillPlaying(self):
        for player in self.players:
            if player.stillPlaying():
                return True
        return False

class Player():
    
    def __init__(self):
        self.ships = [Ship(3), Ship(5), Ship(4), Ship(2), Ship(2)]
        self.shotsFired = np.zeros( (mapSize["x"], mapSize["y"]), dtype=int )
    
    def getHits(self):
        hits = (self.shotsFired == 1)
        return np.count_nonzero(hits)

    def getNumberOfShotsFired(self):
        return np.count_nonzero(self.shotsFired)
    
    def shootAt(self, target):
        successfulHit = False
        alreadyShot = False
        
        x,y = target

        ## Check if already shot there
        if self.shotsFired[x,y] != 0:
            alreadyShot = True
        
        for ship in self.ships:
            if ship.shootAt(target):
                successfulHit = True
        
        self.shotsFired[x,y] = 1 if successfulHit else -1
        return successfulHit, alreadyShot

    def stillPlaying(self):
        shotsFired = self.getNumberOfShotsFired()
        maxShotsFired = mapSize["x"]*mapSize["y"]
        if (shotsFired>maxShotsFired):
            print("Game over:", shotsFired)
        return (self.shipsAlive() > 0) and (shotsFired < maxShotsFired) 
    
    def shipsAlive(self):
        alive = 0
        for ship in self.ships:
            alive += (1 if ship.isAlive() else 0)
        return alive
    
    def getShipMatrix(self):
        board = np.zeros((mapSize["x"],mapSize["y"]), dtype=bool)
        for ship in self.ships:
            board = np.logical_or(board, ship.generateMatrix())
        return board
    
    def getShotMatrix(self):
        return self.shotsFired

class Ship():
    
    def __init__(self, size):
        self.size = size
        self.generate()

    # generate is pulled as a seperate function from init to allow it to be recalled
    # if the ship generated isn't valid    
    def generate(self):
        self.isHorizontal = random.choice( (True, False) )
        self.directionOfMovement = random.choice( (-1, +1) )
        self.shipMatrix = np.zeros((mapSize["x"], mapSize["y"]), dtype=int)
        self.isHitPoints = []
        self.generatePoints()
        
    def generatePoints(self):
        
        # generate the starting co-ordinates for the ship and a zero-hit matrix
        startingXPos = random.randint(0, mapSize["x"]-1)
        startingYPos = random.randint(0, mapSize["y"]-1)
        self.isHitPoints = np.zeros((mapSize["x"], mapSize["y"]), dtype=int)
        
        if self.isHorizontal:
            # generate a matrix of the board length in the x-direction
            ship_length = np.zeros(mapSize["x"])
            # put the ship in this row
            ship_length[startingXPos:startingXPos+self.size] = 1
            # put the row back into the overall matrix
            self.shipMatrix[:, startingYPos] = ship_length
        else:
            ship_length = np.zeros(mapSize["y"])
            ship_length[startingYPos:startingYPos+self.size] = 1
            self.shipMatrix[startingXPos,:] = ship_length

        # check whether the length of the ship fits within the map, indicating a successful positioning of the ship
        if not (self.shipMatrix.sum() == self.size):
            self.generate()
        
    def shootAt(self, target):
        x,y = target
        
        if self.shipMatrix[x,y] == 1:
            self.isHitPoints[x,y] = True
            return True
        return False
    
    def isAlive(self):
        return self.isHitPoints.sum() < self.size
    
    def generateMatrix(self):
        return self.shipMatrix
    

