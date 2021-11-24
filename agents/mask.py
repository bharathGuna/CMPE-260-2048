'''Code relating to the mask learning agents use to understand the game'''
import numpy
from abc import ABC, abstractmethod


class Mask(ABC):
    '''Abstract class identifing the functions a mask class needs'''

    def __init__(self, name, boardSize=4, maxTile=15):
        '''Init the mask class
        input:
            name: Name of mask used for tag
            boardSize: Size of the board
            maxTile: log2 of max tile that can appear on board'''
        self.name = name
        self.boardSize = boardSize
        self.maxTile = maxTile

    @abstractmethod
    def getNumTuples(self):
        '''Return the number of tuples used to describe each state'''
        pass

    @abstractmethod
    def getMaxTupleNum(self):
        '''Return the largest number corresponding to a tuple'''
        pass

    @abstractmethod
    def getTupleNums(self, state):
        '''Transforms a state into its tuple number representation
        input:
            state: state to transform
        output: array of tuple nums corresponding to state'''
        pass

    def getTag(self):
        '''Return tag of mask'''
        return self.name

    def getBoardSize(self):
        '''Return the boardSize'''
        return self.boardSize


class Mask_rxcx4(Mask):
    '''Mask that analyzes the states by looking at each row, column, and 4x4 squares.'''

    def __init__(self, name='4x4x4', boardSize=4, maxTile=12):
        '''Init the mask class
        input:
            name: Name of mask used for tag
            boardSize: Size of the board
            maxTile: log2 of max tile that can appear on board'''
        super().__init__(name, boardSize, maxTile)
        # Define some flags to specify the tuple types
        self.row_flag = 0
        self.column_flag = 1
        self.square_flag = 2

    def getNumTuples(self):
        '''Return the number of tuples used to describe each state'''
        return 2*self.boardSize + (self.boardSize-1)**2

    def getMaxTupleNum(self):
        '''Return the largest number corresponding to a tuple'''
        return self.stateToTupleNum(numpy.full((self.boardSize,
                                    self.boardSize), self.maxTile),
                                    self.square_flag,(self.boardSize-1)**2-1)

    def getTupleNums(self, state):
        '''Transforms a state into its tuple number representation
        input:
            state: state to transform
        output: array of tuple nums corresponding to state'''
        tupleNums = numpy.zeros(self.getNumTuples(), dtype=numpy.int)
        index = 0
        # For each tuple type
        for tupleType in range(self.square_flag+1):
            # If tuple type is row or column there will be boardSize tuples
            if tupleType < self.square_flag:
                maxNum = self.boardSize
            # If tuple type is square there will be (self.boardSize - 1)**2
            # tuples
            else:
                maxNum = (self.boardSize - 1)**2
            # For each tuple of this type
            for tupleIndex in range(maxNum):
                # Record this tuple num
                tupleNums[index] = self.stateToTupleNum(state, tupleType, tupleIndex)
                # Increment index
                index += 1
        return tupleNums

    def stateToTupleNum(self, state, tupleType, tupleIndex):
        '''Get specific tupleNum for a state. Accomplished by creating a base
        16 string where each digit represents a tile in the tuple and then
        transfoming to a decimal integer.
        input:
            state: State to transform
            tupleType: Type of tuple given by flag
            tupleIndex: Which tuple of this type we are looking for'''
        # The first digit of hex string is the tuple type
        # The seconde digit is the tuple index
        hexString = ''.join(['{:x}'.format(tupleType), '{:x}'.format(tupleIndex)])
        # The next for digits are the values in the tuple
        if(tupleType == self.row_flag):
            hexString += ''.join(['{:x}'.format(state[tupleIndex][col]) for col
                                  in range(self.boardSize)])
        elif(tupleType == self.column_flag):
            hexString += ''.join(['{:x}'.format(state[row][tupleIndex]) for row
                                  in range(self.boardSize)])
        elif(tupleType == self.square_flag):
            basex = tupleIndex % (self.boardSize - 1)
            basey = int(tupleIndex / (self.boardSize - 1))
            temp = hexString
            for i in range(2):
                for j in range(2):
                    temp += ''.join('{:x}'.format(state[basex + i][basey + j]))
            hexString += ''.join(['{:x}'.format(state[basex + i][basey + j])
                                  for i in range(2) for j in range(2)])
        # Return the hexString as as a decimal integer
        return int(hexString, base=16)
