
# Code from Chapter 13 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# The basic TD(0) algorithm with the Europe example

from numpy import *
from gameBoard import *
from random import sample

def randomMove(state):
    
    """Uses the enum_actions method to find all available positions on 
    the game board, then randomly chooses which position to take."""
    
    states = enum_actions(state)
    movesLeft = len(states)
    move = random.randint(0, movesLeft)
    state[states[move]] = 1
    
def enum_actions(state):
    
    """Method that returns all available positions on the game board."""
    
    res = list()
    for i in xrange(3):
        for j in xrange(3):
            if state[i,j] == 0:
                res.append( (i,j) )
    return res

def value(board, action, Q):
    
    """Instead of an action state pair, I have combined them into a hash (by 
    just making the move and hashing the board, reversing the move afterwareds)
    So I can now use a dictionary to store the action/state values"""
    
    board[action] = 2
    hashval = hash(board)
    val = Q.get(hashval)
    if val == None:
        if board.won(2): val = 1.0
        elif board.full(): val = 0.0
        else: val = 0.1
        Q[hashval] = val    
    board[action] = 0
    return val

def next_action(board, Q):
    
    """Find the best move we could make given the current board, and
    dictionary Q, by trying all current moves and returning the value
    of each, sorting the values/action pairs by value with a lambda 
    function"""
    
    valuemap = list()
    for action in enum_actions(board):
        val = value(board, action, Q)
        valuemap.append( (val, action) )
    valuemap.sort(key=lambda x:x[0], reverse=True)
    maxval = valuemap[0][0]
    valuemap = filter(lambda x: x[0] >= maxval, valuemap)
    
    return sample(valuemap,1)[0]

def next(board, lastboard_hash, Q, learningRate, epsilon, gamma, runningTotal):
    
    """Pretty messy method with all those args, but nonetheless, this is the 
    main method for making moves and learning each decision."""
    
    if board.won(1):
        val = -1
    elif board.full():
        val = -0.1
    else:
        
        "epsilon-greedy decision"
        
        if (random.rand()<epsilon):            
            movesLeft = len(enum_actions(board))
            randomMove = random.randint(0, movesLeft)            
            action = enum_actions(board)[randomMove]
            val = value(board, action, Q)
            
        else:
            (val, action) = next_action(board, Q)
            
        board[action] = 2    
     
    #learning step
    if lastboard_hash != None:
        
        """Keep the running total so far, then calculate the new value estimate,
        using the Q learning function"""
        
        runningTotal += val
        Q[lastboard_hash] += learningRate * (runningTotal - Q[lastboard_hash])
        
    return hash(board)
    


def TicTacToe():
    
    """Main runnable method, the random player always goes first, this is so
    the Q learner can know if it has lost before a move is made"""

    Q = dict()
    learningRate = 0.7
    gamma = 0.4
    epsilon = 0.1
    
    nits = 0.0
    Qwins = 0.0
    draws = 0.0
    Rwins = 0.0
    printBoard = False
    maxiterations = 10000
    
    
    while nits < maxiterations:
        "Reset the game board"
        board = gameBoard()        
        lastboard_hash = None
        runningTotal = 0
        "While there are moves to be made"
        while not(board.full() or board.won(1) or board.won(2)):
            
            "X player makes his random move"
            randomMove(board)
            
            "O player makes his move using Q learning"
            lastboard_hash = next(board, lastboard_hash, Q, learningRate, epsilon, gamma, runningTotal)
        
        if printBoard:    
            print board    
        nits += 1.0
        
        if board.won(2): Qwins += 1.0
        if board.won(1): Rwins += 1.0 
        if board.full(): draws += 1.0
            
            
    # print Q
    print "Number of Games: ", maxiterations
    print "Q learning wins: ", Qwins/maxiterations
    print "Random wins    : ", Rwins/maxiterations
    print "Draws          : ", draws/maxiterations
   
TicTacToe()
