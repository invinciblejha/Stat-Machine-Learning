import numpy
from numpy import diag

"""Though I would love to claim this code as my own, I found it on
   http://sites.google.com/a/innohead.com/wiki/Home/Python/reinforcement-learning---tic-tac-toe
   very nicely done, this would have taken me some time to design otherwise""" 

class gameBoard(numpy.ndarray):   
    
    symbols = {0: "_", 1: "X", 2: "O"}
    
    def __new__(subtype):
        
        "Class constructor for the game board"
        
        arr = numpy.zeros((3,3), dtype=numpy.int8)
        arr = arr.view(subtype)
        return arr
    
    def __hash__(s):
        
        "Method for returning the unique hash code of any board combination"
        
        flat = s.ravel()
        code = 0
        for i in xrange(9): code += pow(3,i) * flat[i]
        return code
    
    def won(s, player):
        
        "Returns a boolean determining if the player has won"
        
        x = s == player
        return numpy.hstack( (x.all(0), x.all(1), diag(x).all(), diag(x[:,::-1]).all()) ).any()

    def full(s):
        
        "Returns a boolean determining if the game is a draw"
        
        return (s != 0).all()
        
        
    def __str__(s):
        
        "Allows the game board to be printed to console in a human readable way"
        
        out = [""]
        for i in xrange(3):
            for j in xrange(3):
                out.append( s.symbols[ s[i,j] ] )
            out.append("\n")
        return str(" ").join(out)