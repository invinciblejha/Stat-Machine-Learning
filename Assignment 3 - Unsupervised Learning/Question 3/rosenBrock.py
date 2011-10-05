from numpy import *

def rosenBrock(population):    
    
    fitness = [rosenBrookFunction(x[0],x[1]) for x in population]    
        
    return array(fitness)

def rosenBrookFunction(x1, y1):
    
    f = 100 * math.pow((y1 - math.pow(x1,2)),2) + math.pow((1 - x1), 2)
    
    return math.fabs(f)

