from pylab import *
from numpy import *
import pcn


def getDataSperated(dataSet): 
    
    """Method for seperating the data, x from y"""
    
    data = dataSet + '.txt'    
    a = loadtxt(data,delimiter=',')
    x = a[:,0] 
    y = a[:,1] 
    x = transpose(matrix(x)) 
    y = transpose(matrix(y))      
    return x,y  


def getData(dataSet): 
    
    """Method for gathering data from a file"""
    
    data = dataSet + '.txt'    
    a = loadtxt(data,delimiter=',')    
    return a


def showAllDataPlot():
    
    """This method plots the entire set on a scatterplot"""
         
    x, y = getDataSperated('a')    
    x1, y1 = getDataSperated('b')    
    x2, y2 = getDataSperated('c')    
    x3, y3 = getDataSperated('d')    
    ion()
    plot(1,1,'go')
    plot(x1,y1,'rx')
    plot(x2,y2,'mD')
    plot(x3,y3,'b^')
    show()
    
    
def showTwoDataPlot():  
    
    """This method plots the data for set a and b"""
       
    x, y = getDataSperated('a')    
    x1, y1 = getDataSperated('b')    
    ion()
    plot(x,y,'go')
    plot(x1,y1,'rx')    
    show()
    
    
def plotClassificationLine(weights, inputs):
    
    """Plots the linear classification line"""
    
    minx = float(min(inputs[0,:]))
    maxx = float(max(inputs[0,:]))    

    miny = float(-weights[2]-weights[0]*minx)/weights[1]
    maxy = float(-weights[2]-weights[0]*maxx)/weights[1]

    yline = array([miny[0],maxy[0]])  
    xline = array([minx,maxx]) 
    plot(xline,yline,'-')


def doubleSet_Perceptron(setX,setY,iterations): 
    
    """This method is used for using the perceptron on two sets"""
    
    #Create the data
    a = getData(setX)
    b = getData(setY)
    c = ones((50,1))
    d = zeros((50,1))
    targets = concatenate((c,d))
    inputs = concatenate((a,b)) 
    
    # Randomise order of inputs
    change = range(shape(inputs)[0])
    random.shuffle(change)
    inputs = inputs[change,:]
    targets = targets[change,:]
     
    #Do the training, find the weights, plot the classification
    #and create the confusion matrix   
    p = pcn.pcn(inputs,targets)
    weights = p.pcntrain(inputs,targets,0.25,iterations)
    plotClassificationLine(weights,inputs)
    p.confmat(inputs,targets)
    
    
def fullSetPerceptron(ain,bin,cin,din,iterations):
    
    """This method is used for using the perceptron on the entire set
    permutate the methods arguments to change the separated set"""
    
    #Gather the data
    a = getData(ain)
    b = getData(bin)
    c = getData(cin)
    d = getData(din)
    x = ones((50,1))
    y = zeros((150,1))
    targets = concatenate((x,y))
    inputs = concatenate((a,b,c,d)) 
    
    # Randomise order of inputs
    change = range(shape(inputs)[0])
    random.shuffle(change)
    inputs = inputs[change,:]
    targets = targets[change,:]
    
    #Do the training, find the weights, plot the classification
    #and create the confusion matrix  
    p = pcn.pcn(inputs,targets)
    weights = p.pcntrain(inputs,targets,0.25,iterations)
    #plotClassificationLine(weights,inputs)
    p.confmat(inputs,targets)
    
def testSetsInPairs(iterations):
    
    """Method for calling all pairs to be classified from one another"""
    
    print "set A and B:"    
    doubleSet_Perceptron('a','b',iterations)
    print "set A and C:"    
    doubleSet_Perceptron('a','c',iterations) 
    print "set A and D:"    
    doubleSet_Perceptron('a','d',iterations) 
    print "set B and C:"    
    doubleSet_Perceptron('b','c',iterations) 
    print "set B and D:"    
    doubleSet_Perceptron('b','d',iterations)
    print "set C and D"
    doubleSet_Perceptron('c','d',iterations) 
    
    
def testFullSet(iterations):
    
    """Method for calling all sets to be separated from the entire set."""
    
    print "Set A and the Rest"
    fullSetPerceptron('a','b','c','d',iterations)
    print "Set B and the Rest"
    fullSetPerceptron('b','c','d','a',iterations)
    print "Set C and the Rest"
    fullSetPerceptron('c','d','a','b',iterations)
    print "Set D and the Rest"
    fullSetPerceptron('d','a','b','c',iterations)



"""Various calls to perform the linear classification with perceptrons"""

ion()    
doubleSet_Perceptron('a','b',100)
showTwoDataPlot() 
 
#testSetsInPairs(1000)
#testFullSet(1000)
#showAllDataPlot()   

