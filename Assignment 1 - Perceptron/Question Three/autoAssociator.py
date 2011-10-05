from pylab import *
from numpy import *
import mlp


#Load the data into arrays
numbers = loadtxt('numbers.data',delimiter=' ')
noiseNumbers1 = loadtxt('num_noise1.data',delimiter=' ')
noiseNumbers2 = loadtxt('num_noise2.data',delimiter=' ')

#Shape the data into something useful
inputs = numbers
inputs = transpose(inputs)
noiseinputs1 = transpose(noiseNumbers1)
noiseinputs2 = transpose(noiseNumbers2)


def evaluateLearning():
    
    """Main runner method, can be modified to loop through different 
    hidden layers, note I have changed to MLPtrain method to return
    the x and y values of the errors. (Iteration and error amount) """
    
    figure(1)
    
    #Change this to use the mlp with different layers
    #for x in [1,2,3,5,10,15]:
    #for x in [13,14,15,16,17]:        
    for x in [15]:      
        print x
        net = mlp.mlp(inputs,inputs,x,outtype='logistic')
        y, x1 = net.mlptrain(inputs,inputs,0.25,20000)
        net.confmat(inputs,inputs)
        net.confmat(noiseinputs1,inputs)
        net.confmat(noiseinputs2,inputs)
        
        plot(x1,y,'.')
    xlabel('Number of Learning Iterations')
    ylabel('Learning Error (Log 10)')   
    show()   

#Main call to the main method        
evaluateLearning()    
