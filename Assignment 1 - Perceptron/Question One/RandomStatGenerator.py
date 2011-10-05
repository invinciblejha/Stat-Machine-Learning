from pylab import *
from numpy import *
from scipy import stats

    
"""Question One, this generates a random sample from a model of
normal distribution, then finds the slope of the data and 
tests for significance, generates the p-value and adds it to
a list of p-values. All the above is repeated multiple times."""
    

def randomStatsGenerator(startRange, endRange, sampleSize):   
         
    """Normal testing Set, generates the uniform sample """
    
    a = random.uniform(startRange, endRange, sampleSize)
    b = random.uniform(0.0,2.0,sampleSize)   
    a = transpose(matrix(a)) 
    b = transpose(matrix(b)) 
    return a,b


def linreg1(x,y):
    
    """ takes Nx1 matrices x and y, and performs linear regression y = beta x"""
    
    N = len(x) 
    x1 = concatenate((ones((shape(x)[0],1)),x),axis=1)
    V = linalg.inv(transpose(x1)*x1)
    betahat = V*transpose(x1)*y
    yhat = x1*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    s2 = RSS/(N-2)
    tstat = linalg.inv(diag(sqrt(diag(V))))*betahat/sqrt(s2) 
    return betahat,yhat,RSS,V,N,s2,tstat

def plotDataWithRegression(x,y,betahat,a):
    
    """Plots the data with the regression line"""
    
    figure(a)
    plot(x,y,'.')
    minx = float(min(x))
    maxx = float(max(x))
    miny = float(betahat[0,0] + betahat[1,0]*minx)
    maxy = float(betahat[0,0] + betahat[1,0]*maxx)
    xline = array([minx,maxx])  # note how to 
    yline = array([miny,maxy])  # input a vector    
    figure(a)  # adds line to previous plot
    plot(xline,yline,'-')

def printStats(linearRegressionData):
    
    """Method for printing the stats of the linearRegression data"""
    
    print 'N =',linearRegressionData[4]
    print 'betahat =',linearRegressionData[0]  
    print 'RSS =',linearRegressionData[2]
    print 's2 =',linearRegressionData[5]
    print 'tstat =',linearRegressionData[6]

def linregressStats(x,y):
    
    """Method for printing the linearRegression data for the stats method
    contains P-value"""    
    
    gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    print "Gradient and intercept", gradient, intercept
    print "R-squared", r_value**2
    print "p-value", p_value
    print "std-err", std_err


#Create the arrays for holding the sum of data    
pValues = []
tStats = []

def generationLoop(numberOfLoops):
    
    """The loop that does all the work, generates the data and tests the
    significance multiple times"""
    
    #Change these settings to test different variations
    startRange = 10000.0
    endRange = 100000.0
    sampleSize = 50
    
    for x in (range(numberOfLoops)):        
        rawRandomData = randomStatsGenerator(startRange, endRange, sampleSize)
        x = rawRandomData[0] 
        y = rawRandomData[1] 
               
        #Use for Adding an outlier for examination.
        #rawRandomData2 = randomStatsGenerator(0.0, 1.0, 1)
        #a = rawRandomData2[0] 
        #b = rawRandomData2[1]  
        #x = concatenate((x,a))
        #y = concatenate((y,b)) 
        
        betahat, yhat, RSS, V, N, s2, tstat = linreg1(x,y)
        gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)        
        pValues.append(p_value)
        #tStats.append(tstat[1]) 
        
        #Printing Information Methods:
        #print tstat
        #rawLinearRegression = linreg1(x,y)
        #linregressStats(x,y)  
        #printStats(rawLinearRegression)    
        #plotDataWithRegression(x,y,rawLinearRegression[0],2)        
    

"""The various method calls to get the data"""

generationLoop(1000)

hist(array(pValues))
xlabel('P-Values of random data')
ylabel('Frequency')
show()


#Generates a tstats histogram.
#hist(array(tStats))
#xlabel('T-Stats of random data')
#ylabel('Frequency')
#show()



    
