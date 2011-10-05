from pylab import *
from scipy.stats.mstats import mquantiles
from Regression import *

def findOutliers(data, mean, std):
    count = 0
    for x in data:
        absx = abs(x - mean)
        absx = absx/std
        if absx > (std*3):
            count += 1
    
    return count

def showAllDataInOnePlot(data):
    for i in range(p+1):
        for j in range(i):
            k = (p+1)*i+j+1
            figure(1)
            subplot(p+1,p+1,k)
            xticks([])
            yticks([])
            plot(data[:,i],data[:,j],'.')
    show()   
    #bottom row is the response variable


Reg = Regression()

#data = loadtxt('Q1.data')
data = loadtxt('Q2.data',delimiter=',')
p = 10
data = data.reshape(-1,p+1)
split = int(len(data)*.66)
traindata = data[0:split,:] 
testdata = data[split:len(data),:]


ytrain = traindata[:,p:p+1] 
Ntrain = len(ytrain) 

ytest = testdata[:,p:p+1] 
Ntest = len(ytest)

X = data[:,0:p] 
X = matrix(X)

 
#Add Squared Terms    
#X = concatenate((X,pow(data[:,0:p],2)), axis = 1) 

#Add Logged Terms  
#X = concatenate((X,log(data[:,0:p])), axis = 1)  

X1 = concatenate((ones((shape(X)[0],1)),X),axis=1) 
# divide into 2/3 training, 1/3 test
X1train = X1[0:split,:] 
X1test = X1[split:len(data),:]  

showAllDataInOnePlot(data)

for i in range(p+1):
    currentData = data[:,i]
    print "Basic Statistical Data for Column", i+1
    print "Mean        ", mean(currentData)
    print "Median      ", median(currentData)
    print "Stand. Dev. ", std(currentData)
    print "Outliers    ", findOutliers(currentData, mean(currentData), std(currentData))
    print "Quantiles   ", mquantiles(currentData)
    print "\n"

