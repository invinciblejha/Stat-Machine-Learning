from pylab import *
from numpy import *

def nbayesclass(X,y):
    # naive Bayes classifier for y = {0,1}, X a matrix of p predictors
    p = X.shape[1]
    f0k = zeros(shape(X),dtype=float) # density estimate for y = 0, kth predictor
    f1k = zeros(shape(X),dtype=float) # density estimate for y = 1, kth predictor
    for k in range(p):
        x = X[:,k:k+1] # get kth predictor
        x1 = x[y>0.5]  # and divide according 
        x0 = x[y<0.5]  # value of y
        lam = 0.05*(max(x) - min(x)) # bandwidth
        lsq = lam**2
        # calculate density estimates
        for i in range(len(y)):
            K1 = (1/sqrt(2*pi))*exp(-((x1-x[i])**2)/(2*lsq))
            K0 = (1/sqrt(2*pi))*exp(-((x0-x[i])**2)/(2*lsq))
            f1k[i,k] = sum(K1)/(lam*len(x1))
            f0k[i,k] = sum(K0)/(lam*len(x0))
    # multiply over predictors
    f1 = transpose(matrix(f1k[:,0])) 
    f0 = transpose(matrix(f0k[:,0]))
    for k in range(p-1):
        f1 = multiply(f1,f1k[:,k+1:k+2])
        f0 = multiply(f0,f0k[:,k+1:k+2])
    return (f1 > f0) # classification, 0 or 1

data = loadtxt('saheartdis.txt')
data = data.reshape(-1,10)
## reformat data as x and y
y = data[:,9:10] ## chd
X = data[:,0:9] 

nbc = nbayesclass(X,y)
errorrate = (0. + len(y) - sum(y == nbc))/len(y)
print 'errorrate =',errorrate