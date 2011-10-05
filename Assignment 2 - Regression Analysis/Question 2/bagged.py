from pylab import *
from numpy import *

def linregp(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
    betahat = linalg.inv(transpose(X1)*X1)*transpose(X1)*y
    yhat = X1*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat,yhat,RSS

def linregp_ni(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    betahat = linalg.inv(transpose(X)*X)*transpose(X)*y
    yhat = X*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat,yhat,RSS

# training data
data = loadtxt('prostate_train.txt')
p = 8

## reformat data as X and Y
data = data.reshape(-1,p+1)
y = data[:,p]
N = len(y)
X = data[:,0:p]
y = transpose(matrix(y))
X = matrix(X)

# standardize data to unit variance
covX = cov(transpose(X)) # need to transpose X to get a p x p covariance matrix
sdX = sqrt(diag(covX)) # Note that sdX is an array
for i in range(p):
  X[:,i] = X[:,i]/sdX[i]

X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)   

## Bootstrap
print 'Bootstrap'
B = 100  # number of bootstrap samples
baggedbeta = matrix(zeros((p+1,B)))
for b in range(B):
    ytemp = matrix(zeros(shape(y)))  ## bootstrap samples
    Xtemp = matrix(zeros(shape(X)))
    for n in range(N):
        temp = random.randint(0,N,1)
        ytemp[n] = y[temp]   ## add selected to bootstrap sample
        Xtemp[n,:] = X[temp,:]
    bootbeta = linregp(Xtemp,ytemp)[0] ## fit to bootstrap samples
    baggedbeta[:,b] = bootbeta

baggedbetahat = mean(baggedbeta, axis = 1) # take mean for bagged estimate
betahat = linregp(X,y)[0]  # usual estimate
print 'betahat unbagged | bagged'
print concatenate((betahat,baggedbetahat),axis=1)