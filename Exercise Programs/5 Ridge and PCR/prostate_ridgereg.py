### Ridge Regression ###

from numpy import *

def linregp_ni(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    betahat = linalg.inv(transpose(X)*X)*transpose(X)*y
    yhat = X*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat,yhat,RSS

def ridgereg(X,y,lam):
    # takes N x p matrix X (no intercept column) and a N x 1 vector y, and fits  
    # ridge regression y = X*beta with lam*transpose(beta)*beta penalty term
    betahat = linalg.inv(transpose(X)*X + lam*eye(X.shape[1]))*transpose(X)*y
    # add intercept term
    betahat0 = matrix(mean(array(y), axis=0))
    betahat = concatenate((betahat0,betahat),axis=0) 
    X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
    # calculate fitted yhat
    yhat = X1*betahat
    RSS = transpose(y - yhat)*(y - yhat)  # this is not penalized
    return betahat,yhat,RSS

## get data
data = loadtxt('prostate_train.txt')
data = data.reshape(-1,9)

## get alpha = 0.9 values of F distribution
F90 = loadtxt('F90.txt')
F90 = F90.reshape(-1,10)

#print data

## reformat data as X and Y
p = 8
data = data.reshape(-1,p+1)

## reformat data as X and Y
y = data[:,p]
X = data[:,0:p]
y = transpose(matrix(y))
X = matrix(X)

covX = cov(transpose(X)) # need to transpose X to get a p x p covariance matrix
sdX = sqrt(diag(covX)) # Note that sdX is an array
#print 'sdX = ',sdX

# standardize data to unit variance
for i in range(p):
  X[:,i] = X[:,i]/sdX[i]

## center inputs
meanX = mean(array(X), axis=0)
Xc = X - ones(shape(y))*meanX

lam = 32
temp = ridgereg(Xc,y,lam)
betahat = temp[0]
yhat = temp[1]
RSS = temp[2]
print 'betahat = ',betahat
print 'RSS = ',float(RSS)
            
            




















