from pylab import *
from numpy import *

def linregp(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
    betahat = linalg.inv(transpose(X1)*X1)*transpose(X1)*y
    yhat = X1*betahat
    RSS = float(transpose(y - yhat)*(y - yhat))
    return betahat,yhat,RSS

def linregp_ni(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    betahat = linalg.inv(transpose(X)*X)*transpose(X)*y
    yhat = X*betahat
    RSS = float(transpose(y - yhat)*(y - yhat))
    return betahat,yhat,RSS

# training data
data = loadtxt('prostate_train.txt')
p = 8

## reformat data as X and Y
data = data.reshape(-1,p+1)
y = data[:,p]
X = data[:,0:p]
y = transpose(matrix(y))
X = matrix(X)

# standardize data to unit variance
covX = cov(transpose(X)) # need to transpose X to get a p x p covariance matrix
sdX = sqrt(diag(covX)) # Note that sdX is an array
for i in range(p):
  X[:,i] = X[:,i]/sdX[i]
# reduced model
Xred = concatenate((X[:,0:2],X[:,3:5]),axis=1)

## test data
data2 = loadtxt('prostate_test.txt')
data2 = data2.reshape(-1,p+1)
## reformat data as X and Y
y2 = data2[:,p]
X2 = data2[:,0:p]
y2 = transpose(matrix(y2))
X2 = matrix(X2)
for i in range(p):
  X2[:,i] = X2[:,i]/sdX[i] # standardise according to TRAINING data variance
Xred2 = concatenate((X2[:,0:2],X2[:,3:5]),axis=1)

## Test on independent data
print 'Independent training and test data'
# full model, training
tempfull = linregp(X,y)
betafull = tempfull[0]
yhatfull = tempfull[1]
RSSfull = tempfull[2]
print 'beta (full) =',betafull
# fit full model, training, to test data
X21 = concatenate((ones((shape(X2)[0],1)),X2),axis=1)
yhatfull2 = X21*betafull
RSSfull2 = transpose(y2 - yhatfull2)*(y2 - yhatfull2)
# reduced model, training
tempred = linregp_ni(Xred,y)
betared = tempred[0]
yhatred = tempred[1]
RSSred = tempred[2]
print 'beta (reduced) =',betared
# fit reduced model, training, to test data
yhatred2 = Xred2*betared
RSSred2 = transpose(y2 - yhatred2)*(y2 - yhatred2)
print 'full model: RSS(train), RSS(test) =',RSSfull,RSSfull2
print 'reduced model: RSS(train), RSS(test) =',RSSred,RSSred2
print ' '

sigmahat2 = float((1.0/(len(y) - len(betafull)))*transpose(yhatfull - y)*(yhatfull - y))
print 'sigmahat2 = ',sigmahat2
    
## Calculate optimism
X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
print 'Optimism'
opfull = 2.0*len(betafull)*sigmahat2/len(y)
opred = 2.0*len(betared)*sigmahat2/len(y)

errfull = RSSfull/len(y)  ## Training error
errred = RSSred/len(y)
Errfull = errfull + opfull   ## In sample error
Errred = errred + opred
print 'err,op,Err (full)=',errfull,opfull,Errfull
print 'err,op,Err (reduced)=',errred,opred,Errred