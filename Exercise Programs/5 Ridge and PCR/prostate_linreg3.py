from numpy import *
from pylab import *

def linregp_ni(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    betahat = linalg.inv(transpose(X)*X)*transpose(X)*y
    yhat = X*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat,yhat,RSS

data = loadtxt('prostate_train.txt')
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

# standardize data
for i in range(p):
  X[:,i] = X[:,i]/sdX[i]

# specify 2 models
X1 = concatenate((ones((shape(X)[0],1)),X),axis=1) # augment with intercept term
X0 = concatenate((X[:,0:2],X[:,3:5]),axis=1) # take only 1st, 2nd, 4th and 5th predictors

# do regression(s)
temp1 = linregp_ni(X1,y)   
temp0 = linregp_ni(X0,y) 

betahat1 = temp1[0]
RSS1 = temp1[2]
betahat0 = temp0[0]
yhat0 = temp0[1]
RSS0 = temp0[2]

# check z-scores for smaller model
sigmahat2 = float((1.0/(len(y) - len(betahat0)))*transpose(yhat0 - y)*(yhat0 - y))
varbeta = linalg.inv(transpose(X0)*X0)*sigmahat2
# calculate Z-statistics
z = multiply(betahat0,1/transpose(matrix(sqrt(diag(varbeta)))))
# and print
print 'beta, z = ',concatenate((betahat0,z),axis=1)

F = float(((RSS0 - RSS1)/(len(betahat1) - len(betahat0)))/(RSS1/(len(y) - len(betahat1) - 2))) 
print 'F = ',F