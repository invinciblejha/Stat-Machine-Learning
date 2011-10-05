from numpy import *
from pylab import *

def linregp(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
    betahat = linalg.inv(transpose(X1)*X1)*transpose(X1)*y
    yhat = X1*betahat
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
print 'sdX = ',sdX
corrX = covX/dot(transpose(mat(sdX)),mat(sdX))
print 'corrX = ',corrX

#standardize data
for i in range(p):
  X[:,i] = X[:,i]/sdX[i]

# do linear regression
temp = linregp(X,y)
betahat = temp[0]
yhat = temp[1]

# estimate covariance matrix
sigmahat2 = float((1.0/(len(y) - len(betahat)))*transpose(yhat - y)*(yhat - y))
print 'sigmahat2 = ',sigmahat2
X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
varbeta = linalg.inv(transpose(X1)*X1)*sigmahat2

# calculate Z-statistics
z = multiply(betahat,1/transpose(matrix(diag(sqrt(varbeta)))))
# and print
print 'beta, z = ',concatenate((betahat,z),axis=1)
