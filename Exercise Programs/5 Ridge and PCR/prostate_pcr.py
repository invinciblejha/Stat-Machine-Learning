### SVD Ridge Regression ###

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

def ridgereg(X,y,lam):
    # takes N x p matrix X (no intercept column) and a N x 1 vector y, and fits  
    # ridge regression y = X*beta with lam*transpose(beta)*beta penalty term
    betahat0 = matrix(mean(array(y), axis=0))
    betahat = linalg.inv(transpose(X)*X + lam*eye(8))*transpose(X)*y
    #beta = dot(dot(linalg.inv(dot(transpose(X),X)+lam*eye(8)),transpose(X)),y)
    betahat = concatenate((betahat0,betahat),axis=0) 
    X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
    yhat = X1*betahat
    RSS = transpose(y - yhat)*(y - yhat) + lam*transpose(betahat)*betahat
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

## SVD
(U,D,V) = linalg.svd(Xc, full_matrices=0)
#print 'D = ',D
diagD = diag(D)
#Xc_svd = U*diagD*transpose(V)
#print 'V = ',V

Z = Xc*V
m = 4  # 4 PCs
Z4 = Z[:,0:m]  # use first m PCs
Z41 = concatenate((ones((shape(Z4)[0],1)),Z4),axis=1) # add intercept term
thetahat = linregp_ni(Z41,y)[0]
print 'thetahat = ',thetahat
V4 = V[:,0:m]
betahat_pcr = concatenate((thetahat[0],V4*thetahat[1:m+1,0]),axis=0)
print 'betahat_pcr = ',betahat_pcr


  

 




















