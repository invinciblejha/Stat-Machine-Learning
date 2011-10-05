
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

p = 10
data = loadtxt('vowel_train.txt')
data = data.reshape(-1,p+1)
data2 = loadtxt('vowel_test.txt')
data2 = data2.reshape(-1,p+1)

## reformat data as X and Y
y = data[:,0]
N = len(y)
y = transpose(matrix(y))
X = data[:,1:p+1]
X = matrix(X)
K = 11 # number of classes
y2 = data2[:,0]
N2 = len(y2)
y2 = transpose(matrix(y2))
X2 = data2[:,1:p+1]
X2 = matrix(X2)

# fit to training data
nk = zeros((K,1))
for k in range(K):
    nk[k] = sum(y == k+1)
pi = (0. + nk)/N
#print "pi= ",transpose(pi)

mu = zeros((K,X.shape[1]))
for k in range(K):
    for j in range(N):
        mu[k:k+1,:] = mu[k:k+1,:] + (y[j] == k+1)*X[j:j+1,:]/nk[k]
#print "mu= ",mu

Sigma = zeros((X.shape[1],X.shape[1]))
for k in range(K):
    for j in range(N):
        temp = X[j:j+1,:] - mu[k:k+1,:]
        Sigma = Sigma + int(y[j] == k+1)*transpose(temp)*temp/(N-K)
#print "Sigma= ",Sigma
invSigma = linalg.inv(Sigma)

# test on test data
# discriminant function
deltak = matrix(zeros((N2,K)))
for j in range(N2):
    for k in range(K):
        deltak[j,k] = (X2[j:j+1,:] - 0.5*mu[k:k+1,:])*invSigma*transpose(mu[k:k+1,:]) + log(pi[k])
#print "deltak= ",deltak

#classify
deltakT = transpose(deltak)
yhatT = matrix(deltakT.argmax(0) + 1)
yhat = transpose(yhatT)
errorrate = (0. + N2 - sum(y2 == yhat))/N2
print "errorrate (train-test) = ",errorrate

# fit and test on test data
nk = zeros((K,1))
for k in range(K):
    nk[k] = sum(y2 == k+1)
pi = (0. + nk)/N2
#print "pi= ",transpose(pi)

mu = zeros((K,X2.shape[1]))
for k in range(K):
    for j in range(N2):
        mu[k:k+1,:] = mu[k:k+1,:] + (y2[j] == k+1)*X2[j:j+1,:]/nk[k]
#print "mu= ",mu

Sigma = zeros((X2.shape[1],X2.shape[1]))
for k in range(K):
    for j in range(N2):
        temp = X2[j:j+1,:] - mu[k:k+1,:]
        Sigma = Sigma + int(y2[j] == k+1)*transpose(temp)*temp/(N2-K)
#print "Sigma= ",Sigma

# discriminant function
invSigma = linalg.inv(Sigma)
deltak = zeros((N2,K))
for j in range(N2):
    for k in range(K):
        deltak[j,k] = (X2[j:j+1,:] - 0.5*mu[k:k+1,:])*invSigma*transpose(mu[k:k+1,:]) + log(pi[k])
#print "deltak= ",deltak

#classify
deltakT = transpose(deltak)
yhatT = matrix(deltakT.argmax(0) + 1)
yhat = transpose(yhatT)
errorrate = (0. + N2 - sum(y2 == yhat))/N2
print "errorrate (test-test) = ",errorrate