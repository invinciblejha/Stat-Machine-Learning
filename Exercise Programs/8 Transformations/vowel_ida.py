from numpy import *
from pylab import *

data = loadtxt('vowel_train.txt')
p = 10
data = data.reshape(-1,p+1)

## reformat data as X and Y
y = data[:,0]
N = len(y)
y = transpose(matrix(y))
X = data[:,1:p+1]
X = matrix(X)
K = 11 # number of classes

nk = zeros((K,1))
for k in range(K):
    nk[k] = sum(y == k+1)
pi = (0. + nk)/N
print "pi= ",transpose(pi)

mu = zeros((K,X.shape[1]))
for k in range(K):
    for j in range(N):
        mu[k:k+1,:] = mu[k:k+1,:] + (y[j] == k+1)*X[j:j+1,:]/nk[k]
print "mu= ",mu

Sigma = zeros((X.shape[1],X.shape[1]))
for k in range(K):
    for j in range(N):
        temp = X[j:j+1,:] - mu[k:k+1,:]
        Sigma = Sigma + int(y[j] == k+1)*transpose(temp)*temp/(N-K)
print "Sigma= ",Sigma

# discriminant function
invSigma = linalg.inv(Sigma)
deltak = zeros((N,K))
for j in range(N):
    for k in range(K):
        deltak[j,k] = (X[j:j+1,:] - 0.5*mu[k:k+1,:])*invSigma*transpose(mu[k:k+1,:]) + log(pi[k])
print "deltak= ",deltak

#classify
deltakT = transpose(deltak)
yhatT = matrix(deltakT.argmax(0) + 1)
yhat = transpose(yhatT)
errorrate = (0. + N - sum(y == yhat))/N
print "errorrate= ",errorrate
