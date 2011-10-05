
from numpy import *
from pylab import *

data = loadtxt('saheartdis.txt')
P = 9
data = data.reshape(-1,P+1)

## reformat data as X and Y
y = data[:,9:10]
N = len(y)
K = 2
X = data[:,0:9]
X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
X1 = matrix(X1)

# initialize
betaold = zeros((P+1,1))
tol = 0.000001
stepdiff = 1
while stepdiff > tol:
    p = exp(X1*betaold)
    p = p/(1+p)
    temp = multiply(p,(ones(shape(p)) - p))
    W = zeros((N,N))
    for j in range(N):
        W[j,j] = temp[j,0]
    beta = betaold + (linalg.inv(transpose(X1)*W*X1))*transpose(X1)*(matrix(y) - p)
    stepdiff = abs(betaold - beta).max()
    betaold = beta
    #print stepdiff
    
print 'beta= ',beta
p = exp(X1*beta)
p = p/(1+p)
print 'y,p(Y=1)= ',concatenate((y,p),axis = 1)

#classify
yhat = (p >= 0.5)
errorrate = (0. + N - sum(y == yhat))/N
print "errorrate= ",errorrate

# check z-scores 
temp = multiply(p,(ones(shape(p)) - p))
W = zeros((N,N))
for j in range(N):
    W[j,j] = temp[j,0]
varbeta = linalg.inv(transpose(X1)*W*X1)
z = multiply(beta,1/transpose(matrix(sqrt(diag(varbeta)))))
print 'beta, z = ',concatenate((beta,z),axis=1)


        