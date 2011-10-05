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

print 'factor, beta, z, errorrate= '
for factor in range(P):
    X1 = concatenate((ones((shape(X)[0],1)),X[:,factor:factor+1]),axis=1)
    X1 = matrix(X1) 
    #print X1
    # initialize
    betaold = zeros((2,1))
    tol = 0.000001
    stepdiff = 1
    while stepdiff > tol:
        p = exp(X1*betaold)
        p = p/(1+p)
        temp = multiply(p,(ones(shape(p)) - p))
        W = zeros((N,N))
        for j in range(N):
            W[j,j] = temp[j,0]
        temp = transpose(X1)*W*X1 + matrix([[tol,0],[0,0]]) # to avoid singular matrix
        beta = betaold + (linalg.inv(temp))*transpose(X1)*(matrix(y) - p)
        stepdiff = abs(betaold - beta).max()
        betaold = beta     
    p = exp(X1*beta)
    p = p/(1+p)
    yhat = (p >= 0.5)
    errorrate = (0. + N - sum(y == yhat))/N
    # check z-score
    temp = multiply(p,(ones(shape(p)) - p))
    W = zeros((N,N))
    for j in range(N):
        W[j,j] = temp[j,0]
        varbeta = linalg.inv(transpose(X1)*W*X1 + matrix([[tol,0],[0,0]]))
        z = multiply(beta,1/transpose(matrix(sqrt(diag(varbeta)))))
    print factor+1,float(beta[1]),float(z[1]),errorrate



        