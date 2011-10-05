from numpy import *
from pylab import *

def lda(X1,y1,X2,y2,K):
    # performs LDA using training data (X1,y1) and outputs error rate on test data (X2,y2)
    N1 = len(y1)
    N2 = len(y2)
    nk = zeros((K,1))
    for k in range(K):
        nk[k] = sum(y1 == k)
    pi = (0. + nk)/N1
    #print "pi= ",transpose(pi)
    mu = zeros((K,X1.shape[1]))
    for k in range(K):
        for j in range(N1):
            mu[k:k+1,:] = mu[k:k+1,:] + (y1[j] == k)*X1[j:j+1,:]/nk[k]
    #print "mu= ",mu
    Sigma = zeros((X1.shape[1],X1.shape[1]))
    for k in range(K):
        for j in range(N1):
            temp = X1[j:j+1,:] - mu[k:k+1,:]
            Sigma = Sigma + (int(y1[j]) == k)*transpose(temp)*temp/(N1-K)
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
    y2hatT = matrix(deltakT.argmax(0))
    y2hat = transpose(y2hatT)
    errorrate = (0. + N2 - sum(y2 == y2hat))/N2
    #print "errorrate= ",errorrate
    return errorrate,pi,mu,Sigma

def nbayesclass(X1,y1,X2,y2):
    # naive Bayes classifier for y = {0,1,2}, X a matrix of p predictors
    N1 = len(y1)
    N2 = len(y2)
    p = X1.shape[1]
    f0k = zeros(shape(X1),dtype=float) # density estimate for y = 0, kth predictor
    f1k = zeros(shape(X1),dtype=float) # density estimate for y = 1, kth predictor
    f2k = zeros(shape(X1),dtype=float) # density estimate for y = 2, kth predictor
    for k in range(p):
        x = X1[:,k:k+1] # get kth predictor
        x0 = x[y1 == 0] # and divide according
        x1 = x[y1 == 1] # value of y
        x2 = x[y1 == 2]
        xk = X2[:,k:k+1] # test values, kth predictor
        lam = 0.05*(max(x) - min(x)) # bandwidth
        lsq = float(lam**2)
        # calculate density estimates
        for i in range(N2):
            xki = float(xk[i])
            #print pi,lsq,x0.shape,xk.shape
            K0 = (1/sqrt(2*pi))*exp(-(multiply(x0-xki,x0-xki))/(2*lsq))
            K1 = (1/sqrt(2*pi))*exp(-(multiply(x1-xki,x1-xki))/(2*lsq))
            K2 = (1/sqrt(2*pi))*exp(-(multiply(x2-xki,x2-xki))/(2*lsq))
            f0k[i,k] = sum(K0)/(lam*len(x0))
            f1k[i,k] = sum(K1)/(lam*len(x1))
            f2k[i,k] = sum(K2)/(lam*len(x2))
    # multiply over predictors
    f0 = transpose(matrix(f0k[:,0]))
    f1 = transpose(matrix(f1k[:,0]))
    f2 = transpose(matrix(f2k[:,0]))
    for k in range(p-1):
        f0 = multiply(f0,f0k[:,k+1:k+2])
        f1 = multiply(f1,f1k[:,k+1:k+2])
        f2 = multiply(f2,f2k[:,k+1:k+2])
    nbc = zeros(shape(y2))
    for i in range(N2):
        if f0[i] > f1[i]:
            if f0[i] > f2[i]:
                nbc[i] = 0
            else:
                nbc[i] = 2
        else:
            if f1[i] > f2[i]:
                nbc[i] = 1
            else:
                nbc[i] = 2
    errorrate = (0. + N2 - sum(y2 == nbc))/N2
    return errorrate

# get iris data
data = loadtxt('iris.txt',delimiter=',')
p = 4
data = data.reshape(-1,p+1)

## reformat data as X and Y
y = data[:,p]
N = len(y)
y = transpose(matrix(y))
X = data[:,0:p]
X = matrix(X)
K = 3 # number of classes

M = 100 # number of random training/test assignments
ER = zeros((M,2)) # Storage for error rates
for m in range(M):
    # randomly divide data into training and test data
    setsize = int(N*2/3)
    setno0 = random_sample((N,))
    setno1 = sort(setno0)
    boundary = setno1[setsize]
    X1 = X[setno0 < boundary,:]
    y1 = y[setno0 < boundary]
    X2 = X[setno0 >= boundary,:]
    y2 = y[setno0 >= boundary]
    ldaer = lda(X1,y1,X2,y2,K)[0]
    nbcer = nbayesclass(X1,y1,X2,y2)
    ER[m,0] = ldaer
    ER[m,1] = nbcer
    print 'LDA | Naive Bayes error rate =',ldaer,nbcer
print ' '
print 'Mean Error Rate [LDA | NBC]=',mean(ER, axis = 0)

