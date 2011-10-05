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

def nbayesclass(X,y):
    # naive Bayes classifier for y = {0,1}, X a matrix of p predictors
    p = X.shape[1]
    f0k = zeros(shape(X),dtype=float) # density estimate for y = 0, kth predictor
    f1k = zeros(shape(X),dtype=float) # density estimate for y = 1, kth predictor
    for k in range(p):
        x = X[:,k:k+1] # get kth predictor
        x1 = x[y>0.5]  # and divide according 
        x0 = x[y<0.5]  # value of y
        lam = 0.05*(max(x) - min(x)) # bandwidth
        lsq = lam**2
        # calculate density estimates
        for i in range(len(y)):
            K1 = (1/sqrt(2*pi))*exp(-((x1-x[i])**2)/(2*lsq))
            K0 = (1/sqrt(2*pi))*exp(-((x0-x[i])**2)/(2*lsq))
            f1k[i,k] = sum(K1)/(lam*len(x1))
            f0k[i,k] = sum(K0)/(lam*len(x0))
    # multiply over predictors
    f1 = transpose(matrix(f1k[:,0])) 
    f0 = transpose(matrix(f0k[:,0]))
    for k in range(p-1):
        f1 = multiply(f1,f1k[:,k+1:k+2])
        f0 = multiply(f0,f0k[:,k+1:k+2])
    return (f1 > f0) # classification, 0 or 1

data = loadtxt('iris.txt',delimiter=',')
data = data.reshape(-1,5)

## reformat data as x and y
y = data[:,4:5] ## chd
X = data[:,0:4] 

nbc = nbayesclass(X,y)
errorrate = (0. + len(y) - sum(y == nbc))/len(y)
print 'Naive Bayes Error Rate =',errorrate

trainerror = 0
testerror  = 0
for x in range(100):
    order = range(shape(data)[0])
    random.shuffle(order)
    data = data[order,:]
    
    ytrain = data[0:100,4:5]
    Xtrain = data[0:100,0:4]
    ytest  = data[100:150,4:5]
    Xtest  = data[100:150,0:4]
    
    Xtrain = matrix(Xtrain)
    Xtest  = matrix(Xtest)
    
    K = 3
    Ntrain = len(ytrain)
    Ntest  = len(ytest)
    
    # fit to training data
    nk = zeros((K,1))
    for k in range(K):
        nk[k] = sum(ytrain == k+1)
    pi = (0. + nk)/Ntrain
    
    mu = zeros((K,Xtrain.shape[1]))
    for k in range(K):
        for j in range(Ntrain):
            mu[k:k+1,:] = mu[k:k+1,:] + (ytrain[j] == k+1)*Xtrain[j:j+1,:]/nk[k]
            
    Sigma = zeros((Xtrain.shape[1],Xtrain.shape[1]))
    for k in range(K):
        for j in range(Ntrain):
            temp = Xtrain[j:j+1,:] - mu[k:k+1,:]
            Sigma = Sigma + int(ytrain[j] == k+1)*transpose(temp)*temp/(Ntrain-K)
    #print "Sigma= ",Sigma
    invSigma = linalg.inv(Sigma)
    
    # discriminant function
    deltak = matrix(zeros((Ntest,K)))
    for j in range(Ntest):
        for k in range(K):
            deltak[j,k] = (Xtest[j:j+1,:] - 0.5*mu[k:k+1,:])*invSigma*transpose(mu[k:k+1,:]) + log(pi[k])
    #print "deltak= ",deltak
    
    #classify
    deltakT = transpose(deltak)
    yhatT = matrix(deltakT.argmax(0) + 1)
    yhat = transpose(yhatT)
    errorrate = (0. + Ntest - sum(ytest == yhat))/Ntest
    #print "errorrate (train-test) = ",errorrate
    trainerror += errorrate
    
    # fit and test on test data
    nk = zeros((K,1))
    for k in range(K):
        nk[k] = sum(ytest == k+1)
    pi = (0. + nk)/Ntest
    #print "pi= ",transpose(pi)
    
    mu = zeros((K,Xtest.shape[1]))
    for k in range(K):
        for j in range(Ntest):
            mu[k:k+1,:] = mu[k:k+1,:] + (ytest[j] == k+1)*Xtest[j:j+1,:]/nk[k]
    #print "mu= ",mu
    
    Sigma = zeros((Xtest.shape[1],Xtest.shape[1]))
    for k in range(K):
        for j in range(Ntest):
            temp = Xtest[j:j+1,:] - mu[k:k+1,:]
            Sigma = Sigma + int(ytest[j] == k+1)*transpose(temp)*temp/(Ntest-K)
    #print "Sigma= ",Sigma
    
    # discriminant function
    invSigma = linalg.inv(Sigma)
    deltak = zeros((Ntest,K))
    for j in range(Ntest):
        for k in range(K):
            deltak[j,k] = (Xtest[j:j+1,:] - 0.5*mu[k:k+1,:])*invSigma*transpose(mu[k:k+1,:]) + log(pi[k])
    #print "deltak= ",deltak
    
    #classify
    deltakT = transpose(deltak)
    yhatT = matrix(deltakT.argmax(0) + 1)
    yhat = transpose(yhatT)
    errorrate = (0. + Ntest - sum(ytest == yhat))/Ntest
    #print "errorrate (test-test) = ",errorrate
    testerror += errorrate
    
print "Linear Discr. Avg. Error Rate (train-test) = ",trainerror/100
print "Linear Discr. Avg. Error Rate (test-test)  = ",testerror/100