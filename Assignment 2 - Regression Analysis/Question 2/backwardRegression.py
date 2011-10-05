### Backward Stepwise Regression ###

from numpy import *

def linregp_ni(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    betahat = linalg.inv(transpose(X)*X)*transpose(X)*y
    yhat = X*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat,yhat,RSS

## get data
data = loadtxt('prostate_train.txt')
p = 8  # number of predictors
data = data.reshape(-1,p+1)
## reformat data as X and Y
y = transpose(matrix(data[:,8]))
X = matrix(data[:,0:8])
## standardize predictors to variance 1
covX = cov(transpose(X))
sdX = sqrt(diag(covX))
for i in range(8):
  X[:,i] = X[:,i]/sdX[i]

## get alpha = 0.9 values of F distribution
F90 = loadtxt('F90.txt')
F90 = F90.reshape(-1,10)

X1 = concatenate((ones(shape(y)),X),axis=1)  # add column of ones

# start with full model
oldX1 = X1  
oldbeta = linregp_ni(oldX1,y)[0]  
oldRSS = linregp_ni(oldX1,y)[2]
inclpred = list(arange(p+1))

continu = 1
while continu:
    worstRSS = 100*oldRSS  # some large number  
    for i in range(len(inclpred)):  #try deleting each predictor from current model
        X1 = concatenate((oldX1[:,0:i],oldX1[:,i+1:oldX1.shape[1]]),axis=1) #delete ith column of X 
        beta = linregp_ni(X1,y)[0]  #fit linear regression model
        RSS = linregp_ni(X1,y)[2]
        #print i,RSS
        if RSS < worstRSS:   # worst additional predictor so far
            worstRSS = RSS   # record the details
            worstpred = i  # note that this is the index in inclpred, not the predictor itself
            worstbeta = beta
    #find the F-ratio for the worst additional predictor
    F = (worstRSS - oldRSS)/(oldRSS/(len(y) - len(worstbeta) - 2)) 
    print 'worstpred=',worstpred,' worstRSS=',float(worstRSS),' F=',float(F),' critF=',F90[len(y) - len(worstbeta) -2,1]
    if F < F90[len(y) - len(worstbeta) -2,1]:  # not significant?
        ## update model
        oldX1 = concatenate((oldX1[:,0:worstpred],oldX1[:,worstpred+1:oldX1.shape[1]]),axis=1) 
        oldbeta = worstbeta
        oldRSS = worstRSS
        inclpred.pop(worstpred)
    else:  # finished
        continu = 0 
    if len(inclpred) == 0:  # stop if all predictors deleted
            continu = 0
            
print 'Final model =',inclpred

