from pylab import *
from numpy import *
from ridgeRegression import *
from principalComponents import *

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
    betahat = linalg.inv(transpose(X)*X + lam*eye(X.shape[1]))*transpose(X)*y
    # add intercept term
    betahat0 = matrix(mean(array(y), axis=0))
    betahat = concatenate((betahat0,betahat),axis=0) 
    X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
    # calculate fitted yhat
    yhat = X1*betahat
    RSS = transpose(y - yhat)*(y - yhat)  # this is not penalized
    return betahat,yhat,RSS

def principalReg(Xc, y, m):
    (U,D,V) = linalg.svd(Xc, full_matrices=0)        
    diagD = diag(D)        
    Z = Xc*V    
    Z4 = Z[:,0:m]  # use first m PCs
    Z41 = concatenate((ones((shape(Z4)[0],1)),Z4),axis=1) # add intercept term
    thetahat = linregp_ni(Z41,y)[0]        
    V4 = V[:,0:m]
    betahat_pcr = concatenate((thetahat[0],V4*thetahat[1:m+1,0]),axis=0)
    #Calculate RSS and yhat
    X1 = concatenate((ones((shape(Xc)[0],1)),Xc),axis=1)        
    yhat = X1*betahat_pcr
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat_pcr, yhat, RSS

def fitToModel(Xc,y,beta):
    X1 = concatenate((ones((shape(Xc)[0],1)),Xc),axis=1)        
    yhat = X1*beta
    RSS = transpose(y - yhat)*(y - yhat)
    return beta, yhat, RSS

def principalComponentTesting(X2, y2):
    
    for index in range(2,15):
        print 'Number of principal components:', index
        pcr = principalComponents(index)
        temppcr = pcr.getPrincipalComponents()
        betapcr = temppcr[0]
        yhatpcr = temppcr[1]
        RSSpcr = temppcr[2]
        
        print 'RSS prc = ', RSSpcr    
        plot(index,RSSpcr,'^')
        
        meanX = mean(array(X2), axis=0)
        XcTest = X2 - ones(shape(y2))*meanX 
        betapcr2, yhatpcr2, RSSpcr2 = fitToModel(XcTest, y2, betapcr)
        plot(index,RSSpcr2,'*')

    xlabel('Number of Principal Components')
    ylabel('RSS: (*) for Test RSS, (^) for Train RSS')    
    title("Principal Component Testing")
    show()
    
def ridgeRegressionTesting(X2, y2):
    
    for index in range(1,70):
        #print 'Lambda used for ridge regression:', index
        
        #betaridge, yhatridge, RSSridge = ridgereg(X,y,index)
        dflambda = trace(X2*linalg.inv(transpose(X2)*X2+index*eye(8))*transpose(X2))
        
        print 'RSS prc = ', RSSridge    
        plot(index,dflambda,'^')
        
        #meanX = mean(array(X2), axis=0)
        #XcTest = X2 - ones(shape(y2))*meanX 
        #betaridge2, yhatridge2, RSSridge2 = fitToModel(XcTest, y2, betaridge)
        #plot(index,RSSridge2,'*')

    xlabel('Lambda Value')
    ylabel('Degrees of Freedom')    
    title("Lambda Testing for RidgeRegression")
    show()



"""Create Training data and standardise according to training data variance"""

data = loadtxt('prostate_train.txt')
p = 8

## reformat data as X and Y
data = data.reshape(-1,p+1)
y = data[:,p]
N = len(y)
X = data[:,0:p]
y = transpose(matrix(y))
X = matrix(X)

# standardize data to unit variance
covX = cov(transpose(X)) # need to transpose X to get a p x p covariance matrix
sdX = sqrt(diag(covX)) # Note that sdX is an array
for i in range(p):
  X[:,i] = X[:,i]/sdX[i]
  
# reduced model
Xred = concatenate((X[:,0:2],X[:,3:5]),axis=1)

#Center the inputs
meanX = mean(array(X), axis=0)
Xcentered = X - ones(shape(y))*meanX 

"""Create Test data and standardise according to training data variance"""

data2 = loadtxt('prostate_test.txt')
data2 = data2.reshape(-1,p+1)

## reformat data as X and Y
y2 = data2[:,p]
X2 = data2[:,0:p]
y2 = transpose(matrix(y2))
X2 = matrix(X2)
for i in range(p):
  X2[:,i] = X2[:,i]/sdX[i] # standardise according to TRAINING data variance

Xred2 = concatenate((X2[:,0:2],X2[:,3:5]),axis=1)       # Reduced model

#Center the inputs

#X2 = X2 - ones(shape(y2))*meanX

#Add intercept to X2, with centered inputs
Xintercept2 = concatenate((ones((shape(X2)[0],1)),X2),axis=1)



"""Full model, fit to training, then fit full 
    training model, to test data"""
    
betafull, yhatfull, RSSfull = linregp(X,y)
yhatfull2 = Xintercept2*betafull
RSSfull2 = transpose(y2 - yhatfull2)*(y2 - yhatfull2)
print 'beta (full) =',betafull


"""Reduced model, fit to training, then fit reduced 
    training model, to test data"""
    
betared, yhatred, RSSred = linregp_ni(Xred,y)
yhatred2 = Xred2*betared
RSSred2 = transpose(y2 - yhatred2)*(y2 - yhatred2)
print 'beta (reduced) =',betared


"""Ridge Regression Model, fit to training, then fit ridge regression
    training model, to test data"""    

betaridge, yhatridge, RSSridge = ridgereg(Xcentered,y,32)
#Remove the centering bais in the ridge models beta hat
meanX = matrix(meanX)
betaridgenoi = betaridge[1:,:]
betaridge2 = betaridge
betaridge2[0] = betaridge2[0] - dot(meanX, betaridgenoi)
yhatridge2 = Xintercept2*betaridge2
RSSridge2 = transpose(y2 - yhatridge2)*(y2 - yhatridge2)

print 'beta (ridge) =',betaridge2


"""Principal Components Regression Model, fit to training, then fit pcr
    training model, to test data"""  
    
betapcr, yhatpcr, RSSpcr = principalReg(Xcentered,y,5)
#Remove the centering bais in the pcr models beta hat
betapcrnoi = betapcr[1:,:]
betapcr2 = betapcr
betapcr2[0] = betapcr2[0] - dot(meanX, betapcrnoi)
yhatpcr2 = Xintercept2*betapcr2
RSSpcr2 = transpose(y2 - yhatpcr2)*(y2 - yhatpcr2)

print 'beta (pcr) =',betapcr


#ridgeRegressionTesting(X,y)
#principalComponentTesting(X2,y2)


print ' '
print 'Full model     : RSS(train), RSS(test) =',RSSfull,RSSfull2
print 'Reduced model  : RSS(train), RSS(test) =',RSSred,RSSred2
print 'Ridge model    : RSS(train), RSS(test) =',RSSridge,RSSridge2
print 'PCR model      : RSS(train), RSS(test) =',RSSpcr,RSSpcr2
print ' '


## Test via cross-validation
print 'Cross-validation'
print ' '
K = 5 # divide training data into K sets
RSSCV = zeros((K,4)) # Storage for RSS
setno = zeros((N,))
for i in range(N):
    setno[i] = random.randint(0,K,1) # assign data randomly among K sets
for k in range(K):
    yk = y[setno == k] ## data in kth set
    Xk = X[setno == k,:]    
    Xkred = Xred[setno == k,:]
    Xkc = Xcentered[setno == k,:]
    
    
    
    ynotk = y[setno != k] ## data not in kth set
    Xnotk = X[setno != k,:]
    Xnotkred = Xred[setno != k,:]
    Xnotkc = Xcentered[setno != k,:] 
    
    
    
    betakfull = linregp(Xnotk,ynotk)[0] ## fit model to data not in kth set
    betakred = linregp_ni(Xnotkred,ynotk)[0]
    betakridge = ridgereg(Xnotkc,ynotk,32)[0]
    betakpcr = principalReg(Xnotkc, ynotk,5)[0]
    
    #Change the pcr and ridge betahat to suit uncentered data

    betakridgenoi = betakridge[1:,:]
    betakridge2 = betakridge
    betakridge2[0] = betakridge2[0] - dot(meanX, betakridgenoi)
    betakridge = betakridge2
    
    betakpcrnoi = betakpcr[1:,:]
    betakpcr2 = betakpcr
    betakpcr2[0] = betakpcr2[0] - dot(meanX, betakpcrnoi)
    betakpcr = betakpcr2
    
    
    Xk = concatenate((ones((shape(Xk)[0],1)),Xk),axis=1)
    
    ykfull = Xk*betakfull ## test model on data in kth set    
    RSSCV[k,0] = transpose(yk - ykfull)*(yk - ykfull)    
        
    ykred = Xkred*betakred
    RSSCV[k,1] = transpose(yk - ykred)*(yk - ykred)
    
    ykridge = Xk*betakridge
    RSSCV[k,2] = transpose(yk - ykridge)*(yk - ykridge)
    
    ykpcr = Xk*betakpcr
    RSSCV[k,3] = transpose(yk - ykpcr)*(yk - ykpcr)
    
    
    

MRSSCV = mean(RSSCV, axis = 0)  ## get average over all data

print 'Total Loss (CV) [full | reduced | ridge | principal components] = \n',RSSCV
print ' '
print 'Mean Loss RSS [full | reduced | ridge | principal components] = \n',MRSSCV