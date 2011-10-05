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

def baggedModel(X,y):
    
    B = 100  # number of bootstrap samples
    baggedbeta = matrix(zeros((p+1,B)))
    for b in range(B):
        ytemp = matrix(zeros(shape(y)))  ## bootstrap samples
        Xtemp = matrix(zeros(shape(X)))
        for n in range(N):
            temp = random.randint(0,N,1)
            ytemp[n] = y[temp]   ## add selected to bootstrap sample
            Xtemp[n,:] = X[temp,:]
        bootbeta = linregp(Xtemp,ytemp)[0] ## fit to bootstrap samples
        baggedbeta[:,b] = bootbeta
    
    baggedbetahat = mean(baggedbeta, axis = 1) # take mean for bagged estimate
    return baggedbetahat

def fillOutBeta(predictors, beta):
    
    """This method creates an array that can be used as a betahat
    it adds zeros to where we are not using predictors"""
    
    temp = []
    for idx in range(0,9):
        if predictors.count(idx) > 0:
            temp.append(float(beta[predictors.index(idx)]))
        else:
            temp.append(0.0)   
        
    return transpose(matrix(temp))

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

covX = cov(transpose(X)) # need to transpose X to get a p x p covariance matrix
sdX = sqrt(diag(covX)) # Note that sdX is an array

for i in range(p):
  X[:,i] = X[:,i]/sdX[i]

Xred = concatenate((X[:,0:2],X[:,3:5]),axis=1)      # Reduced model
Xbackward = concatenate((X[:,0:6],X[:,7:8]),axis=1) # Backward model
Xbrute = concatenate((X[:,0:2],X[:,4:5]),axis=1)    # Brute force model

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
  X2[:,i] = X2[:,i]/sdX[i] 
  
Xred2 = concatenate((X2[:,0:2],X2[:,3:5]),axis=1)       # Reduced model
Xbackward2 = concatenate((X2[:,0:6],X2[:,7:8]),axis=1)  # Backward model
Xbrute2 = concatenate((X2[:,0:2],X2[:,4:5]),axis=1)     # Brute force model

#Center the inputs

X2centered = X2 - ones(shape(y2))*meanX

#Add intercept to X2, with centered inputs
Xintercept = concatenate((ones((shape(X)[0],1)),X),axis=1)   
Xintercept2 = concatenate((ones((shape(X2)[0],1)),X2),axis=1)
Xintercept2Centered = concatenate((ones((shape(X2centered)[0],1)),X2centered),axis=1)

print 'Independent training and test data'


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
fullbetareduced = fillOutBeta([1,2,4,5],betared)
print 'beta (reduced) =',fullbetareduced



"""Backward Regression Model, fit to training, then fit backward 
    training model, to test data"""
    
betaback, yhatback, RSSback = linregp_ni(Xbackward,y)
yhatback2 = Xbackward2*betaback
RSSback2 = transpose(y2 - yhatback2)*(y2 - yhatback2)
fullbetaback = fillOutBeta([1,2,3,4,5,6,8],betaback)
print 'beta (backward) =',fullbetaback



"""Ridge Regression Model, fit to training, then fit ridge regression
    training model, to test data"""    

ridgelambda = 32
betaridge, yhatridge, RSSridge = ridgereg(Xcentered,y,ridgelambda)
yhatridge2 = Xintercept2Centered*betaridge
RSSridge2 = transpose(y2 - yhatridge2)*(y2 - yhatridge2)

#Calculate Lambda
dflambda = trace(Xcentered*linalg.inv(transpose(Xcentered)*Xcentered+ridgelambda*eye(8))*transpose(Xcentered))



print 'beta (ridge) =',betaridge

"""Ridge Regression Model, Remove the centering bais in the ridge models beta hat"""  

meanX = matrix(meanX)
betaridgenoi = betaridge[1:,:]
betaridge3 = betaridge
betaridge3[0] = betaridge3[0] - dot(meanX, betaridgenoi)
yhatridge3 = Xintercept2*betaridge3
RSSridge3 = transpose(y2 - yhatridge3)*(y2 - yhatridge3)

print 'beta (ridge no centering) =', betaridge3


"""Brute Force Model, fit to training, then fit brute force 
    training model, to test data"""
    
betabrute, yhatbrute, RSSbrute = linregp_ni(Xbrute,y)
yhatbrute2 = Xbrute2*betabrute
RSSbrute2 = transpose(y2 - yhatbrute2)*(y2 - yhatbrute2)
fullbetabrute = fillOutBeta([1,2,4],betabrute)

print 'beta (brute) =',fullbetabrute


"""Calculate the bagged model betahat."""

betabagged = baggedModel(X,y)
print 'beta (bagged) =', betabagged


print ' '
print 'Full model     : RSS(train), RSS(test) =',RSSfull,RSSfull2
print 'Reduced model  : RSS(train), RSS(test) =',RSSred,RSSred2
print 'Backward model : RSS(train), RSS(test) =',RSSback,RSSback2
print 'Brute model    : RSS(train), RSS(test) =',RSSbrute,RSSbrute2
print 'Ridge model    : RSS(train), RSS(test) =',RSSridge,RSSridge3
print ' '


# get sigma2 from full (low-bias) model
sigma2 = (std(yhatfull - y))**2
print 'sigma2 =',sigma2

""" Calculate AIC """
print ' ' 
aicfull =   RSSfull/N   + 2.0 * len(betafull)   *sigma2/N 
aicred =    RSSred/N    + 2.0 * len(betared)    *sigma2/N 
aicback =   RSSback/N   + 2.0 * len(betaback)   *sigma2/N 
aicbrute =  RSSbrute/N  + 2.0 * len(betabrute)  *sigma2/N 
aicridge =  RSSridge/N  + 2.0 * len(betaridge)  *sigma2/N 

print 'AIC (full | reduced | backward | brute | ridge) =', aicfull, aicred, aicback, aicbrute, aicridge


""" Question 1 - part (a) i - i. Calculate the BIC, and use these to calculate the posterior probabilities """

""" Calculate BIC """

print ' '
bicfull =   N/sigma2 *   ((RSSfull/N)   + math.log(N) * (len(betafull)  * sigma2/N))
bicred  =   N/sigma2 *   ((RSSred/N)    + math.log(N) * (len(betared)   * sigma2/N))
bicback =   N/sigma2 *   ((RSSback/N)   + math.log(N) * (len(betaback)  * sigma2/N))
bicbrute =  N/sigma2 *   ((RSSbrute/N)  + math.log(N) * (len(betabrute) * sigma2/N))
bicridge =  N/sigma2 *   ((RSSridge/N)  + math.log(N) * (dflambda * sigma2/N))
print 'BIC (full | reduced | backward | brute | ridge) =', bicfull, bicred, bicback, bicbrute, bicridge

"""Posterior probabilities"""

ebicfull  = math.exp(-(.5 * bicfull))
ebicred   = math.exp(-(.5 * bicred))
ebicback  = math.exp(-(.5 * bicback))
ebicbrute = math.exp(-(.5 * bicbrute))
ebicridge = math.exp(-(.5 * bicridge))
sumebic = ebicfull + ebicred + ebicback + ebicbrute + ebicridge

print ' '
print 'Full Model Posterior probability                 ', int(ebicfull/sumebic*100),"%"
print 'Reduced Model Posterior probability              ', int(ebicred/sumebic*100),"%"
print 'Backwards Regression Model Posterior probability ', int(ebicback/sumebic*100),"%"
print 'Brute Force Model Posterior probability          ', int(ebicbrute/sumebic*100),"%"
print 'Ridge Regression Model Posterior probability     ', int(ebicridge/sumebic*100),"%"
print ' '


""" Question 1 - Part (a) ii - Compare the prediction errors on training and test data """

## Training Optimism
opfull  = 2.0*len(betafull)  *sigma2/len(y)
opred   = 2.0*len(betared)   *sigma2/len(y)
opback  = 2.0*len(betaback)  *sigma2/len(y)
opbrute = 2.0*len(betabrute) *sigma2/len(y)
opridge = 2.0*len(betaridge) *sigma2/len(y)

## Training error
errfull  = RSSfull  / len(y)
errred   = RSSred   / len(y)
errback  = RSSback  / len(y)
errbrute = RSSbrute / len(y)
errridge = RSSridge / len(y)

## In sample error - Training
Errfull   = errfull  + opfull
Errred    = errred   + opred
Errback   = errback  + opback
Errbrute  = errbrute + opbrute
Errridge  = errridge + opridge


print '(error | optimism | total) (full)       =',errfull, opfull, '',Errfull
print '(error | optimism | total) (reduced)    =',errred, opred, Errred
print '(error | optimism | total) (backward)   =',errback, opback, Errback
print '(error | optimism | total) (brute)      =',errbrute, opbrute, Errbrute
print '(error | optimism | total) (ridge)      =',errridge, opridge, '',Errridge


""" Part (b) - Calculate the Average model, and compare the performance with that of 
    the bagged model on the test data."""
    
avgbeta = array   
avgbeta = betafull * (ebicfull / sumebic)
avgbeta = avgbeta + fullbetareduced * (ebicred / sumebic)
avgbeta = avgbeta + fullbetaback * (ebicback / sumebic)
avgbeta = avgbeta + fullbetabrute * (ebicbrute / sumebic)
avgbeta = avgbeta + betaridge3 * (ebicridge / sumebic)

yhatavg = Xintercept2*avgbeta
RSSavg = transpose(y2 - yhatavg)*(y2 - yhatavg)

yhatbagged = Xintercept2*betabagged
RSSbagged = transpose(y2 - yhatbagged)*(y2 - yhatbagged)

print " "
print "Average Beta = ", avgbeta
print "Average Model    : RSS(Test) =", RSSavg
print "Bagged Model     : RSS(Test) =", RSSbagged

