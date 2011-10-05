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

def linregp_ni(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    betahat = linalg.inv(transpose(X)*X)*transpose(X)*y
    yhat = X*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat,yhat,RSS

# training data
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

## test data
data2 = loadtxt('prostate_test.txt')
data2 = data2.reshape(-1,p+1)
## reformat data as X and Y
y2 = data2[:,p]
X2 = data2[:,0:p]
y2 = transpose(matrix(y2))
X2 = matrix(X2)
for i in range(p):
  X2[:,i] = X2[:,i]/sdX[i] # standardise according to TRAINING data variance
Xred2 = concatenate((X2[:,0:2],X2[:,3:5]),axis=1)

## Test on independent data
print 'Independent training and test data'
# full model, training
tempfull = linregp(X,y)
betafull = tempfull[0]
yhatfull = tempfull[1]
RSSfull = tempfull[2]
print 'beta (full) =',betafull
# full model, test
# fit full model, training, to test data
X21 = concatenate((ones((shape(X2)[0],1)),X2),axis=1)
yhatfull2 = X21*betafull
RSSfull2 = transpose(y2 - yhatfull2)*(y2 - yhatfull2)
# reduced model, training
# reduced model, training
tempred = linregp_ni(Xred,y)
betared = tempred[0]
yhatred = tempred[1]
RSSred = tempred[2]
print 'beta (reduced) =',betared
# fit reduced model, training, to test data
yhatred2 = Xred2*betared
RSSred2 = transpose(y2 - yhatred2)*(y2 - yhatred2)
print 'full model: RSS(train), RSS(test) =',RSSfull,RSSfull2
print 'reduced model: RSS(train), RSS(test) =',RSSred,RSSred2
print ' '


## Bootstrap
print 'Bootstrap'
B = 100  # number of bootstrap samples
X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)   
BRSS = zeros((B,2))    # bootstrap estimated RSS [full | reduced models]
BRSScv = zeros((N,3))  # error for yi [full | reduced | number of estimates]
for b in range(B):
    ytemp = matrix(zeros(shape(y)))  ## bootstrap samples
    Xtemp = matrix(zeros(shape(X)))
    Xredtemp = matrix(zeros(shape(Xred)))
    unselected = list(arange(N))     ## the (un)selected elements we will test over (CV)
    for n in range(N):
        temp = random.randint(0,N,1)
        if sum(unselected == temp) > 0:
            unselected.remove(temp)
        ytemp[n] = y[temp]   ## add selected to bootstrap sample
        Xtemp[n,:] = X[temp,:]
        Xredtemp[n,:] = Xred[temp,:]
    bootfull = linregp(Xtemp,ytemp)[0] ## fit to bootstrap samples
    bootred = linregp_ni(Xredtemp,ytemp)[0]
    BRSS[b,0] = transpose(y - X1*bootfull)*(y - X1*bootfull)/N ## bootstrap errors
    BRSS[b,1] = transpose(y - Xred*bootred)*(y - Xred*bootred)/N
    lenus = len(unselected)
    for i in range(lenus):   ## bootstrap CV errors
        j = unselected[i]
        BRSScv[j,0] = BRSScv[j,0] + (y[j] - X1[j,:]*bootfull)**2
        BRSScv[j,1] = BRSScv[j,1] + (y[j] - Xred[j,:]*bootred)**2
        BRSScv[j,2] = BRSScv[j,2] + 1
for n in range(N):  ## normalize bootstrap CV errors by number of bootstrap sets
    BRSScv[n,0] = BRSScv[n,0]/BRSScv[n,2]
    BRSScv[n,1] = BRSScv[n,1]/BRSScv[n,2]
BRSScv = BRSScv[:,0:2]

Errboot = mean(BRSS,axis = 0)
print 'Bootstrap Error [full | reduced]=',Errboot

Errbootcv = mean(BRSScv,axis = 0)
print 'Bootstrap CV Error [full | reduced]=',Errbootcv

errfull = float(linregp(X,y)[2]/N)  ## Training error
errred = float(linregp_ni(Xred,y)[2]/N)
errall = array([errfull,errred])
est632 = 0.632*Errbootcv + 0.368*errall
print 'Bootstrap 0.632 estimator [full | reduced]=',est632