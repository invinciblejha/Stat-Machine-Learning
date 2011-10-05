from pylab import *
from numpy import *

def linreg(X,y):
    X = concatenate((ones((shape(X)[0],1)),X),axis=1)
    beta = dot(dot(linalg.inv(dot(transpose(X),X)),transpose(X)),y)
    yhat = dot(X,beta)
    RSS = dot(yhat-y,yhat-y)
    return beta,yhat,RSS

def linreg0(X,y):
   beta = dot(dot(linalg.inv(dot(transpose(X),X)),transpose(X)),y)
   yhat = dot(X,beta)
   RSS = dot(yhat-y,yhat-y)
   return beta,yhat,RSS

# training data
data = loadtxt('prostate_train.txt')
data = data.reshape(-1,9)
## reformat data as X and Y
y = data[:,8]
N = len(y)
X = data[:,0:8]
covX = cov(transpose(matrix(X)))
sdX = sqrt(diag(covX))
for i in range(8):
  X[:,i] = X[:,i]/sdX[i]
## test data
data2 = loadtxt('prostate_test.txt')
data2 = data2.reshape(-1,9)
## reformat data as X and Y
y2 = data2[:,8]
y2 = transpose(matrix(y2))
X2 = data2[:,0:8]
for i in range(8):
  X2[:,i] = X2[:,i]/sdX[i]
Xred2 = concatenate((X2[:,0:2],X2[:,3:5]),axis=1)

## Test on independent data
print 'Independent training and test data'
# full model, training
betafull = linreg(X,y)[0]
betafull = transpose(matrix(betafull))
yfull = transpose(matrix(linreg(X,y)[1]))
print 'beta (full) =',betafull
# full model, test
X2 = concatenate((ones((shape(X2)[0],1)),X2),axis=1)
#print y2.shape,X2.shape,betafull.shape
RSSfull = transpose(y2 - X2*betafull)*(y2 - X2*betafull)
# reduced model, training
Xred = concatenate((X[:,0:2],X[:,3:5]),axis=1)
betared = linreg0(Xred,y)[0]
betared = transpose(matrix(betared))
yred = transpose(matrix(linreg0(Xred,y)[1]))
print 'beta (reduced) =',betared
# reduced model, test
RSSred = transpose(y2 - Xred2*betared)*(y2 - Xred2*betared)
print 'full model, RSS(test) =',RSSfull
print 'reduced model, RSS(test) =',RSSred

print ' '

## Calculate AIC
print 'AIC'
# get sigma2 from reduced (low-bias) model
sigma2 = (std(yred - y))**2
print 'sigma2=',sigma2
# calculate AIC
aicfull = linreg(X,y)[2]/len(y) + 2.0*len(betafull)*sigma2/len(y) ## AIC
aicred = linreg0(Xred,y)[2]/len(y) + 2.0*len(betared)*sigma2/len(y)  ## AIC
print 'AIC (full | reduced) =',aicfull,aicred