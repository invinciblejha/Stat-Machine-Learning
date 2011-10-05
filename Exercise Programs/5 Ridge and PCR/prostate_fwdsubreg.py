### Stepwise Regression ###

from pylab import *
from numpy import *


def linregp_ni(X,y):
    # takes N x p matrix X and a N x 1 vector y, and fits  
    # linear regression y = X*beta. NOTE indenting
    betahat = linalg.inv(transpose(X)*X)*transpose(X)*y
    yhat = X*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat,yhat,RSS

def fstepreg(X,y,p):
   oldbeta = mean(array(y))  #start with intercept
   oldX1 = ones(shape(y))  #intercept term
   oldRSS = transpose(y - oldX1*oldbeta)*(y - oldX1*oldbeta)
   inclpred = [-1]      # this is a LIST of included predictors
                        #included predictor = -1, the intercept
                        #--remember indexing starts at 0 in X
   exclpred = list(arange(p))
   continu = 1
   while continu:
      bestRSS = 100*oldRSS  #some large number
      for i in range(len(exclpred)): #try adding each predictor to current model
          j = exclpred[i]
          X1 = concatenate((oldX1,X[:,j:j+1]),axis=1) 
               #add jth column of X to current predictor matrix
          beta = linregp_ni(X1,y)[0]  #fit lin reg model
          RSS = transpose(y - X1*beta)*(y - X1*beta)
          if RSS < bestRSS:  # best additional predictor so far
               bestRSS = RSS   # --record the details
               bestpred = j  
               bestbeta = beta
      F = (oldRSS - bestRSS)/(bestRSS/(len(y) - len(bestbeta) - 1))
         #find the F-ratio for the best additional predictor
      if F > F90[len(y) - len(bestbeta) - 1,1]:  #significant?
                                                 #--update model
         oldX1 = concatenate((oldX1,X[:,bestpred:bestpred+1]),axis=1)
         oldbeta = bestbeta
         oldRSS = bestRSS
         inclpred.append(bestpred)
         exclpred.remove(bestpred)
      else:  #finished
         continu = 0 
      if len(exclpred) == 0:  #stop if all predictors included
         continu = 0
   return oldbeta,oldRSS,inclpred,exclpred


## get data
data = loadtxt('prostate_train.txt')
data = data.reshape(-1,9)

## get alpha = 0.9 values of F distribution
F90 = loadtxt('F90.txt')
F90 = F90.reshape(-1,10)

#print data

## reformat data as X and Y
p = 8
data = data.reshape(-1,p+1)

## reformat data as X and Y
y = data[:,p]
X = data[:,0:p]
y = transpose(matrix(y))
X = matrix(X)

covX = cov(transpose(X)) # need to transpose X to get a p x p covariance matrix
sdX = sqrt(diag(covX)) # Note that sdX is an array
#print 'sdX = ',sdX

# standardize data to unit variance
for i in range(p):
  X[:,i] = X[:,i]/sdX[i]

temp = fstepreg(X,y,p)
betahat = temp[0]
RSS = temp[1]
inclpred = temp[2]
exclpred = temp[3]

print 'included predictors = ',inclpred
print 'excluded predictors = ',exclpred
print 'betahat = ',betahat
print 'RSS = ',float(RSS)

            




















