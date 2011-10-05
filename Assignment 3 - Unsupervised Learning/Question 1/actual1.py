from pylab import *
from numpy import *

def fit_beta(X1,y):
    # fits a logistic regression model to the data in X1,y
    N = len(y)
    # number of data
    P1 = X1.shape[1] # number of predictors
    # initialise
    betaold = zeros((P1,1))
    tol = 0.000001
    stepdiff = 1
    while stepdiff > tol:
        p = exp(X1*betaold)
        p = p/(1+p)
        temp = multiply(p,(ones(shape(p)) - p))
        W = zeros((N,N))
        for j in range(N):
            W[j,j] = temp[j,0]
        betanew = betaold + (linalg.inv(transpose(X1)*W*X1))*transpose(X1)*(matrix(y) - p)
        stepdiff = abs(betaold - betanew).max()
        betaold = betanew
    return betaold

def error_rate(X1,betahat,y):
    # calculates the error rate for a predictor betahat on data in X1,y, and returns z-scores
    N = len(y) # number of predictors
    p = exp(X1*betahat)
    p = p/(1+p)
    yhat = (p >= 0.5)
    errorrate = (0. + N - sum(y == yhat))/N
    # check z-scores
    temp = multiply(p,(ones(shape(p)) - p))
    W = zeros((N,N))
    for j in range(N):
        W[j,j] = temp[j,0]
    varbeta = linalg.inv(transpose(X1)*W*X1)
    z = multiply(betahat,1/transpose(matrix(sqrt(diag(varbeta)))))
    return errorrate,concatenate((y,p),axis = 1),concatenate((betahat,z),axis=1)

data = loadtxt('saheartdis.txt')
P = 9
data = data.reshape(-1,P+1)

# divide into 2/3 training, 1/3 test
traindata = data[0:308,:]
testdata = data[308:462,:]

## reformat data as X and Y
ytrain = traindata[:,9:10]
Ntrain = len(ytrain)
ytest = testdata[:,9:10]
Ntest = len(ytest)
K = 2 # number of classes
X = data[:,0:9]
X = matrix(X)

# add in quadratic terms, except for x4 (famhist)
for i in range(4):
    X = concatenate((X,multiply(X[:,i],X[:,i])), axis = 1)
for i in range(5,P,1):
    X = concatenate((X,multiply(X[:,i],X[:,i])), axis = 1)
X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)

# divide into 2/3 training, 1/3 test
X1train = X1[0:308,:]
X1test = X1[308:462,:]

# fit full model
betanew = fit_beta(X1train,ytrain)
#print 'beta= ',betanew
print 'beta, z = ',error_rate(X1train,betanew,ytrain)[2]
print 'error rate (train) = ',error_rate(X1train,betanew,ytrain)[0]
print ''

# fit parsimonious models based on previous z-values
print 'error rate (test)'

Xselect0 = array([5, 7, 9])
betanew = fit_beta(X1train[:,Xselect0],ytrain)
print 'Model x5,x7,x9 =',error_rate(X1test[:,Xselect0],betanew,ytest)[0]
print 'betanew =',betanew
print ''

Xselect1 = array([0, 5, 7, 9])
betanew = fit_beta(X1train[:,Xselect1],ytrain)
print 'Model x0,x5,x7,x9 =',error_rate(X1test[:,Xselect1],betanew,ytest)[0]
print 'betanew =',betanew
print ''

Xselect2 = array([0, 5, 7, 9, 15])
betanew = fit_beta(X1train[:,Xselect2],ytrain)
print 'Model x0,x5,x7,x9,x7*x7=',error_rate(X1test[:,Xselect2],betanew,ytest)[0]
print 'betanew =',betanew
print ''

Xselect3 = array([0, 3, 5, 7, 9, 15])
betanew = fit_beta(X1train[:,Xselect3],ytrain)
print 'Model x0,x3,x5,x7,x9,x7*x7=',error_rate(X1test[:,Xselect3],betanew,ytest)[0]
print 'betanew =',betanew
print ''

Xselect_1 = array([5, 7])
betanew = fit_beta(X1train[:,Xselect_1],ytrain)
print 'Model x5,x7 =',error_rate(X1test[:,Xselect_1],betanew,ytest)[0]
print 'betanew =',betanew
print ''

