from pylab import *
from numpy import *

data = loadtxt('saheartdis.txt')
P = 9
data = data.reshape(-1,P+1)

## reformat data as X and Y
order = range(shape(data)[0])
random.shuffle(order)
data = data[order,:]
#y = data[0:308,9:10]
y = pow(data[0:308,9:10], 2)
N = len(y)
K = 2

#X = data[0:308,0:9]
X = pow(data[0:308,0:9], 2)
X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
X1 = matrix(X1)

#ytest = data[308:462,9:10]
ytest = pow(data[308:462,9:10], 2)
Ntest = len(ytest)
#Xtest = data[308:462:,0:9]
Xtest = pow(data[308:462:,0:9], 2)
X1test = concatenate((ones((shape(Xtest)[0],1)),Xtest),axis=1)
X1test = matrix(X1test)

# initialize
betaold = zeros((P+1,1))
tol = 0.000001
stepdiff = 1
while stepdiff > tol:
    p = exp(X1*betaold)
    p = p/(1+p)
    temp = multiply(p,(ones(shape(p)) - p))
    W = zeros((N,N))
    for j in range(N):
        W[j,j] = temp[j,0]
    beta = betaold + (linalg.inv(transpose(X1)*W*X1))*transpose(X1)*(matrix(y) - p)
    stepdiff = abs(betaold - beta).max()
    betaold = beta
    #print stepdiff

"""Create Parsimonious Model (Predictors 1,4,7,10)"""

parsibeta = copy(beta)
#parsibeta[0][0] = 0.0
parsibeta[1][0] = 0.0
parsibeta[2][0] = 0.0
#parsibeta[3][0] = 0.0
parsibeta[4][0] = 0.0
parsibeta[5][0] = 0.0
parsibeta[7][0] = 0.0
parsibeta[8][0] = 0.0


"""Setup testing for test and training full beta models"""

p = exp(X1*beta)
p = p/(1+p)
ptest = exp(X1test*beta)
ptest = ptest/(1+ptest)

yhat = (p >= 0.5)
errorrate = (0. + N - sum(y == yhat))/N
yhattest = (ptest >= 0.5)
errorratetest = (0. + Ntest - sum(ytest == yhattest))/Ntest
print "------------------------------------------------------"
print "Error in full model"
print "------------------------------------------------------"
print "Train Errorrate = ",errorrate
print "Test Errorrate  = ",errorratetest


""" The following commands test the Parsimonious model on both the test
and training set"""
for i in range(3,7):
    if i == 4:
        parsibeta = copy(beta)        
        parsibeta[1][0] = 0.0
        parsibeta[2][0] = 0.0        
        parsibeta[4][0] = 0.0        
        parsibeta[7][0] = 0.0
        parsibeta[8][0] = 0.0
    if i == 5:
        parsibeta = copy(beta)
        parsibeta[1][0] = 0.0                
        parsibeta[4][0] = 0.0        
        parsibeta[7][0] = 0.0
        parsibeta[8][0] = 0.0
    if i == 6:
        parsibeta = copy(beta)
        parsibeta[4][0] = 0.0        
        parsibeta[7][0] = 0.0
        parsibeta[8][0] = 0.0
    
    
    # Parsimonious Model
    p2 = exp(X1*parsibeta)
    p2 = p2/(1+p2)
    ptest2 = exp(X1test*parsibeta)
    ptest2 = ptest2/(1+ptest2)
    
    # Testing Parsimonious Model
    yhat = (p2 >= 0.5)
    errorrate = (0. + N - sum(y == yhat))/N
    yhattest = (ptest2 >= 0.5)
    errorratetest = (0. + Ntest - sum(ytest == yhattest))/Ntest
    print "------------------------------------------------------"
    print "Error rate of Model with", i, "Predictors"
    print "------------------------------------------------------"
    print "Train Errorrate = ",errorrate
    print "Test Errorrate  = ",errorratetest



# check z-scores 
temp = multiply(p,(ones(shape(p)) - p))
W = zeros((N,N))
for j in range(N):
    W[j,j] = temp[j,0]
varbeta = linalg.inv(transpose(X1)*W*X1)
z = multiply(beta,1/transpose(matrix(sqrt(diag(varbeta)))))
print "\n"
print 'beta, z = ',concatenate((beta,z),axis=1)


        