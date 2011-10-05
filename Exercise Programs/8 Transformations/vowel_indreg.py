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

data = loadtxt('vowel_train.txt')
p = 10
data = data.reshape(-1,p+1)

## reformat data as X and Y
y = data[:,0]
N = len(y)
y = transpose(matrix(y))
X = data[:,1:p+1]
X = matrix(X)
#print y

# make an indicator matrix Y
K = 11 # number of classes
Y = zeros((N,K))
for n in range(N):
    Y[n,int(y[n])-1] = 1   
#print Y

#temp = linregp(X,Y)
#betahat = temp[0]
Yhat = linregp(X,Y)[1]

#print Yhat.shape
YhatT = transpose(Yhat)
yhatT = YhatT.argmax(0) + 1
yhat = transpose(yhatT)
errorrate = (0. + N - sum(y == yhat))/N
#print(sum(y == yhat))
#print(len(y))
print 'errorrate = ',errorrate
