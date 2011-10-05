from pylab import *
from numpy import *



def linreg1(x,y):
    # takes Nx1 matrices x and y, and performs
    # linear regression y = beta x. NOTE indenting
    print x.shape
    x1 = concatenate((ones((shape(x)[0],1)),x),axis=1)
    betahat = linalg.inv(transpose(x1)*x1)*transpose(x1)*y
    yhat = x1*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat,yhat,RSS




data = loadtxt('age.txt')   
data = data.reshape(-1,2)
#print data

# separate data into x and y using slice operator
y = data[:,0]
x = data[:,1]

# jitter data for plotting
xtemp = x + uniform(-0.25, 0.25, x.shape)
ytemp = y + uniform(-0.25, 0.25, y.shape)
# this adds a uniform r.v. on (-0.25,0.25) to each
# x and y value to separate them so we can see
# all the points. The last argument produces a
# random vector the same size as x, so you can 
# add them together.

x = transpose(matrix(x))
y = transpose(matrix(y))

        
temp = linreg1(x,y) # this avoids calling the function 3 times
betahat = temp[0]
yhat = temp[1]
residual = y - yhat # residuals
RSS = temp[2] # residual sum of squares

#print 'RSS =',RSS
#RSS = [[ 512.19534501]]

hist(array(residual))   

xlabel('Height (cm), jittered')
ylabel('Residual Age (months)')
show()



