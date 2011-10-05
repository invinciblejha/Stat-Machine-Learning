from numpy import *
from pylab import *

def linreg1(x,y):
    # takes Nx1 matrices x and y, and performs linear regression y = beta x
    x1 = concatenate((ones((shape(x)[0],1)),x),axis=1)
    V = linalg.inv(transpose(x1)*x1)
    betahat = V*transpose(x1)*y
    yhat = x1*betahat
    RSS = transpose(y - yhat)*(y - yhat)
    return betahat,yhat,RSS,V

data = loadtxt('age.txt')   
data = data.reshape(-1,2)
#print data

# separate data into x and y using slice operator
y = data[:,0]
x = data[:,1]

#print x.shape
#print y.shape

x = transpose(matrix(x)) 
y = transpose(matrix(y)) 
#'matrix(z)' makes array z into a matrix. Note the transpose. NumPy makes all 1-D arrays into ROW vectors
#print 'x.shape =',x.shape


# jitter data for plotting
xtemp = x + uniform(-0.25, 0.25, x.shape)
# this adds a uniform r.v. on (-0.25,0.25) to each
# x value to separate them so we can see
# all the points. The last argument produces a
# random vector the same size as x, so you can 
# add them together.

N = len(x) # length of the vector x
print 'N =',N

temp = linreg1(x,y) # this avoids calling the function 3 times
betahat = temp[0]
print 'betahat =',betahat     
yhat = temp[1]
residual = y - yhat # residuals
RSS = temp[2] # residual sum of squares
print 'RSS =',RSS


#print residual.shape
#print xtemp.shape

s2 = RSS/(N-2)
print 's2 =',s2

V = temp[3]
tstat = linalg.inv(diag(sqrt(diag(V))))*betahat/sqrt(s2) 
#take diagonal elements of V, square-root, then make into 
#diagonal matrix, invert (i.e., v -> 1/v), etc., to get
#student t on N-2 d.o.f.
print 'tstat =',tstat # approx. > 2 is significantly different from zero
                 
#figure(2)
#plot(array(xtemp),array(residual),'.')
#xlabel('Height (cm), jittered')
#ylabel('Residual Age (months)')
#show()   # forces pylab to print current figure