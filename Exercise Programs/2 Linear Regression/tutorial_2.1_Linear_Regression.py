#os.chdir("C:\Documents and Settings\\mbebbing\\My Documents\\Teaching\\161326")
from numpy import *
from pylab import *

data = loadtxt('age.txt')   
data = data.reshape(-1,2)
#print data

# separate data into x and y using slice operator
y = data[:,0]
x = data[:,1]

#figure(1)
#plot(x,y,'.')
#xlabel('Height (cm)')
#ylabel('Age (months)')
#show()   # forces pylab to print current figure

# jitter data for plotting
xtemp = x + uniform(-0.25, 0.25, x.shape)
ytemp = y + uniform(-0.25, 0.25, y.shape)
# this adds a uniform r.v. on (-0.25,0.25) to each
# x and y value to separate them so we can see
# all the points. The last argument produces a
# random vector the same size as x, so you can 
# add them together.
figure(2)
plot(xtemp,ytemp,'.')
xlabel('Height (cm)')
ylabel('Age (months)')
#show()   # forces pylab to print current figure

N = len(x) # length of the vector x
print 'N =',N

x = transpose(matrix(x)) 
y = transpose(matrix(y)) 
#'matrix(z)' makes array z into a matrix. Note the transpose. NumPy makes all 1-D arrays into ROW vectors
#print 'x.shape =',x.shape
x1 = concatenate((ones((shape(x)[0],1)),x),axis=1)
#'shape(x)[0]' is the number of rows of x, 
#'ones(N,M)' is an NxM matrix of ones, 
#'concatenate((A,B),axis = 1)' joins A and B together along axis 1, ie, side by side.
print x1

betahat = linalg.inv(transpose(x1)*x1)*transpose(x1)*y
# we have invoked the 'inverse' function from a linear 
# algebra package. The inverse of z is inv(z) such 
# that z*inv(z) is the identity matrix (ones on 
# diagonal, zeros off-diagonal).
print 'betahat =',betahat     
# Note that betahat is now a 2x1 matrix
                  
# calculate endpoints of regression line and plot
minx = float(min(x))
maxx = float(max(x))
miny = float(betahat[0,0] + betahat[1,0]*minx)
maxy = float(betahat[0,0] + betahat[1,0]*maxx)
xline = array([minx,maxx])  # note how to 
yline = array([miny,maxy])  # input a vector
#print xline,yline
figure(2)  # adds line to previous plot
plot(xline,yline,'-')

show() # forces pylab to print all figures