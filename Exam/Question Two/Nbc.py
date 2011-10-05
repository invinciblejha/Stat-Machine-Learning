from pylab import *
from numpy import *
def nbayesclass(X1,y1,X2,y2):
    # naive Bayes classifier for y = {0,1,2}, X a matrix of p predictors
    N1 = len(y1)
    N2 = len(y2)
    p = X1.shape[1]    
    f1k = zeros(shape(X1),dtype=float) # density estimate for y = 1, kth predictor
    f2k = zeros(shape(X1),dtype=float) # density estimate for y = 2, kth predictor
    f3k = zeros(shape(X1),dtype=float) # density estimate for y = 3, kth predictor    
    f5k = zeros(shape(X1),dtype=float) # density estimate for y = 5, kth predictor
    f6k = zeros(shape(X1),dtype=float) # density estimate for y = 6, kth predictor
    f7k = zeros(shape(X1),dtype=float) # density estimate for y = 7, kth predictor
    for k in range(p):
        x = X1[:,k:k+1] # get kth predictor        
        x1 = x[y1 == 1] # value of y
        x2 = x[y1 == 2]
        x3 = x[y1 == 3]        
        x5 = x[y1 == 5]
        x6 = x[y1 == 6]
        x7 = x[y1 == 7]
        xk = X2[:,k:k+1] # test values, kth predictor
        lam = 0.05*(max(x) - min(x)) # bandwidth
        lsq = float(lam**2)
        # calculate density estimates
        for i in range(N2):
            xki = float(xk[i])
            #print pi,lsq,x0.shape,xk.shape            
            K1 = (1/sqrt(2*pi))*exp(-(multiply(x1-xki,x1-xki))/(2*lsq))
            K2 = (1/sqrt(2*pi))*exp(-(multiply(x2-xki,x2-xki))/(2*lsq))
            K3 = (1/sqrt(2*pi))*exp(-(multiply(x3-xki,x3-xki))/(2*lsq))           
            K5 = (1/sqrt(2*pi))*exp(-(multiply(x5-xki,x5-xki))/(2*lsq))
            K6 = (1/sqrt(2*pi))*exp(-(multiply(x6-xki,x6-xki))/(2*lsq))
            K7 = (1/sqrt(2*pi))*exp(-(multiply(x7-xki,x7-xki))/(2*lsq))
            
            f1k[i,k] = sum(K1)/(lam*len(x1))
            f2k[i,k] = sum(K2)/(lam*len(x2))
            f3k[i,k] = sum(K3)/(lam*len(x3))            
            f5k[i,k] = sum(K5)/(lam*len(x5))
            f6k[i,k] = sum(K6)/(lam*len(x6))
            f7k[i,k] = sum(K7)/(lam*len(x7))
            
    # multiply over predictors    
    f1 = transpose(matrix(f1k[:,0]))
    f2 = transpose(matrix(f2k[:,0]))
    f3 = transpose(matrix(f3k[:,0]))    
    f5 = transpose(matrix(f5k[:,0]))
    f6 = transpose(matrix(f6k[:,0]))
    f7 = transpose(matrix(f7k[:,0]))
    for k in range(p-1):        
        f1 = multiply(f1,f1k[:,k+1:k+2])
        f2 = multiply(f2,f2k[:,k+1:k+2])
        f3 = multiply(f3,f3k[:,k+1:k+2])        
        f5 = multiply(f5,f5k[:,k+1:k+2])
        f6 = multiply(f6,f6k[:,k+1:k+2])
        f7 = multiply(f7,f7k[:,k+1:k+2])
    nbc = zeros(shape(y2))
    for i in range(N2):
        if f1[i] > max(f2[i],f3[i],f5[i],f6[i],f7[i]):            
            nbc[i] = 1
        elif f2[i] > max(f1[i],f3[i],f5[i],f6[i],f7[i]):
            nbc[i] = 2
        elif f3[i] > max(f1[i],f2[i],f5[i],f6[i],f7[i]):
            nbc[i] = 3        
        elif f5[i] > max(f1[i],f2[i],f3[i],f6[i],f7[i]):
            nbc[i] = 5  
        elif f6[i] > max(f1[i],f2[i],f3[i],f5[i],f7[i]):
            nbc[i] = 6
        else:
            nbc[i] = 7   
    errorrate = (0. + N2 - sum(y2 == nbc))/N2
    return errorrate

# get iris data
data = loadtxt('Q2.data',delimiter=',')
p = 10
data = data.reshape(-1,p+1)
order = range(shape(data)[0])
random.shuffle(order)
data = data[order,:]
## reformat data as X and Y
y = data[:,p]
N = len(y)
y = transpose(matrix(y))
X = data[:,0:p]
X = matrix(X)
K = 6 # number of classes


M = 20 # number of random training/test assignments
ER = zeros((M,1)) # Storage for error rates
for m in range(M):
    # randomly divide data into training and test data
    setsize = int(N*2/3)
    setno0 = random_sample((N,))
    setno1 = sort(setno0)
    boundary = setno1[setsize]
    X1 = X[setno0 < boundary,:]
    y1 = y[setno0 < boundary]
    X2 = X[setno0 >= boundary,:]
    y2 = y[setno0 >= boundary]    
    nbcer = nbayesclass(X1,y1,X2,y2)    
    ER[m,0] = nbcer
    print 'Naive Bayes error rate =',nbcer
print ' '
print 'Mean Error Rate [NBC]=',mean(ER, axis = 0)


