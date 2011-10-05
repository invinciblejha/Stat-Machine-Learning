from numpy import *
import numpy.random as random

def makedataA1():
    
    a = squeeze(transpose([0.1*random.randn(50,1)+1,random.randn(50,1)]))
    b = squeeze(transpose([0.1*random.randn(50,1)-1,random.randn(50,1)]))
    e = squeeze(transpose([0.1*random.randn(50,1),random.randn(50,1)]))
    
    c = zeros((50,2))
    c[:,0] = e[:,0]*cos(pi/4)-e[:,1]*sin(pi/4)
    c[:,1] = e[:,0]*sin(pi/4)+e[:,1]*cos(pi/4)
    
    d = squeeze(transpose([random.randn(50,1),0.1*random.randn(50,1)+1]))
    
    return a,b,c,d


a,b,c,d = makedataA1()
print a
print b
print c
print d