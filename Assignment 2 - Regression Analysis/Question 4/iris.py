
# Code from Chapter 9 of Machine Learning: An Algorithmic Perspective
# by Stephen Marsland (http://seat.massey.ac.nz/personal/s.r.marsland/MLBook.html)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008

# Examples of using the k-means and SOM algorithms on the Iris dataset

from pylab import *
from numpy import *
import operator

def removeDuplicates(x):

    d = {}
    for (a,b) in x:
        d[(a,b)] = (a,b) 
        
    return array(d.values())


def calculatePenalty(setA, setB, setC, fullset, x, y):
    
    
    penalty = 0
    nonClassified = x * y - len(fullset)
    penalty += (nonClassified * .75)/(x * y)
    
    setAB = len(setA) + len(setB)
    setAC = len(setA) + len(setC)
    setBC = len(setB) + len(setC)
    setABC = setAB = len(setC)
    
    setABpenalty = 0.0
    for (a,b) in setA:
        for (c,d) in setB:
            if (a,b) == (c,d):
                setABpenalty +=1.0
    
    penalty += setABpenalty/setAB
            
    setACpenalty = 0.0           
    for (a,b) in setA:
        for (c,d) in setC:
            if (a,b) == (c,d):
                setACpenalty +=1.0
    
    penalty += setACpenalty/setAC
    
    
    setBCpenalty = 0.0
    for (a,b) in setB:
        for (c,d) in setC:
            if (a,b) == (c,d):
                setBCpenalty += 1.0
                
    penalty += setBCpenalty/setBC
    
    
    setABCpenalty = 0.0            
    for (a,b) in setA:
        for (c,d) in setB:
            for (e,f) in setC:
                if (a,b) == (c,d) == (e,f):
                    setABCpenalty += 2.0            
    penalty += setBCpenalty/setABC
                
    return penalty
                
    
    
    
    
    
         

iris = loadtxt('iris_proc.data',delimiter=',')
iris[:,:4] = iris[:,:4]-iris[:,:4].mean(axis=0)
imax = concatenate((iris.max(axis=0)*ones((1,5)),iris.min(axis=0)*ones((1,5))),axis=0).max(axis=0)
iris[:,:4] = iris[:,:4]/imax[:4]

target = iris[:,4]



#print train.max(axis=0), train.min(axis=0)

#import kmeansnet
#net = kmeansnet.kmeans(3,train)
#net.kmeanstrain(train)
#cluster = net.kmeansfwd(test)
#print 1.*cluster
#print iris[3::4,4]

import som

lowestPenalty = 1000
bestSize = 0

for netsize in range(2,10):
    
    
    
    penaltySum = 0
    for i in range(10):
        
        order = range(shape(iris)[0])
        random.shuffle(order)
        iris = iris[order,:]
        target = target[order,:]
    
        train = iris[::2,0:4]
        traint = target[::2]
        valid = iris[1::4,0:4]
        validt = target[1::4]
        test = iris[3::4,0:4]
        testt = target[3::4]
        
        net = som.som(netsize, netsize, train)
        #print 'Network Size         :', netsize
        net.somtrain(train, 400)
        
        best = zeros(shape(train)[0], dtype=int)
        for i in range(shape(train)[0]):
            best[i], activation = net.somfwd(train[i, :])
        
        
        #plot(net.map[0, :], net.map[1, :], 'k.', ms=15)
        
        where = find(traint == 0)
        #plot(net.map[0, best[where]], net.map[1, best[where]], 'rs', ms=30)    
        #Find all the unique points that map the red squares in train data
        rs11 = net.map[0, best[where]]
        rs12 = net.map[1, best[where]]
        rs1 = [[rs11[index], rs12[index]] for index in range(len(rs11))]
        rs1 = removeDuplicates(rs1)
        #print "Red Squares Train    :", len(rs1)    
        
        
        where = find(traint == 1)
        #plot(net.map[0, best[where]], net.map[1, best[where]], 'gv', ms=30)    
        #Find all the unique points that map the green triagles in train data
        gv11 = net.map[0, best[where]]
        gv12 = net.map[1, best[where]]
        gv1 = [[gv11[index], gv12[index]] for index in range(len(gv11))]
        gv1 = removeDuplicates(gv1)
        #print "Green Triagles Train :", len(gv1)
        
        where = find(traint == 2)
        #plot(net.map[0, best[where]], net.map[1, best[where]], 'b^', ms=30)    
        #Find all the unique points that map the blue trangles in train data
        bv11 = net.map[0, best[where]]
        bv12 = net.map[1, best[where]]
        bv1 = [[bv11[index], bv12[index]] for index in range(len(bv11))]
        bv1 = removeDuplicates(bv1)
        #print "Blue Triagles Train  :", len(bv1)
        
        
        #Find all the all the points that have been clasified             
        all11 = concatenate((rs11,concatenate((gv11,bv11))))
        all12 = concatenate((rs12,concatenate((gv12,bv12))))
        all1 = [[all11[index], all12[index]] for index in range(len(all11))]
        all1 = removeDuplicates(all1)
        #print "All Clasified Train  :", len(all1)
        
        trainpenalty = calculatePenalty(rs1, gv1, bv1, all1, netsize, netsize)
        #print "Training Set Penalty :", trainpenalty
        
        #axis([-0.1, 1.1, -0.1, 1.1])
        #axis('off')
        #figure(2)
        
        best = zeros(shape(test)[0], dtype=int)
        for i in range(shape(test)[0]):
            best[i], activation = net.somfwd(test[i, :])
    
        
        
        #plot(net.map[0, :], net.map[1, :], 'k.', ms=15)
        
        where = find(testt == 0)
        #plot(net.map[0, best[where]], net.map[1, best[where]], 'rs', ms=30)    
        #Find all the unique points that map the red squares in train data
        rs21 = net.map[0, best[where]]
        rs22 = net.map[1, best[where]]
        rs2 = [[rs21[index], rs22[index]] for index in range(len(rs21))]
        rs2 = removeDuplicates(rs2)
        #print "Red Squares Test     :", len(rs2)    
        
        
        where = find(testt == 1)
        #plot(net.map[0, best[where]], net.map[1, best[where]], 'gv', ms=30)
        #Find all the unique points that map the green triagles in train data
        gv21 = net.map[0, best[where]]
        gv22 = net.map[1, best[where]]
        gv2 = [[gv21[index], gv22[index]] for index in range(len(gv21))]
        gv2 = removeDuplicates(gv2)
        #print "Green Triagles Test  :", len(gv2)
        
        
        where = find(testt == 2)
        #plot(net.map[0, best[where]], net.map[1, best[where]], 'b^', ms=30)
        #Find all the unique points that map the blue trangles in train data
        bv21 = net.map[0, best[where]]
        bv22 = net.map[1, best[where]]
        bv2 = [[bv21[index], bv22[index]] for index in range(len(bv21))]
        bv2 = removeDuplicates(bv2)
        #print "Blue Triagles Test   :", len(bv2)
        
        #Find all the all the points that have been clasified             
        all21 = concatenate((rs11,concatenate((gv21,bv21))))
        all22 = concatenate((rs12,concatenate((gv22,bv22))))
        all2 = [[all21[index], all22[index]] for index in range(len(all21))]
        all2 = removeDuplicates(all2)
        #print "All Clasified Test   :", len(all2)
        
        testpenalty = calculatePenalty(rs2, gv2, bv2, all2, netsize, netsize)
        #print "Test Set Penalty     :", testpenalty
        
        
        sumPenalty = testpenalty + trainpenalty
        
    sumPenalty = sumPenalty/10.
    print "Network Size :", netsize, " has average penalty of :", sumPenalty
    if sumPenalty < lowestPenalty:
        lowestPenalty = sumPenalty
        bestSize = netsize
    
    #axis([-0.1, 1.1, -0.1, 1.1])
    #axis('off')
    #show()

print "The Best Network Size for the data is :", bestSize


