from pylab import *
from numpy import *

data = loadtxt('Q2.data',delimiter=',')
p = 10

data[:,:p] = data[:,:p]-data[:,:p].mean(axis=0)
imax = concatenate((data.max(axis=0)*ones((1,11)),data.min(axis=0)*ones((1,11))),axis=0).max(axis=0)
data[:,:p] = data[:,:p]/imax[:p]

target = data[:,p]


# Randomly order the data
order = range(shape(data)[0])
random.shuffle(order)
data = data[order,:]
target = target[order,:]

train = data[::2,0:p]
traint = target[::2]
valid = data[1::4,0:p]
validt = target[1::4]
test = data[3::4,0:p]
testt = target[3::4]

#print train.max(axis=0), train.min(axis=0)

# Train the network
import kmeansnet
net = kmeansnet.kmeans(6,train)
net.kmeanstrain(train)
cluster = net.kmeansfwd(test)
kprediction = 1.*cluster
actual = data[3::4,p]
correct = 0.
for i in range(len(actual)):
    if kprediction[i] == actual[i]:
        correct += 1.
    
print 'K-means percentage correct =', correct/len(actual)

import som
net = som.som(7,7,train)
net.somtrain(train,400)

best = zeros(shape(train)[0],dtype=int)
for i in range(shape(train)[0]):
    best[i],activation = net.somfwd(train[i,:])

plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = find(traint == 1)
plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
where = find(traint == 2)
plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
where = find(traint == 3)
plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
where = find(traint == 5)
plot(net.map[0,best[where]],net.map[1,best[where]],'r*',ms=30)
where = find(traint == 6)
plot(net.map[0,best[where]],net.map[1,best[where]],'gs',ms=30)
where = find(traint == 7)
plot(net.map[0,best[where]],net.map[1,best[where]],'b.',ms=30)
axis([-0.1,1.1,-0.1,1.1])
axis('off')
figure(2)

best = zeros(shape(test)[0],dtype=int)
for i in range(shape(test)[0]):
    best[i],activation = net.somfwd(test[i,:])

plot(net.map[0,:],net.map[1,:],'k.',ms=15)
where = find(testt == 1)
plot(net.map[0,best[where]],net.map[1,best[where]],'rs',ms=30)
where = find(testt == 2)
plot(net.map[0,best[where]],net.map[1,best[where]],'gv',ms=30)
where = find(testt == 3)
plot(net.map[0,best[where]],net.map[1,best[where]],'b^',ms=30)
where = find(testt == 5)
plot(net.map[0,best[where]],net.map[1,best[where]],'r*',ms=30)
where = find(testt == 6)
plot(net.map[0,best[where]],net.map[1,best[where]],'gs',ms=30)
where = find(testt == 7)
plot(net.map[0,best[where]],net.map[1,best[where]],'b.',ms=30)
axis([-0.1,1.1,-0.1,1.1])
axis('off')
show()