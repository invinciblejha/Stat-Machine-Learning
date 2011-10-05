from numpy import *

data = loadtxt('Q2.data',delimiter=',')
p = 10


# Split into training, validation, and test sets
target = zeros((shape(data)[0],6));
indices = where(data[:,p]==1)
target[indices,0] = 1
indices = where(data[:,p]==2)
target[indices,1] = 1
indices = where(data[:,p]==3)
target[indices,2] = 1
indices = where(data[:,p]==5)
target[indices,3] = 1
indices = where(data[:,p]==6)
target[indices,4] = 1
indices = where(data[:,p]==7)
target[indices,5] = 1

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
import mlp
net = mlp.mlp(train,traint,5,outtype='softmax')
net.earlystopping(train,traint,valid,validt,0.1)
net.confmat(test,testt)