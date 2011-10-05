from numpy import *

data = loadtxt('../Q2.data',delimiter=',')
p = 10


data[where(data[:,p] <= 4),p] = 1
data[where(data[:,p] >  4),p] = 0
target = zeros((shape(data)[0],1))
indices = where(data[:,p]==1)
target[indices,0] = 1


# Randomly order the data
order = range(shape(data)[0])
random.shuffle(order)
data = data[order,:]
target = target[order,:]

# Split into training, validation, and test sets
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