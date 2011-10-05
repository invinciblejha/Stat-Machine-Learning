from pylab import *
from numpy import *


"""Load in data and calculate the split ratio"""
data = loadtxt('Q1.data')
p = 13


for i in range(p):  
  data[:,i] = data[:,i]-data[:,i].mean()
  data[:,i] = data[:,i]/data[:,i].max()
  
data[where(data[:,p]<=5),p] = 1
data[where((data[:,p]>5) & (data[:,p]<=10)),p] = 2
data[where((data[:,p]>10) & (data[:,p]<=15)),p] = 3
data[where((data[:,p]>15) & (data[:,p]<=20)),p] = 4
data[where((data[:,p]>20) & (data[:,p]<=25)),p] = 5
data[where((data[:,p]>25) & (data[:,p]<=30)),p] = 6
data[where((data[:,p]>30) & (data[:,p]<=35)),p] = 7
data[where((data[:,p]>35) & (data[:,p]<=40)),p] = 8
data[where((data[:,p]>40) & (data[:,p]<=45)),p] = 9
data[where(data[:,p]>45),p] = 10
  


# Split into training, validation, and test sets
target = zeros((shape(data)[0],10));
indices = where(data[:,p]==0) 
target[indices,0] = 1
indices = where(data[:,p]==1)
target[indices,1] = 1
indices = where(data[:,p]==2)
target[indices,2] = 1
indices = where(data[:,p]==4) 
target[indices,3] = 1
indices = where(data[:,p]==5)
target[indices,4] = 1
indices = where(data[:,p]==6)
target[indices,5] = 1
indices = where(data[:,p]==7)
target[indices,6] = 1
indices = where(data[:,p]==8)
target[indices,7] = 1
indices = where(data[:,p]==9)
target[indices,8] = 1
indices = where(data[:,p]==10)
target[indices,9] = 1

# Randomly order the data
order = range(shape(data)[0])
random.shuffle(order)
data = data[order,:]
target = target[order,:]

train = concatenate((data[::2,0:4],data[::2,5:p]),axis=1)
traint = target[::2]
valid = concatenate((data[1::4,0:4],data[1::4,5:p]),axis=1)
validt = target[1::4]
test = concatenate((data[3::4,0:4],data[3::4,5:p]),axis=1)
testt = target[3::4]

import mlp
net = mlp.mlp(train,traint,3,outtype='linear')
net.earlystopping(train,traint,valid,validt,0.25)
#net.mlptrain(traindata,traintargets,0.25,1000)
#testdata = concatenate((testdata,-ones((shape(testdata)[0],1))),axis=1)
#testout = net.mlpfwd(testdata)
net.confmat(test,testt)

test = concatenate((test,-ones((shape(test)[0],1))),axis=1)
testout = net.mlpfwd(test)

#figure()
#plot(arange(shape(test)[0]),testout,'.')
#plot(arange(shape(test)[0]),testt,'x')
#legend(('Predictions','Targets'))
#print 0.5*sum((testt-testout)**2)
#show()


