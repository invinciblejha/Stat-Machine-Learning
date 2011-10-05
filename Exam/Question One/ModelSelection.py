from pylab import *
from numpy import *
from Regression import *

Reg = Regression()

"""Load in data and calculate the split ratio"""
data = loadtxt('Q1.data')
p = 13

"""Shuffle Data"""
data = data.reshape(-1,p+1)
order = range(shape(data)[0])
random.shuffle(order)
data = data[order,:]
split = int(len(data)*.66)

covX = cov(transpose(data)) 
sdX = sqrt(diag(covX))
for i in range(p+1):    
    data[:,i] = data[:,i]/sdX[i]
  

traindata = data[0:split,:] 
testdata = data[split:len(data),:]

"""Response splitting"""
ytrain = traindata[:,p] 
ytrain = transpose(matrix(ytrain))
N = len(ytrain) 
ytest = testdata[:,p] 
ytest = transpose(matrix(ytest))
Ntest = len(ytest)


"""Add Squared Terms"""  
#X = concatenate((X,pow(data[:,0:p],2)), axis = 1) 

"""divide into 2/3 training, 1/3 test"""
Xtrain = traindata[:,0:p] 
Xtrain = matrix(Xtrain)
Xtest = testdata[:,0:p] 
Xtest = matrix(Xtest)

"""Add intercept"""
X1train = concatenate((ones((shape(Xtrain)[0],1)),Xtrain),axis=1) 
X1test = concatenate((ones((shape(Xtest)[0],1)),Xtest),axis=1) 

Xred = concatenate((Xtrain[:,6],concatenate((Xtrain[:,8],concatenate((Xtrain[:,11],Xtrain[:,12:13]),axis=1)),axis=1)),axis=1)
Xred2 = concatenate((Xtest[:,6],concatenate((Xtest[:,8],concatenate((Xtest[:,11],Xtest[:,12:13]),axis=1)),axis=1)),axis=1)

XBackward = concatenate((Xtrain[:,0],concatenate((Xtrain[:,3],concatenate((Xtrain[:,5],Xtrain[:,10:13]),axis=1)),axis=1)),axis=1)
XBackward2 = concatenate((Xtest[:,0],concatenate((Xtest[:,3],concatenate((Xtest[:,5],Xtest[:,10:13]),axis=1)),axis=1)),axis=1)

XForward = concatenate((Xtrain[:,0],concatenate((Xtrain[:,3],concatenate((Xtrain[:,5],concatenate((Xtrain[:,7],Xtrain[:,10:13]),axis=1)),axis=1)),axis=1)),axis=1)
XForward2 = concatenate((Xtest[:,0],concatenate((Xtest[:,3],concatenate((Xtest[:,5],concatenate((Xtest[:,7],Xtest[:,10:13]),axis=1)),axis=1)),axis=1)),axis=1)
"""Center the inputs"""
meanX = mean(array(Xtrain), axis=0)
Xctrain = Xtrain - ones(shape(ytrain))*meanX 



"""Calculating Z Values"""
temp = Reg.linregp(Xtrain, ytrain)
betahat = temp[0]
yhat = temp[1]  

"""Estimate covariance matrix"""
sigmahat2 = float((1.0/(len(ytrain) - len(betahat)))*transpose(yhat - ytrain)*(yhat - ytrain))
varbeta = linalg.inv(transpose(X1train)*X1train)*sigmahat2
z = multiply(betahat,1/transpose(matrix(diag(sqrt(varbeta)))))

print 'sigmahat2 = ',sigmahat2
print 'beta, z = ',concatenate((betahat,z),axis=1)


"""Full model, fit to training, then fit full 
    training model, to test data"""

betafull, yhatfull, RSSfull = Reg.linregp(Xtrain, ytrain)
yhatfull2 = X1test*betafull
RSSfull2 = transpose(ytest - yhatfull2)*(ytest - yhatfull2)
print 'beta (full) =',betafull


"""Reduced model, fit to training, then fit reduced 
    training model, to test data"""
    
betared, yhatred, RSSred = Reg.linregp_ni(Xred,ytrain)
yhatred2 = Xred2*betared
RSSred2 = transpose(ytest - yhatred2)*(ytest - yhatred2)
print 'beta (reduced) =',betared

"""Calculate the bagged model betahat."""

betabagged = Reg.baggedModel(Xtrain,ytrain, p, N)
yhatbag = X1train*betabagged
yhatbagged = X1test*betabagged
RSSbag = transpose(ytrain - yhatbag)*(ytrain - yhatbag)
RSSbagged = transpose(ytest - yhatbagged)*(ytest - yhatbagged)
print 'beta (bagged) =', betabagged


"""Principal Components Regression Model, fit to training, then fit pcr
    training model, to test data"""  
#Reg.principalComponentTesting(Xctrain, ytrain, X1test, ytest, meanX)  
pcrno = 10  
betapcr, yhatpcr, RSSpcr = Reg.principalReg(Xctrain,ytrain,pcrno)
betapcrnoi = betapcr[1:,:]
betapcr2 = betapcr
betapcr2[0] = betapcr2[0] - dot(meanX, betapcrnoi)
yhatpcr2 = X1test*betapcr2
RSSpcr2 = transpose(ytest - yhatpcr2)*(ytest - yhatpcr2)   
print 'beta (pcr) =',betapcr 


"""Ridge Regression Model, fit to training, then fit ridge regression
    training model, to test data"""    

#Reg.ridgeRegressionTesting(Xtrain, ytrain, Xtest, ytest, 13)
ridgelambda = 32
betaridge, yhatridge, RSSridge = Reg.ridgereg(Xctrain,ytrain,ridgelambda)
#Remove the centering bais in the ridge models beta hat

betaridgenoi = betaridge[1:,:]
betaridge2 = betaridge
betaridge2[0] = betaridge2[0] - dot(meanX, betaridgenoi)
yhatridge2 = X1test*betaridge2
RSSridge2 = transpose(ytest - yhatridge2)*(ytest - yhatridge2)
#Calculate Lambda
dflambda = trace(Xctrain*linalg.inv(transpose(Xctrain)*Xctrain+ridgelambda*eye(13))*transpose(Xctrain))


"""Backward Stepwise Regression Model, fit to training, then fit backward 
    training model, to test data"""
#Reg.revstepwise(X1train[0:100,:], ytrain[0:100,:], 13)    
betaback, yhatback, RSSback = Reg.linregp_ni(XBackward,ytrain)
yhatback2 = XBackward2*betaback
RSSback2 = transpose(ytest - yhatback2)*(ytest - yhatback2)
fullbetaback = Reg.fillOutBeta([0,3,5,10,11,12],betaback, 14)
print 'beta (backward) =',fullbetaback


"""Forward Stepwise Regression Model, fit to training, then fit backward 
    training model, to test data"""
#print Reg.fstepreg(Xtrain[0:100,:], ytrain[0:100,:], 13)    
betafor, yhatfor, RSSfor = Reg.linregp_ni(XForward,ytrain)
yhatfor2 = XForward2*betafor
RSSfor2 = transpose(ytest - yhatfor2)*(ytest - yhatfor2)
fullbetafor = Reg.fillOutBeta([0,3,5,7,10,11,12],betafor, 14)
print 'beta (backward) =',fullbetaback



"""Sigma2 from full (low-bias) model"""
sigma2 = (std(yhatfull - ytrain))**2
#print 'sigma2 =',sigma2

""" Calculate BIC """

print ' '
bicfull =   N/sigma2 *   ((RSSfull/N)   + math.log(N) * (len(betafull)  * sigma2/N))
bicpcr  =   N/sigma2 *   ((RSSpcr/N)    + math.log(N) * (pcrno   * sigma2/N))
bicback =   N/sigma2 *   ((RSSback/N)   + math.log(N) * (len(betaback)  * sigma2/N))
bicfor  =   N/sigma2 *   ((RSSfor/N)  + math.log(N) * (len(betafor) * sigma2/N))
bicridge =  N/sigma2 *   ((RSSridge/N)  + math.log(N) * (dflambda * sigma2/N))
print 'BIC (full | pcr | backward | forward | ridge) =', bicfull, bicpcr, bicback, bicfor, bicridge


"""Posterior probabilities"""

ebicfull  = math.exp(-(.5 * bicfull))
ebicpcr   = math.exp(-(.5 * bicpcr))
ebicback  = math.exp(-(.5 * bicback))
ebicfor = math.exp(-(.5 * bicfor))
ebicridge = math.exp(-(.5 * bicridge))
sumebic = ebicfull + ebicpcr + ebicback + ebicfor + ebicridge

print ' '
print 'Full Model Posterior probability                 ', int(ebicfull/sumebic*100),"%"
print 'PCR Model Posterior probability                  ', int(ebicpcr/sumebic*100),"%"
print 'Backwards Regression Model Posterior probability ', int(ebicback/sumebic*100),"%"
print 'Forwards Regression Model Posterior probability  ', int(ebicfor/sumebic*100),"%"
print 'Ridge Regression Model Posterior probability     ', int(ebicridge/sumebic*100),"%"
print ' '

## Training Optimism
opfull  = 2.0*len(betafull)  *sigma2/len(ytrain)
oppcr   = 2.0*len(betapcr)   *sigma2/len(ytrain)
opback  = 2.0*len(betaback)  *sigma2/len(ytrain)
opfor   = 2.0*len(betafor)   *sigma2/len(ytrain)
opridge = 2.0*len(betaridge) *sigma2/len(ytrain)

## Training error
errfull  = RSSfull  / len(ytrain)
errpcr   = RSSpcr   / len(ytrain)
errback  = RSSback  / len(ytrain)
errfor   = RSSfor   / len(ytrain)
errridge = RSSridge / len(ytrain)

## In sample error - Training
Errfull   = errfull  + opfull
Errpcr    = errpcr   + oppcr
Errback   = errback  + opback
Errfor    = errfor   + opfor
Errridge  = errridge + opridge


print '(error | optimism | total) (full)       =',errfull, opfull, Errfull
print '(error | optimism | total) (reduced)    =',errpcr, oppcr, Errpcr
print '(error | optimism | total) (backward)   =',errback, opback, Errback
print '(error | optimism | total) (brute)      =',errfor, opfor, Errfor
print '(error | optimism | total) (ridge)      =',errridge, opridge, Errridge

avgbeta = array   
avgbeta = betafull * (ebicfull / sumebic)
avgbeta = avgbeta + betapcr * (ebicpcr / sumebic)
avgbeta = avgbeta + fullbetaback * (ebicback / sumebic)
avgbeta = avgbeta + fullbetafor * (ebicfor / sumebic)
avgbeta = avgbeta + betaridge2 * (ebicridge / sumebic)

yhatavg = X1test*avgbeta
yhatavg1 = X1train*avgbeta
RSS2avg = transpose(ytest - yhatavg)*(ytest - yhatavg)
RSSavg = transpose(ytrain - yhatavg1)*(ytrain - yhatavg1)

#print "Average Beta = ", avgbeta



print ' '
print 'Full model     : RSS(train), RSS(test) =',RSSfull,RSSfull2
print 'Reduced model  : RSS(train), RSS(test) =',RSSred,RSSred2
print 'PCR model      : RSS(train), RSS(test) =',RSSpcr,RSSpcr2
print 'Ridge model    : RSS(train), RSS(test) =',RSSridge,RSSridge2
print 'Backward model : RSS(train), RSS(test) =',RSSback,RSSback2
print 'Forward model  : RSS(train), RSS(test) =',RSSfor,RSSfor2
print 'Bagged model   : RSS(train), RSS(test) =',RSSbag,RSSbagged
print "Average Model  : RSS(train), RSS(test) =",RSSavg,RSS2avg