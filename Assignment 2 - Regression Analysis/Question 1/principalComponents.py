### SVD Ridge Regression ###

from numpy import *

class principalComponents:
    
    def __init__(self, principalComponents = 4):
        ## get data
        data = loadtxt('prostate_train.txt')
        data = data.reshape(-1,9)
        
        ## get alpha = 0.9 values of F distribution
        F90 = loadtxt('F90.txt')
        F90 = F90.reshape(-1,10)
        
        ## reformat data as X and Y
        p = 8
        data = data.reshape(-1,p+1)
        
        ## reformat data as X and Y
        y = data[:,p]
        X = data[:,0:p]
        y = transpose(matrix(y))
        X = matrix(X)
        
        covX = cov(transpose(X)) # need to transpose X to get a p x p covariance matrix
        sdX = sqrt(diag(covX)) # Note that sdX is an array        
        
        # standardize data to unit variance
        for i in range(p):
          X[:,i] = X[:,i]/sdX[i]
        
        ## center inputs
        meanX = mean(array(X), axis=0)
        Xc = X - ones(shape(y))*meanX
        
        ## SVD
        (U,D,V) = linalg.svd(Xc, full_matrices=0)
        #print 'D = ',D
        diagD = diag(D)
        #Xc_svd = U*diagD*transpose(V)
        #print 'V = ',V
        
        Z = Xc*V
        m = principalComponents  # 4 PCs
        Z4 = Z[:,0:m]  # use first m PCs
        Z41 = concatenate((ones((shape(Z4)[0],1)),Z4),axis=1) # add intercept term
        self.thetahat = self.linregp_ni(Z41,y)[0]        
        V4 = V[:,0:m]
        self.betahat_pcr = concatenate((self.thetahat[0],V4*self.thetahat[1:m+1,0]),axis=0)
        self.fitToModel(Xc, y, self.betahat_pcr)
        
    
    def fitToModel(self,Xc,y,beta):
        X1 = concatenate((ones((shape(Xc)[0],1)),Xc),axis=1)        
        self.yhat = X1*beta
        self.RSS = transpose(y - self.yhat)*(y - self.yhat)
        
    def principalReg(self, Xc, y):
        (U,D,V) = linalg.svd(Xc, full_matrices=0)        
        diagD = diag(D)        
        Z = Xc*V
        m = 4  # 4 PCs
        Z4 = Z[:,0:m]  # use first m PCs
        Z41 = concatenate((ones((shape(Z4)[0],1)),Z4),axis=1) # add intercept term
        self.thetahat = self.linregp_ni(Z41,y)[0]        
        V4 = V[:,0:m]
        betahat_pcr = concatenate((self.thetahat[0],V4*self.thetahat[1:m+1,0]),axis=0)
        return betahat_pcr
        
    
    def linregp(self,X,y):
        # takes N x p matrix X and a N x 1 vector y, and fits  
        # linear regression y = X*beta. NOTE indenting
        X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
        betahat = linalg.inv(transpose(X1)*X1)*transpose(X1)*y
        yhat = X1*betahat
        RSS = transpose(y - yhat)*(y - yhat)
        return betahat,yhat,RSS
    
    def linregp_ni(self,X,y):
        # takes N x p matrix X and a N x 1 vector y, and fits  
        # linear regression y = X*beta. NOTE indenting
        betahat = linalg.inv(transpose(X)*X)*transpose(X)*y
        yhat = X*betahat
        RSS = transpose(y - yhat)*(y - yhat)
        return betahat,yhat,RSS
    
    def ridgereg(self,X,y,lam):
        # takes N x p matrix X (no intercept column) and a N x 1 vector y, and fits  
        # ridge regression y = X*beta with lam*transpose(beta)*beta penalty term
        betahat0 = matrix(mean(array(y), axis=0))
        betahat = linalg.inv(transpose(X)*X + lam*eye(8))*transpose(X)*y
        #beta = dot(dot(linalg.inv(dot(transpose(X),X)+lam*eye(8)),transpose(X)),y)
        betahat = concatenate((betahat0,betahat),axis=0) 
        X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
        yhat = X1*betahat
        RSS = transpose(y - yhat)*(y - yhat) + lam*transpose(betahat)*betahat
        return betahat,yhat,RSS
    
    def getPrincipalComponents(self):
        
        
        return self.betahat_pcr, self.yhat, self.RSS
    
    


  

 




















