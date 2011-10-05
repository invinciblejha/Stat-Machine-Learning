from pylab import *
from numpy import *

class Regression:
    
    def __init__(self):
        ## get alpha = 0.9 values of F distribution
        self.F90 = loadtxt('F90.txt')
        self.F90 = self.F90.reshape(-1,10)
    
    def linregp(self, X, y):
        # takes N x p matrix X and a N x 1 vector y, and fits  
        # linear regression y = X*beta. NOTE indenting
        X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
        betahat = linalg.inv(transpose(X1)*X1)*transpose(X1)*y
        yhat = X1*betahat
        RSS = transpose(y - yhat)*(y - yhat)
        
        return betahat, yhat, RSS
    
    
    def linregp_ni(self, X, y):
        # takes N x p matrix X and a N x 1 vector y, and fits  
        # linear regression y = X*beta. NOTE indenting
        betahat = linalg.inv(transpose(X)*X)*transpose(X)*y
        yhat = X*betahat
        RSS = transpose(y - yhat)*(y - yhat)
        
        return betahat, yhat, RSS
    
    
    def ridgereg(self, X, y, lam):
        # takes N x p matrix X (no intercept column) and a N x 1 vector y, and fits  
        # ridge regression y = X*beta with lam*transpose(beta)*beta penalty term
        betahat = linalg.inv(transpose(X)*X + lam*eye(X.shape[1]))*transpose(X)*y
        # add intercept term
        betahat0 = matrix(mean(array(y), axis=0))
        betahat = concatenate((betahat0,betahat),axis=0) 
        X1 = concatenate((ones((shape(X)[0],1)),X),axis=1)
        # calculate fitted yhat
        yhat = X1*betahat
        RSS = transpose(y - yhat)*(y - yhat)  # this is not penalized
        #Note that Betahat has centering Bias
        return betahat, yhat, RSS
    
    
    def principalReg(self, Xc, y, m):
        (U,D,V) = linalg.svd(Xc, full_matrices=0)        
        diagD = diag(D)        
        Z = Xc*V    
        Z4 = Z[:,0:m]  # use first m PCs
        Z41 = concatenate((ones((shape(Z4)[0],1)),Z4),axis=1) # add intercept term
        thetahat = self.linregp_ni(Z41,y)[0]        
        V4 = V[:,0:m]
        betahat_pcr = concatenate((thetahat[0],V4*thetahat[1:m+1,0]),axis=0)
        #Calculate RSS and yhat
        X1 = concatenate((ones((shape(Xc)[0],1)),Xc),axis=1)        
        yhat = X1*betahat_pcr
        RSS = transpose(y - yhat)*(y - yhat)
        
        return betahat_pcr, yhat, RSS
    
    
    def ridgeRegressionTesting(self, X, y, Xtest, ytest, components):
        
        """Ridge regression testing to find the best lambda to use with ridge regression"""
        X = matrix(X)
        for index in range(1,70):            
            dflambda = trace(X*linalg.inv(transpose(X)*X+index*eye(components))*transpose(X))            
            
            plot(index,dflambda,'^')    
            dflambda = trace(Xtest*linalg.inv(transpose(Xtest)*Xtest+index*eye(components))*transpose(Xtest))            
            
            plot(index,dflambda,'*')             
            
        xlabel('Lambda Value')
        ylabel('Degrees of Freedom')    
        title("Lambda Testing for RidgeRegression")
        show()
        
    def principalComponentTesting(self, Xtrain, ytrain, X1test, ytest, meanX):
        
        """PCR regression testing to find the best number of principal 
        components for the training and test data"""
    
        for index in range(2,15):
            print 'Number of principal components:', index
            
            temppcr = self.principalReg(Xtrain, ytrain, index)
            betapcr = temppcr[0]
            yhatpcr = temppcr[1]
            RSSpcr = temppcr[2]
            
            print 'RSS prc = ', RSSpcr    
            plot(index,RSSpcr,'^')
            
            #Remove the intercepts centre bias
            betapcrnoi = betapcr[1:,:]
            betapcr2 = betapcr
            betapcr2[0] = betapcr2[0] - dot(meanX, betapcrnoi)
            yhatpcr2 = X1test*betapcr2
            RSSpcr2 = transpose(ytest - yhatpcr2)*(ytest - yhatpcr2)
            plot(index,RSSpcr2,'*')
    
        xlabel('Number of Principal Components')
        ylabel('RSS: (*) for Test RSS, (^) for Train RSS')    
        title("Principal Component Testing")
        show()
        
        
    def baggedModel(self, X, y, p, N):
    
        B = 100  # number of bootstrap samples
        baggedbeta = matrix(zeros((p+1,B)))
        for b in range(B):
            ytemp = matrix(zeros(shape(y)))  ## bootstrap samples
            Xtemp = matrix(zeros(shape(X)))
            for n in range(N):
                temp = random.randint(0,N,1)
                ytemp[n] = y[temp]   ## add selected to bootstrap sample
                Xtemp[n,:] = X[temp,:]
            bootbeta = self.linregp(Xtemp,ytemp)[0] ## fit to bootstrap samples
            baggedbeta[:,b] = bootbeta
        
        baggedbetahat = mean(baggedbeta, axis = 1) # take mean for bagged estimate
        
        return baggedbetahat
    
    def fillOutBeta(self, predictors, beta, noPredictors):
    
        """This method creates an array that can be used as a betahat
        it adds zeros to where we are not using predictors"""
        
        temp = []
        for idx in range(0,noPredictors):
            if predictors.count(idx) > 0:
                temp.append(float(beta[predictors.index(idx)]))
            else:
                temp.append(0.0)   
            
        return transpose(matrix(temp))
    
    def fstepreg(self, X, y, p):
        
        """Performs forward stepwise regression on the data, caution, modifies the 
        data while doing the regression to find the predictors"""
        
        oldbeta = mean(array(y))  #start with intercept
        oldX1 = ones(shape(y))  #intercept term
        oldRSS = transpose(y - oldX1*oldbeta)*(y - oldX1*oldbeta)
        inclpred = [-1]      # this is a LIST of included predictors
                             #included predictor = -1, the intercept
                             #--remember indexing starts at 0 in X
        exclpred = list(arange(p))
        continu = 1
        while continu:
           bestRSS = 100*oldRSS  #some large number
           for i in range(len(exclpred)): #try adding each predictor to current model
               j = exclpred[i]
               X1 = concatenate((oldX1,X[:,j:j+1]),axis=1) 
                    #add jth column of X to current predictor matrix
               beta = self.linregp_ni(X1,y)[0]  #fit lin reg model
               RSS = transpose(y - X1*beta)*(y - X1*beta)
               if RSS < bestRSS:  # best additional predictor so far
                    bestRSS = RSS   # --record the details
                    bestpred = j  
                    bestbeta = beta
           F = (oldRSS - bestRSS)/(bestRSS/(len(y) - len(bestbeta) - 1))
              #find the F-ratio for the best additional predictor
           if F > self.F90[len(y) - len(bestbeta) - 1,1]:  #significant?
                                                      #--update model
              oldX1 = concatenate((oldX1,X[:,bestpred:bestpred+1]),axis=1)
              oldbeta = bestbeta
              oldRSS = bestRSS
              inclpred.append(bestpred)
              exclpred.remove(bestpred)
           else:  #finished
              continu = 0 
           if len(exclpred) == 0:  #stop if all predictors included
              continu = 0
        return oldbeta,oldRSS,inclpred,exclpred
    
    def revstepwise(self, X, y, p):
        
        """Performs reverse stepwise regression on the data, caution, modifies the 
        data while doing the regression to find the predictors"""
        
        # start with full model
        oldX1 = X  
        print shape(X)
        print shape(y)
        oldbeta = self.linregp_ni(oldX1,y)[0]  
        oldRSS = self.linregp_ni(oldX1,y)[2]
        inclpred = list(arange(p+1))
        
        continu = 1
        while continu:
            worstRSS = 100*oldRSS  # some large number  
            for i in range(len(inclpred)):  #try deleting each predictor from current model
                X = concatenate((oldX1[:,0:i],oldX1[:,i+1:oldX1.shape[1]]),axis=1) #delete ith column of X 
                beta = self.linregp_ni(X,y)[0]  #fit linear regression model
                RSS = self.linregp_ni(X,y)[2]
                #print i,RSS
                if RSS < worstRSS:   # worst additional predictor so far
                    worstRSS = RSS   # record the details
                    worstpred = i  # note that this is the index in inclpred, not the predictor itself
                    worstbeta = beta
            #find the F-ratio for the worst additional predictor
            F = (worstRSS - oldRSS)/(oldRSS/(len(y) - len(worstbeta) - 2)) 
            #print 'worstpred=',worstpred,' worstRSS=',float(worstRSS),' F=',float(F),' critF=',self.F90[len(y) - len(worstbeta) -2,1]
            if F < self.F90[len(y) - len(worstbeta) -2,1]:  # not significant?
                ## update model
                oldX1 = concatenate((oldX1[:,0:worstpred],oldX1[:,worstpred+1:oldX1.shape[1]]),axis=1) 
                oldbeta = worstbeta
                oldRSS = worstRSS
                inclpred.pop(worstpred)
            else:  # finished
                continu = 0 
            if len(inclpred) == 0:  # stop if all predictors deleted
                    continu = 0
        print 'Final model =',inclpred