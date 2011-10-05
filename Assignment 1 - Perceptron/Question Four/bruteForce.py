from pylab import *
from numpy import *


def getWeightedBinary(binaryNumber):
    
    """Method to get a binary permutation in array form of the trainingData"""
    
    x = 9
    trainingSetY = []    
    for binaryWeight in binaryNumber:    
        weighted = int(binaryWeight) * x    
        x = x-1
        if weighted > 0:            
            trainingSetY.append(weighted-1)
    trainingSetY = trainingSetY[::-1] 
    return trainingSetY


def linregp(tempTrainingSet,trainingSetY):
    
    """takes N x p matrix tempTrainingSet and a N x 1 vector trainingSetY, and fits  
    linear regression trainingSetY = tempTrainingSet*beta. NOTE indenting""" 
    
    betahat = linalg.inv(transpose(tempTrainingSet)*tempTrainingSet)*transpose(tempTrainingSet)*trainingSetY
    yhat = tempTrainingSet*betahat
    RSS = transpose(trainingSetY - yhat)*(trainingSetY - yhat)
    return betahat,yhat,RSS


def linearRegressionWithBetahat(tempTrainingSet,trainingSetY,betahat):
    
    """The same as above, but allows a betahat matrix to be input"""
    
    yhat = tempTrainingSet*betahat
    RSS = transpose(trainingSetY - yhat)*(trainingSetY - yhat)
    return betahat,yhat,RSS


def printPredictorsRss():
    
    """Method for printing the predictors RSS"""
    
    print Predictors1[0]
    print Predictors2[0]
    print Predictors3[0]
    print Predictors4[0]
    print Predictors5[0]
    print Predictors6[0]
    print Predictors7[0]
    print Predictors8[0]
    print Predictors9[0]
    

def printPredictors():
    
    """Method for printing the best predictors"""
    
    print Predictors1
    print Predictors2
    print Predictors3
    print Predictors4
    print Predictors5
    print Predictors6
    print Predictors7
    print Predictors8
    print Predictors9
    

def fillOutBeta(predictors, beta):
    
    """This method creates an array that can be used as a betahat
    it adds zeros to where we are not using predictors"""
    
    temp = []
    for idx in range(0,9):
        if predictors.count(idx) > 0:
            temp.append(float(beta[predictors.index(idx)]))
        else:
            temp.append(0.0)   
        
    return temp


def standardiseData(tempTrainingSet, tempTestSet):
    
    """Method to standardise the trainingData around a specified mean and variance"""
    
    tempTestSet[:,:8] = tempTestSet[:,:8]-tempTrainingSet[:,:8].mean(axis=0)
    #Centralise the trainingData around 0
    tempTestSet[:,:8] = tempTestSet[:,:8]/tempTrainingSet[:,:8].var(axis=0)
    #Normalise the trainingData inbetween -1 and 1


def createBetaHats():
    
    """Creates the betahats for each set of predictors"""
    
    bhat1 = transpose(matrix(fillOutBeta(Predictors1[1], Predictors1[2])))
    bhat2 = transpose(matrix(fillOutBeta(Predictors2[1], Predictors2[2])))
    bhat3 = transpose(matrix(fillOutBeta(Predictors3[1], Predictors3[2])))
    bhat4 = transpose(matrix(fillOutBeta(Predictors4[1], Predictors4[2])))
    bhat5 = transpose(matrix(fillOutBeta(Predictors5[1], Predictors5[2])))
    bhat6 = transpose(matrix(fillOutBeta(Predictors6[1], Predictors6[2])))
    bhat7 = transpose(matrix(fillOutBeta(Predictors7[1], Predictors7[2])))
    bhat8 = transpose(matrix(fillOutBeta(Predictors8[1], Predictors8[2])))
    bhat9 = transpose(matrix(fillOutBeta(Predictors9[1], Predictors9[2])))
    
    return bhat1, bhat2, bhat3, bhat4, bhat5, bhat6, bhat7, bhat8, bhat9


def switchOnLength(length, rss, predictors, betahat):
    
    """Finds the amount of predictors and sets the it to the best if the
    Rss for that predictor set is the lowest"""
    
    if length == 1:
        if Predictors1[0] > rss:
            Predictors1[0] = rss
            Predictors1[1] = predictors  
            Predictors1[2] = betahat              
    if length == 2:
        if Predictors2[0] > rss:
            Predictors2[0] = rss
            Predictors2[1] = predictors 
            Predictors2[2] = betahat              
    if length == 3:
        if Predictors3[0] > rss:
            Predictors3[0] = rss
            Predictors3[1] = predictors
            Predictors3[2] = betahat   
    if length == 4:
        if Predictors4[0] > rss:
            Predictors4[0] = rss
            Predictors4[1] = predictors
            Predictors4[2] = betahat   
    if length == 5:
        if Predictors5[0] > rss:
            Predictors5[0] = rss
            Predictors5[1] = predictors
            Predictors5[2] = betahat   
    if length == 6:
        if Predictors6[0] > rss:
            Predictors6[0] = rss
            Predictors6[1] = predictors
            Predictors6[2] = betahat   
    if length == 7:
        if Predictors7[0] > rss:
            Predictors7[0] = rss
            Predictors7[1] = predictors
            Predictors7[2] = betahat   
    if length == 8:
        if Predictors8[0] > rss:
            Predictors8[0] = rss
            Predictors8[1] = predictors
            Predictors8[2] = betahat   
    if length == 9:
        if Predictors9[0] > rss:
            Predictors9[0] = rss
            Predictors9[1] = predictors 
            Predictors9[2] = betahat   
   
#Setup the predictors sets with high rss values to begin with
Predictors1 = [1000, [], []]
Predictors2 = [1000, [], []]
Predictors3 = [1000, [], []]
Predictors4 = [1000, [], []]
Predictors5 = [1000, [], []]
Predictors6 = [1000, [], []]
Predictors7 = [1000, [], []]
Predictors8 = [1000, [], []]
Predictors9 = [1000, [], []]
        

#Load the training trainingData and shape the matrix
trainingData = loadtxt('prostate_train.txt')
trainingData = trainingData.reshape(-1,9)
trainingSetY = trainingData[:,8]
trainingSetY = transpose(matrix(trainingSetY))

#Create the Predictors trainingData set and concatinate with intercept
tempTrainingSet = trainingData[:,0:8]
trainingSetX = concatenate((tempTrainingSet,ones((shape(tempTrainingSet)[0],1))),axis=1)

#Load the testing trainingData for validating the results of the training set
testData = loadtxt('prostate_test.txt')
testData = testData.reshape(-1,9)
testingSetY = testData[:,8]
testingSetY = transpose(matrix(testingSetY))

#Create the tempTrainingSet trainingData that will be used with the previously calculated beta's
tempTestSet = testData[:,0:8]
testingSetX = concatenate((tempTestSet,ones((shape(tempTestSet)[0],1))),axis=1)


#Standardise training set and testingset (testing set to be standardised with training set data)
standardiseData(trainingSetX, testingSetX)
standardiseData(trainingSetX, trainingSetX)

#Brute force the trainingData and find the best predictors for the training
#set for all permutations of predictors.
results = []
for index in range(512):
    #Skip for Zero
    if index == 0:
        continue
    # Permutate through the Predictors
    binaryNumber = '{0:009b}'.format(index)    
    dataSelection = getWeightedBinary(binaryNumber)    
 
    #Pull the selected trainingData out
    tempTrainingSet = trainingSetX[:,dataSelection]
    predictorNumber = len(dataSelection)
    tempTrainingSet = matrix(tempTrainingSet)    
    
    #Do regression and save the best results
    betahat,yhat,RSS = linregp(tempTrainingSet,trainingSetY)
    switchOnLength(predictorNumber, RSS[0,0], dataSelection, betahat)


#Set to print the best predictors
#printPredictors()


#Create the betaHats with zeros filled out for testing purposes
betaHats = createBetaHats()

print "Training RSS"
printPredictorsRss()

print "Testing RSS"

#For each set of predictors, find the rss with linear regression with the betahats.
for index in range(0,9):
    betahat,yhat,RSS = linearRegressionWithBetahat(testingSetX,testingSetY,betaHats[index])
    print float(RSS)





    
    



