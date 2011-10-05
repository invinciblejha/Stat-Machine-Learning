from numpy import *

def calc_entropy(p):
    if p!=0:
        return -p * math.log(p, 2)
    else:
        return 0
    
round = [['jim', .5, 0], ['jane', .25, 0], ['sarah', .125, 0], ['simon', .125, 0], ['john', 0.0, 0]]

for firstIdx in range(len(round)):
    
    name = round[firstIdx][0]
    percentage = round[firstIdx][1]    
    
    entropy = calc_entropy(percentage)
   
    percentage2 = 0
    for secondIdx in range(len(round)):
        if secondIdx == firstIdx:
            continue
        
        percentage2 += round[secondIdx][1]        
    
    entropy += calc_entropy(percentage2) 
    round[firstIdx][2] = entropy  
    gain = entropy - percentage*calc_entropy(percentage)
    print 'Entropy for', name, 'buying a round is', entropy, 'the information gain for knowing this is', gain, '\n'
    
avgquestions = 1.0*round[0][1]+2.0*round[1][1]+3.0*round[2][1]+4.0*round[3][1]
print "Average amount of questions to ask :",avgquestions
avgquestions = 1.0*round[0][1]+2.0*round[1][1]+3.0*round[2][1]+3.0*round[3][1]
print "But if simon and sarah are asked as a couple it would be :",avgquestions