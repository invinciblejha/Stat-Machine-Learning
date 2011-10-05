# Make an array a of size 6 * 4 where every element is a 2.
from numpy import *

x = ones((6,4))
a = x+x
dot(x,transpose(matrix(x)))
print a

# Make an array b of size 6 * 4 that has 3 on the leading diagonal and 1 everywhere else.

x = ones((6,4))
b = x + eye(6,4)*2
print b

# Can you multiply these two matrices together? Why does a * b work, but not dot(a,b)?

# Yes, a * b multiples each position together. wheras dot(a,b) trys to multiple the vector components together
# and since we are trying to multiple a 6,4 by 6,4 and not 4,6 the vector multiplications fail.

# Compute dot(a.transpose(),b) and dot(a,b.transpose()). Why are the results different shapes?

# a 6,4 * 4,6 = 6,6     wheras 4,6 * 6,4 = 4,4 

# Write a function that prints some output on the screen and make sure you can run it in Eclipse

def someFunction ():
     print "the function works"
     
def randomArrayFunction ():
    a = random.randint(0.0, 50.0, 20)
    b = random.randint(0.0, 1.0, 20)
    c = random.randint(-50.0, 50.0, 20)
    print a
    print b
    print c
    print "mean of a is ",mean(a) 
    print "sum of a is  ",sum(a)
    print "mean of b is ",mean(b)
    print "sum of b is  ",sum(b)
    print "mean of c is ",mean(c)
    print "sum of c is  ",sum(c)
    
a = random.randint(0.0, 50.0, 20)
b = random.randint(0.0, 1.0, 20)
c = random.randint(-50.0, 50.0, 20)
print "the random array a = ",a
print "the random array b = ",b
print "the random array c = ",c
print "mean of a is ",mean(a) 
print "sum of a is  ",sum(a)
print "mean of b is ",mean(b)
print "sum of b is  ",sum(b)
print "mean of c is ",mean(c)
print "sum of c is  ",sum(c)

    


    
    