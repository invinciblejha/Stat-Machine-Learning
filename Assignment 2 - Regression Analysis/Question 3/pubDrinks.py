from numpy import *
import dtree

tree = dtree.dtree()
drink,classes,features = tree.read_data('drinks.data')
#print drink
#print classes
#print features
t=tree.make_tree(drink,classes,features)
tree.printTree(t,' ')

#print tree.classifyAll(t,drink)

for i in range(len(drink)):
    tree.classify(t,drink[i])

print "\n---------------Pruned Tree------------------"    
prunedTree = tree.pruningAlgorithm(drink, classes, features, t, 1.1)

tree.printTree(prunedTree,' ')