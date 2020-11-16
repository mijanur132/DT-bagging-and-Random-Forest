import pandas as pd
import numpy as np
import random
from statistics import *
from scipy import stats
import sys
from Node import *
from DecisionTree import *


#(i) decision tree
def decisionTree(trainingSet, target='decision', isRandomForrest=0, leaf_lim=50, depth=8):
    tree = DecisionTree(trainingSet, target, isRandomForrest, leaf_lim, depth)
    return tree

def build_tree(tree):
    tree.fit()
    return tree.root

def test_tree(tree,trainingSet,testSet):
    test_preds, test_acc = tree.test(testSet)
    train_preds, train_acc = tree.test(trainingSet)
    return train_preds,test_preds,train_acc,test_acc

## (ii) bagging()
def bagging(trainingSet = None, testSet = None, bagging_num = 30, depth = 8):
    trainBagPreds,testBagPreds,trainAcc = [],[],0
    for iteration in range(bagging_num):
        trainSet = trainingSet.sample(frac = 1, replace = True).reset_index(drop = True)
        dtree = decisionTree(trainSet, target = 'decision', isRandomForrest=0, leaf_lim = 50, depth = depth)  # Instantiate the decision tree.
        root =build_tree(dtree) # Train the decision tree.
        trainPreds,preds, train_acc,test_acc = test_tree(dtree,trainingSet,testSet) # Test this decision tree on the training set.
        trainAcc += train_acc
        trainBagPreds.append(trainPreds)
        testBagPreds.append(preds)
    trainBagPreds,testBagPreds,finalTrainPreds= np.array(trainBagPreds),np.array(testBagPreds),[]
    trainPredDf = pd.DataFrame(trainBagPreds)
    for i in range(trainBagPreds.shape[1]):
        finalTrainPreds.append( trainPredDf[i].value_counts().idxmax() )
    finalTestPreds,testPredDf = [],pd.DataFrame(testBagPreds)
    for i in range(testBagPreds.shape[1]):
        finalTestPreds.append( testPredDf[i].value_counts().idxmax() )
    return np.array(finalTrainPreds), np.array(finalTestPreds), (trainAcc / bagging_num)

def randomForests(trainingSet = None, testSet = None, bagging_num = 5, depth = 8):
    trainBagPreds = []
    testBagPreds = []
    trainAcc = 0
    for iteration in range(bagging_num):
        trainSet = trainingSet.sample(frac = 1, replace = True).reset_index(drop = True)
        dtree = decisionTree(trainSet, target = 'decision', isRandomForrest=1, leaf_lim = 50, depth = depth)
        root = build_tree(dtree)
        trainPreds,preds, train_acc,test_acc = test_tree( dtree,trainingSet,testSet)
        trainAcc += train_acc
        trainBagPreds.append(trainPreds)
        testBagPreds.append(preds)

    trainBagPreds = np.array(trainBagPreds)
    testBagPreds = np.array(testBagPreds)

    finalTrainPreds = []
    trainPredDf = pd.DataFrame(trainBagPreds)
    for i in range(trainBagPreds.shape[1]):
        finalTrainPreds.append( trainPredDf[i].value_counts().idxmax() )
    finalTestPreds = []
    testPredDf = pd.DataFrame(testBagPreds)
    for i in range(testBagPreds.shape[1]):
        finalTestPreds.append( testPredDf[i].value_counts().idxmax() )
    return np.array(finalTrainPreds), np.array(finalTestPreds), (1.0*trainAcc/bagging_num)

if __name__ == "__main__":
    trainFileName,testFileName,modelIdx = sys.argv[1],sys.argv[2],int(sys.argv[3])
    trainingSet,testSet = pd.read_csv(trainFileName), pd.read_csv(testFileName)
    bagPreds=[]
    if (modelIdx == 1):
        tree = decisionTree(trainingSet, target = 'decision', isRandomForrest=0, leaf_lim = 50, depth = 8)
        root = build_tree(tree)
        a,b,acc, test_acc = test_tree(tree,trainingSet, testSet)
        print('Training Accuracy DT: %.2f' % acc)
        print('Testing Accuracy DT: %.2f' % test_acc)

    elif (modelIdx == 2):

        trainBagPreds, testBagPreds, trainAcc = bagging(trainingSet, testSet, bagging_num = 30)
        print ("Training Accuracy BT: %.2f" % trainAcc)
        print ("Testing Accuracy BT: %.2f" %  (np.mean(testBagPreds == testSet["decision"].values)*100) )

    else:
        trainBagPreds, testBagPreds, trainAcc = randomForests(trainingSet, testSet, bagging_num = 30)
        print ("Training Accuracy RF: %.2f" % trainAcc)
        print ("Testing Accuracy RF: %.2f" % (np.mean(testBagPreds == testSet["decision"].values)*100) )
