import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy import stats
from statistics import *
trees = __import__('trees')


trainSet = pd.read_csv("trainingSet.csv")

columns = trainSet.columns

trainSet = trainSet.sample(random_state = 18, frac = 1).reset_index(drop = True)
trainingSet = trainSet.sample(random_state = 32, frac = 0.5).reset_index(drop = True)

trainSet.shape, trainingSet.shape

# 10% samples in each fold.
size = int(0.1 * trainingSet.shape[0])

# This is the variable that will keep track of all the data for the folds, for us. From this, we'll construct a 
# a hold-out test set.

start = 0
# Build up the fold array.
fold = []

while(start <= trainingSet.shape[0]-size):
    tempfold = np.array(trainingSet.values[start:start+size])
    fold.append(tempfold)
    start += size

num_folds = 10

models =  ["Bagging", "Random Forests"]

num_trees = [10, 20, 40, 50]
depth = 8

bg = {}

rf = {}

for model in models:

    # Vary over different number of trees.
    for num_tree in num_trees:
    
        if (model == "Bagging"):
            bg[num_tree] = []
        else:
            rf[num_tree] = []
            
        # Construction of the training and the testing sets.
        for i in range(num_folds):
            
            # Assume that fold[i] is going to be the test set.
            testData = fold[i]

            # Construction of the training and the testing data.
            if (i == 0):
                trainData = np.array(fold[i+1:]).reshape((-1,trainingSet.shape[1]))
            elif (i == num_folds - 1):
                trainData = np.array(fold[:i]).reshape((-1,trainingSet.shape[1]))
            else:
                trainData = np.vstack((np.array(fold[:i]), np.array(fold[i+1:]))).reshape((-1,trainingSet.shape[1]))

            if (model == "Bagging"):
                trainBagPreds, testBagPreds = trees.bagging(pd.DataFrame(trainData, columns = columns), pd.DataFrame(testData, columns = columns), bagging_num = num_tree, depth = depth)
                test_acc = np.mean(testBagPreds == pd.DataFrame(testData, columns = columns)["decision"].values)
                bg[num_tree].append(test_acc)

                pickle.dump( bg, open( "bg_numtree.pickle", "wb" ) )
                
            else: # Random Forests.
                trainBagPreds, testBagPreds = trees.randomForests(pd.DataFrame(trainData, columns = columns), pd.DataFrame(testData, columns = columns), bagging_num = num_tree, depth = depth)
                
                if (testBagPreds.shape[0] != testData[:,-1].reshape((-1,1)).shape[0]):
                    print ("TestBag : ", testBagPreds.shape)
                    print (testData[:,-1].reshape((-1,1)).shape)
                    print ("Size mismatch RF!!")
                    break
                    
                test_acc = np.mean(testBagPreds == pd.DataFrame(testData, columns = columns)["decision"].values.reshape((-1,1)))
                rf[num_tree].append(test_acc)
                
                pickle.dump( rf, open( "rf_numtree.pickle", "wb" ) )

# pickle.dump( bg, open( "bg_numtree.pickle", "wb" ) )

# pickle.dump( rf, open( "rf_numtree.pickle", "wb" ) )

bg = pickle.load( open( "bg_numtree.pickle", "rb" ) )

rf = pickle.load( open( "rf_numtree.pickle", "rb" ) )

for key, val in bg.items():
    for i in range(len(bg[key])):
        bg[key][i] = bg[key][i] * 100
        
bg_stderr = []
bg_avgacc = []

for num_tree in num_trees:
    sigma_l = stdev(bg[num_tree])
    bg_stderr.append(sigma_l / np.sqrt(num_folds))
    bg_avgacc.append(np.mean(bg[num_tree]))

for key, val in rf.items():
    for i in range(len(rf[key])):
        rf[key][i] = rf[key][i] * 100
        
rf_stderr = []
rf_avgacc = []

for num_tree in num_trees:
    sigma_l = stdev(rf[num_tree])
    rf_stderr.append(sigma_l / np.sqrt(num_folds))
    rf_avgacc.append(np.mean(rf[num_tree]))

plt.figure(figsize = (10,5))

plt.title("Number of Tree(s) vs Testing Accuracy")

plt.errorbar(num_trees, bg_avgacc, marker = 'o', yerr = bg_stderr)
plt.errorbar(num_trees, rf_avgacc, marker = 'o', yerr = rf_stderr)

plt.xlabel("Number of tree(s).")
plt.ylabel("Test Accuracy.")
plt.legend(["bt_test", "rf_test"])

plt.show()

##########. Hypothesis Testing #############


'''
        We could use a t-test to check if the difference in the mean accuracy between the two models is statistically significant, e.g. reject the null hypothesis that assumes that the two samples have the same distribution.
'''

# Paired t-test.

# Calculate the p-value associated with the models.
test_vals = stats.ttest_rel(bg_avgacc, rf_avgacc)

'''
    Null Hypothesis : Differences between the models is not significant.
    
    p-value : 4.45 e-08
    
    p-value < significance value (0.05)
    
    Reject the null hypothesis, meaning the differences between the models is actually significant.
'''

print ( test_vals )