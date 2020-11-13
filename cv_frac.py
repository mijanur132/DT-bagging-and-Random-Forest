import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy import stats
from statistics import *

trees = __import__('trees')

trainSet = pd.read_csv("trainingSet.csv")

columns = trainSet.columns

trainingSet = trainSet.sample(random_state = 18, frac = 1).reset_index(drop = True)

trainingSet.shape

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

models =  ["Decision Tree", "Bagging", "Random Forests"]

tfracs = [0.05, 0.075, 0.1, 0.15, 0.2]

depth = 8

dt = {}

bg = {}

rf = {}

for model in models:

    # Construction of the training and the testing sets.
    for i in range(num_folds):

        if (model == "Decision Tree"):
            dt[i] = []
        elif (model == "Bagging"):
            bg[i] = []
        else:
            rf[i] = []

        for tfrac in tfracs:
            
            # Assume that fold[i] is going to be the test set.
            testData = fold[i]

            # Construction of the training and the testing data.
            if (i == 0):
                trainData = np.array(fold[i+1:]).reshape((-1,trainingSet.shape[1]))
            elif (i == num_folds - 1):
                trainData = np.array(fold[:i]).reshape((-1,trainingSet.shape[1]))
            else:
                trainData = np.vstack((np.array(fold[:i]), np.array(fold[i+1:]))).reshape((-1,trainingSet.shape[1]))

            # Now that the training data has been constructed, we need to sample only tfrac% of it.
            trainData = pd.DataFrame(trainData)
            trainData = trainData.sample(random_state = 32, frac = tfrac).reset_index(drop = True).values
            
            if (model == "Decision Tree"):
                # Instantiate the decision tree.
                dtree = trees.DecisionTree(pd.DataFrame(trainData, columns = columns), target = 'decision', leaf_lim = 50, depth = depth)

                # Train the decision tree.
                root = dtree.fit()

                # Test this decision tree.
                preds, test_acc = trees.test(dtree, pd.DataFrame(testData, columns = columns))
                dt[i].append(test_acc)
                pickle.dump( dt, open( "dt_tfrac.pickle", "wb" ) )

            elif (model == "Bagging"):
                trainBagPreds, testBagPreds = trees.bagging(pd.DataFrame(trainData, columns = columns), pd.DataFrame(testData, columns = columns), bagging_num = 30, depth = depth)
                test_acc = np.mean(testBagPreds == pd.DataFrame(testData, columns = columns)["decision"].values)
                bg[i].append(test_acc)

                pickle.dump( bg, open( "bg_tfrac.pickle", "wb" ) )

            else: # Random Forests.
                trainBagPreds, testBagPreds = trees.randomForests(pd.DataFrame(trainData, columns = columns), pd.DataFrame(testData, columns = columns), bagging_num = 30, depth = depth)

                test_acc = np.mean(testBagPreds == pd.DataFrame(testData, columns = columns)["decision"].values.reshape((-1,1)))
                print (test_acc)
                rf[i].append(test_acc)

                pickle.dump( rf, open( "rf_tfrac.pickle", "wb" ) )

# pickle.dump( dt, open( "dt_tfrac.pickle", "wb" ) )

# pickle.dump( bg, open( "bg_tfrac.pickle", "wb" ) )

# pickle.dump( rf, open( "rf_tfrac.pickle", "wb" ) )



dt = pickle.load( open( "dt_tfrac.pickle", "rb" ) )

bg = pickle.load( open( "bg_tfrac.pickle", "rb" ) )

rf = pickle.load( open( "rf_tfrac.pickle", "rb" ) )


vals = []
for key, val in dt.items():
    vals.append(val)

dtVals = np.mean( np.array(vals), axis = 0 )  

vals = []
for key, val in bg.items():
    vals.append(val)

bgVals = np.mean( np.array(vals), axis = 0 )  
bgVals *= 100

vals = []
for key, val in rf.items():
    vals.append(val)

rfVals = np.mean( np.array(vals), axis = 0 )  
rfVals *= 100

rf_stderr = []
sigma_l = stdev(rfVals)

for i in range(rfVals.shape[0]):
    rf_stderr.append(sigma_l / np.sqrt(num_folds))

bg_stderr = []
sigma_l = stdev(bgVals)

for i in range(bgVals.shape[0]):
    bg_stderr.append(sigma_l / np.sqrt(num_folds))

dt_stderr = []
sigma_l = stdev(dtVals)

for i in range(dtVals.shape[0]):
    dt_stderr.append(sigma_l / np.sqrt(num_folds))

plt.figure(figsize = (10,5))

plt.title("Training Fraction of the training data vs Testing Accuracy")

plt.errorbar(tfracs, dtVals, marker = 'o', yerr = dt_stderr)
plt.errorbar(tfracs, bgVals, marker = 'o', yerr = bg_stderr)
plt.errorbar(tfracs, rfVals, marker = 'o', yerr = rf_stderr)

plt.xlabel("Training fraction of the data.")
plt.ylabel("Test Accuracy.")
plt.legend(["dt_test", "bt_test", "rf_test"])

plt.show()

##########. Hypothesis Testing #############


'''
        We could use a t-test to check if the difference in the mean accuracy between the two models is statistically significant, e.g. reject the null hypothesis that assumes that the two samples have the same distribution.
'''

# Paired t-test.

# Calculate the p-value associated with the models.
test_vals = stats.ttest_rel(dtVals, rfVals*100)

'''
    Null Hypothesis : Differences between the models is not significant.
    
    p-value : 4.45 e-08
    
    p-value < significance value (0.05)
    
    Reject the null hypothesis, meaning the differences between the models is actually significant.
'''

print (stats.ttest_rel(rfVals, dtVals) )