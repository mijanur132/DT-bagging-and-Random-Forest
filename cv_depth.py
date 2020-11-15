import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

from scipy import stats
from statistics import *


def prepareData(trainSet):
    trainSet = trainSet.sample(random_state = 18, frac = 1).reset_index(drop = True)
    trainingSet = trainSet.sample(random_state = 32, frac = 0.5).reset_index(drop = True)
    fullTrainSetSize=trainingSet.shape[0]
    Foldsize = int(0.1 * fullTrainSetSize)
    start = 0
    fold = []
    fold1 = []
    # while(start <= fullTrainSetSize-Foldsize):
    #     tempfold = np.array(trainingSet.values[start:start+Foldsize])
    #     fold.append(tempfold)
    #     start += Foldsize
    for i in range(10):
         fold1.append(trainingSet.values[(i)*Foldsize:(i+1)*Foldsize])
    return trainingSet,fold1

def run_model(trees,fold,trainingSet):
    columns = trainingSet.columns
    num_folds = 10
    models =  ["Decision Tree","Bagging","Random Forests"]
    depths = [3, 5, 7, 9]
    dt,bg,rf = {},{},{}
    for model in models:   #for each model
        # Vary over different tree depths.
        for depth in depths:
            if (model == "Decision Tree"):
                dt[depth] = []
            elif (model == "Bagging"):
                bg[depth] = []
            else:
                rf[depth] = []
            trainData=np.array
            for i in range(num_folds):
                testData = fold[i]
                folds=list(range(num_folds))
                del folds[i]
                trainData=np.array(fold[folds[0]])
                del folds[0]
                for j in folds:
                    trainData = np.vstack((trainData, np.array(fold[j])))
                trainData=np.array((trainData))
                trainData=trainData.reshape((-1,trainingSet.shape[1]))
                if (model == "Decision Tree"):
                    dtree = trees.DecisionTree(pd.DataFrame(trainData, columns = columns), target = 'decision',isRandomForrest=0, leaf_lim = 50, depth = depth)
                    root = dtree.fit()
                    preds, test_acc = dtree.test(pd.DataFrame(testData, columns = columns))
                    print(test_acc)
                    dt[depth].append(test_acc)

                elif (model == "Bagging"):
                    trainBagPreds, testBagPreds = trees.bagging(pd.DataFrame(trainData, columns = columns), pd.DataFrame(testData, columns = columns), bagging_num = 30, depth = depth)
                    test_acc = np.mean(testBagPreds == pd.DataFrame(testData, columns = columns)["decision"].values)
                    bg[depth].append(test_acc)

                    pickle.dump( bg, open( "bg_depth.pickle", "wb" ) )

                elif(model=="Random Forests"): # Random Forests.
                    trainBagPreds, testBagPreds = trees.randomForests(pd.DataFrame(trainData, columns = columns), pd.DataFrame(testData, columns = columns), bagging_num = 30, depth = depth)

                    test_acc = np.mean(testBagPreds == pd.DataFrame(testData, columns = columns)["decision"].values.reshape((-1,1)))
                    print (test_acc)
                    rf[depth].append(test_acc)

                    pickle.dump( rf, open( "rf_depth.pickle", "wb" ) )
                else:
                    "Give a model..will yea??"



    dt = pickle.load( open( "Colab Pickles/Q3/dt.pickle", "rb" ) )

    bg = pickle.load( open( "Colab Pickles/Q3/bg.pickle", "rb" ) )

    rf = pickle.load( open( "Colab Pickles/Q3/rf.pickle", "rb" ) )

    dt_stderr = []
    dt_avgacc = []

    for depth in depths:
        sigma_l = stdev(dt[depth])
        dt_stderr.append(sigma_l / np.sqrt(num_folds))
        dt_avgacc.append(np.mean(dt[depth]))

    for key, val in bg.items():
        for i in range(len(bg[key])):
            bg[key][i] = bg[key][i] * 100

    bg_stderr = []
    bg_avgacc = []

    for num_tree in depths:
        sigma_l = stdev(bg[num_tree])
        bg_stderr.append(sigma_l / np.sqrt(num_folds))
        bg_avgacc.append(np.mean(bg[num_tree]))

    for key, val in rf.items():
        for i in range(len(rf[key])):
            rf[key][i] = rf[key][i] * 100

    rf_stderr = []
    rf_avgacc = []

    for num_tree in depths:
        sigma_l = stdev(rf[num_tree])
        rf_stderr.append(sigma_l / np.sqrt(num_folds))
        rf_avgacc.append(np.mean(rf[num_tree]))


# Plot the Figure.
    plt.figure(figsize = (10,5))

    plt.title("Depth of the Tree vs Testing Accuracy")

    plt.errorbar(depths, dt_avgacc, marker = 'o', yerr = dt_stderr)
    plt.errorbar(depths, bg_avgacc, marker = 'o', yerr = bg_stderr)
    plt.errorbar(depths, rf_avgacc, marker = 'o', yerr = rf_stderr)

    plt.xlabel("Depth of the tree(s).")
    plt.ylabel("Test Accuracy.")
    plt.legend(["dt_test", "bt_test", "rf_test"])

    plt.show()

    ##########. Hypothesis Testing #############


    '''
            We could use a t-test to check if the difference in the mean accuracy between the two models is statistically significant, e.g. reject the null hypothesis that assumes that the two samples have the same distribution.
    '''

    # Paired t-test.

    # Calculate the p-value associated with the models.
    test_vals = stats.ttest_rel(dt_avgacc, bg_avgacc)
    print(stats.ttest_rel(dt_avgacc, bg_avgacc))
'''
    Null Hypothesis : Differences between the models is not significant.
    
    p-value : 4.45 e-08
    
    p-value < significance value (0.05)
    
    Reject the null hypothesis, meaning the differences between the models is actually significant.
'''



# stats.ttest_rel(rf_avgacc, bg_avgacc)

# stats.ttest_rel(dt_avgacc, rf_avgacc)
def main():
    print("nothing")
    trees = __import__('trees')
    trainSet = pd.read_csv("trainingSet.csv")
    trainingSet,fold=prepareData(trainSet)
    run_model(trees,fold,trainingSet)


if __name__ == "__main__":
    main()
