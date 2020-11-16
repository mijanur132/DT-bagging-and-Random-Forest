import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from trees import *
from plots import *
from scipy import stats
from statistics import *

num_fold=10
bagging_num=30

def prepareData(trainSet):
    trainingSet = trainSet.sample(random_state=18, frac=1).reset_index(drop=True)
    fullTrainSetSize = trainingSet.shape[0]
    Foldsize = int(0.1 * fullTrainSetSize)
    fold1 = []
    for i in range(num_fold):
        fold1.append(trainingSet.values[(i) * Foldsize:(i + 1) * Foldsize])
    return trainingSet, fold1

def stderr_avg(dt, num_fold):
    dt_stderr = []
    dt_avgacc = []
    for key, val in dt.items():
        print(key, val)
        sigma_l = stdev(dt[key])
        dt_stderr.append(sigma_l / np.sqrt(num_fold))
        dt_avgacc.append(np.mean(val))
    print(dt_stderr,dt_avgacc)
    return dt_stderr, dt_avgacc


def run_model(trees, fold, trainingSet):
    columns = trainingSet.columns
    models = ["Decision Tree", "Bagging", "Random Forests"]
    #models = ["Decision Tree"]
    depth = 8
    tfracs = [0.05, 0.075, 0.1, 0.15, 0.2]


    dt, bg, rf = {}, {}, {}
    for model in models:
        for tfrac in tfracs:
            if (model == "Decision Tree"):
                dt[tfrac] = []
            elif (model == "Bagging"):
                bg[tfrac] = []
            else:
                rf[tfrac] = []
            for i in range(num_fold):
                testData = fold[i]
                folds = list(range(num_fold))
                del folds[i]
                trainData = np.array(fold[folds[0]])
                del folds[0]
                for j in folds:
                    trainData = np.vstack((trainData, np.array(fold[j])))
                trainData = pd.DataFrame(trainData)
                trainData = trainData.sample(random_state=32, frac=tfrac).reset_index(drop=True).values
                #trainData = trainData.reshape((-1, trainingSet.shape[1]))
                tdata, tstdata = pd.DataFrame(trainData, columns=columns), pd.DataFrame(testData, columns=columns)

                if (model == "Decision Tree"):
                    dtree = decisionTree(tdata, target='decision', isRandomForrest=0, leaf_lim=50, depth=depth)
                    root = build_tree(dtree)
                    a, b, acc, test_acc = test_tree(dtree, tdata, tstdata)
                    # predss, tesst_acc = trees.test(dtree, pd.DataFrame(testData, columns=columns))
                    # print(test_acc)
                    dt[tfrac].append(test_acc)

                elif (model == "Bagging"):
                    trainBagPreds, testBagPreds, trainAcc = bagging(tdata, tstdata, bagging_num, depth=depth)
                    # test_acc = np.mean(testBagPreds == pd.DataFrame(testData, columns = columns)["decision"].values)
                    test_acc = np.mean(testBagPreds == tstdata["decision"].values) * 100
                    # print(test_acc)
                    bg[tfrac].append(test_acc)


                elif (model == "Random Forests"):  # Random Forests.
                    trainBagPreds, testBagPreds, trainAcc = randomForests(tdata, tstdata, bagging_num, depth=depth)
                    # test_acc = np.mean(testBagPreds == pd.DataFrame(testData, columns = columns)["decision"].values.reshape((-1,1)))
                    test_acc = np.mean(testBagPreds == tstdata["decision"].values) * 100
                    tesst_acc = np.mean(
                        testBagPreds == pd.DataFrame(testData, columns=columns)["decision"].values.reshape((-1, 1)))
                    print(test_acc, tesst_acc)
                    rf[tfrac].append(test_acc)

                else:
                    "Give a model..will yea??"
                print(model, depth, i, test_acc)

    dt_stderr, dt_avgacc = stderr_avg(dt, num_fold)
    bg_stderr, bg_avgacc = stderr_avg(bg, num_fold)
    rf_stderr, rf_avgacc = stderr_avg(rf, num_fold)

    cv_frac_plot(tfracs, dt_avgacc, dt_stderr, bg_avgacc, bg_stderr, rf_avgacc, rf_stderr)
    # plt.figure(figsize=(10, 5))
    # plt.title("Depth of the Tree vs Testing Accuracy")
    #
    # plt.errorbar(tfracs, dt_avgacc, marker='o', yerr=dt_stderr)
    # #plt.errorbar(tfracs, bg_avgacc, marker='o', yerr=bg_stderr)
    # #plt.errorbar(tfracs, rf_avgacc, marker='o', yerr=rf_stderr)
    #
    # plt.xlabel("Depth of the tree(s).")
    # plt.ylabel("Test Accuracy.")
    # plt.legend(["dt_test", "bt_test", "rf_test"])
    # plt.savefig("cv_frac.png")
    # plt.show()

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
    trainingSet, fold = prepareData(trainSet)
    run_model(trees, fold, trainingSet)


if __name__ == "__main__":
    main()
