import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from trees import *

from scipy import stats
from statistics import *

num_folds=10
def prepareData(trainSet):
    trainSet = trainSet.sample(random_state=18, frac=1).reset_index(drop=True)
    trainingSet = trainSet.sample(random_state=32, frac=0.5).reset_index(drop=True)

    fullTrainSetSize = trainingSet.shape[0]
    Foldsize = int(0.1 * fullTrainSetSize)
    fold1 = []
    for i in range(10):
        fold1.append(trainingSet.values[(i) * Foldsize:(i + 1) * Foldsize])
    return trainingSet, fold1


def stderr_avg(bg, num_trees, num_folds):
    bg_stderr = []
    bg_avgacc = []
    for num_tree in num_trees:
        sigma_l = stdev(bg[num_tree])
        bg_stderr.append(sigma_l / np.sqrt(num_folds))
        bg_avgacc.append(np.mean(bg[num_tree]))

    return bg_stderr, bg_avgacc


def run_model(trees, fold, trainingSet):
    columns = trainingSet.columns
    models = ["Bagging", "Random Forests"]
    num_trees = [10, 20, 40, 50]
    depth = 8
    bg, rf = {}, {}
    for model in models:
        for num_tree in num_trees:
            if (model == "Bagging"):
                bg[num_tree] = []
            else:
                rf[num_tree] = []
            for i in range(num_folds):
                testData = fold[i]
                folds = list(range(num_folds))
                del folds[i]
                trainData = np.array(fold[folds[0]])
                del folds[0]
                for j in folds:
                    trainData = np.vstack((trainData, np.array(fold[j])))
                trainData = np.array((trainData))
                trainData = trainData.reshape((-1, trainingSet.shape[1]))
                tdata, tstdata = pd.DataFrame(trainData, columns=columns), pd.DataFrame(testData, columns=columns)
                if (model == "Bagging"):
                    trainBagPreds, testBagPreds, trainAcc = bagging(tdata, tstdata, bagging_num=30, depth=depth)
                    test_acc = np.mean(testBagPreds == tstdata["decision"].values) * 100
                    bg[num_tree].append(test_acc)

                elif (model == "Random Forests"):  # Random Forests.
                    trainBagPreds, testBagPreds, trainAcc = randomForests(tdata, tstdata, bagging_num=30, depth=depth)
                    if (testBagPreds.shape[0] != tstdata[:, -1].reshape((-1, 1)).shape[0]):
                        print("TestBag : ", testBagPreds.shape)
                        print(tstdata[:, -1].reshape((-1, 1)).shape)
                        print("Size mismatch RF!!")
                        break
                    test_acc = np.mean(testBagPreds == tstdata["decision"].values) * 100
                    rf[depth].append(test_acc)
                else:
                    "Give a model..will yea??"
                print(model, depth, i, test_acc)


    bg_stderr, bg_avgacc = stderr_avg(bg, num_trees, num_folds)
    rf_stderr, rf_avgacc = stderr_avg(rf, num_trees, num_folds)


    plt.figure(figsize=(10, 5))
    plt.title("Depth of the Tree vs Testing Accuracy")


    plt.errorbar(num_trees, bg_avgacc, marker='o', yerr=bg_stderr)
    plt.errorbar(num_trees, rf_avgacc, marker='o', yerr=rf_stderr)

    plt.xlabel("Number of tree(s).")
    plt.ylabel("Test Accuracy.")
    plt.legend(["bt_test", "rf_test"])
    plt.savefig("num_tree.png")
    plt.show()

def main():
    print("nothing")
    trees = __import__('trees')
    trainSet = pd.read_csv("trainingSet.csv")
    trainingSet, fold = prepareData(trainSet)
    run_model(trees, fold, trainingSet)


if __name__ == "__main__":
    main()
