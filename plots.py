import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def cv_depth_plot(depths,dt_avgacc,dt_stderr,bg_avgacc,bg_stderr,rf_avgacc,rf_stderr):
    plt.figure(figsize=(10, 5))
    plt.title("Tree Depth vs Testing Accuracy")

    plt.errorbar(depths, dt_avgacc, marker='x',color="red", yerr=dt_stderr)
    plt.errorbar(depths, bg_avgacc, marker = 'o',color="green", yerr = bg_stderr)
    plt.errorbar(depths, rf_avgacc, marker = '+', color="yellow", yerr = rf_stderr)

    plt.xlabel("Tree Depth.")
    plt.ylabel("Test Accuracy.")
    plt.legend(["DT", "Bagging", "RF"])
    plt.savefig("cv_depth.png")
    plt.show()

def cv_frac_plot(tfracs, dt_avgacc, dt_stderr, bg_avgacc, bg_stderr, rf_avgacc, rf_stderr):
    plt.figure(figsize=(10, 5))
    plt.title("Fraction of the Training Data vs Testing Accuracy")

    plt.errorbar(tfracs, dt_avgacc, marker='o', yerr=dt_stderr)
    plt.errorbar(tfracs, bg_avgacc, marker='o', yerr=bg_stderr)
    plt.errorbar(tfracs, rf_avgacc, marker='o', yerr=rf_stderr)

    plt.xlabel("Fraction.")
    plt.ylabel("Test Accuracy.")
    plt.legend(["DT", "Bagging", "RF"])
    plt.savefig("cv_frac.png")
    plt.show()

def cv_num_tree_plot(num_trees, bg_avgacc, bg_stderr, rf_avgacc, rf_stderr):
    plt.figure(figsize=(10, 5))
    plt.title("Number of Trees vs Testing Accuracy")


    plt.errorbar(num_trees, bg_avgacc, marker='o', yerr=bg_stderr)
    plt.errorbar(num_trees, rf_avgacc, marker='o', yerr=rf_stderr)

    plt.xlabel("Number of tree(s).")
    plt.ylabel("Test Accuracy.")
    plt.legend(["Bagging", "RF"])
    plt.savefig("num_tree.png")
    plt.show()

def cv_perceptron_plotLR(depths,dt_avgacc):
    plt.figure(figsize=(10, 5))
    plt.title("Testing Accuracy vs Learning Rate")

    plt.errorbar(depths, dt_avgacc, marker='x',color="red")

    plt.xlabel("Learning Rate")
    plt.ylabel("Test Accuracy.")
    plt.legend(["Single Layer Perceptron"])
    plt.savefig("cv_perceptron_lr.png")
    plt.show()