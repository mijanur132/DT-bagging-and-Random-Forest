import numpy as np
from Node import *
import pandas as pd
import random
from statistics import *
from scipy import stats
import sys

class DecisionTree:
    # Takes in the data and the target column (as a string).    # Parameterized Constructor to intitalize the decision tree.
    def __init__(self, data, target, isRandomForrest, leaf_lim = 50, depth = None):
        self.data, self.target, self.root, self.depth,self.leaf_lim,self.isRandomForrest = data,target, None, depth, leaf_lim,isRandomForrest
    def giniIndex(dset, target):#
        return 1 - (len(dset[dset[target]==True])/len(dset))**2-(len(dset[dset[target]==False])/len(dset))**2

    # Method that builds the decision tree.
    def build(self, data,isRandomForrest,level = 0):
        left_dataset =  right_dataset = best_feature = feature_values = None
        min_gini = np.inf
        gini = len(data)*DecisionTree.giniIndex(data,self.target)
        node = Node()
        node.data = data # Assign the data to the node, i.e. this variable keeps track of all the data that is present under this# node.
        # If the number of examples for this node is less than 50, then just make it a leaf node.
        if (node.data.shape[0] <= self.leaf_lim):
            node.leaf,node.gini = True,gini   # Make this node a leaf node.
            node.result = len( data[data[self.target]==True] ) >= len( data[data[self.target]==False] )
            return node
        node.level, node.gini = level,gini
        #print("level,depth",level,self.depth)
        if self.depth is not None and level == self.depth:  # When we reach the maximum permissible depth. (Have reached a leaf node).
            node.leaf = True
            # The resulting class of this leaf, is going to be the majority between of the classes between 0 and 1.
            node.result = len( data[data[self.target]==True] ) >= len( data[data[self.target]==False] )
            return node

        if gini == 0:   # If the calculated impurity for a node is zero, then it is homogeneous and hence we can make it a leaf   # node.
            node.gini, node.leaf = 0, True
            node.result = len( data[data[self.target] == True] ) >= len( data[data[self.target] == False] )
            return node

        if self.isRandomForrest==1:
            #print("random Forest")
            p = list(range(data.columns.shape[0]))
            columns = data.columns
            setp = random.sample(p, int(np.sqrt(len(p))))
            new_columns = [columns[i] for i in setp]
        else:
            new_columns=data.columns

        for feature in new_columns:    # Otherwise, we need to determine the feature (attribute) on which to split.
            if feature == self.target:  # We should not split on the target column.
                continue
            unique = data[feature].unique() # Find all the unique values of this feature.
            tmngini,tldset,trdset,tbftr = np.inf,None,None,None    # Initialize gini, left and right datasets and best feature values
            if len(unique)==1:  # We can't split based on a single value. There must be atleast 2 unique values to be able to split on.
                continue

            for st in range(1,2**len(unique)-1):   # Find the best values for the split on the given feature.
                lvals = [unique[x] for x in [t[0] for t in enumerate(list(bin(st)[2:])[::-1]) if t[1]=='1']]
                lset = data[data[feature].isin(lvals)] # Find the left data set
                rvals = list(set(unique)-set(lvals))         # Find the right data set
                rset = data[data[feature].isin(rvals)]
                if len(lvals)>len(rvals):   # Avoid dealing with duplicate sets
                    continue
                # Find gini index for left,right, total weighted split
                lgini,rgini = DecisionTree.giniIndex(lset,self.target),DecisionTree.giniIndex(rset,self.target)
                tgini = len(lset)*lgini+len(rset)*rgini

                if tgini < tmngini:   # Update the minimum gini
                    tmngini,tldset,trdset,tbftr=tgini,lset,rset,lvals
            if tmngini<min_gini:   #Update minimum gini
                min_gini,left_dataset,right_dataset,feature_values,best_feature = tmngini,tldset,trdset,tbftr,feature

        if min_gini>tmngini:  # No improvement in gini value after split, Make it as leaf node.
            node.leaf = True
            node.result = len( data[data[self.target]] ) > len( data[data[self.target]] )
            return node
        node.min_gini,node.feature_values,node.split= min_gini,feature_values,best_feature
        if left_dataset is not None:   # Build tree for left or right dataset.
            node.left = self.build(left_dataset,isRandomForrest,level+1)
        if right_dataset is not None:
            node.right = self.build(right_dataset,isRandomForrest,level+1)
        if node.left==None and node.right==None:    # If both the trees are not built, it has to be leaf.
            node.leaf = True
        return node

    def fit(self):
        self.root = self.build(self.data,self.isRandomForrest)
        return self.root

    def __predict__(self,s,root):
        if root is None:
            return False
        if root.leaf:
            return root.result
        if s[root.split] in root.feature_values:
            return self.__predict__(s,root.left)
        return self.__predict__(s,root.right)

    def predict(self,s):
        return self.__predict__(s,self.root)

    def test(self,test_df=None):
        score, preds = 0, []
        for i in range(len(test_df)):
            pred = self.predict(test_df.loc[i])
            preds.append(pred)
            if pred == test_df.loc[i].decision:
                score += 1
        return preds, (score * 100 / len(test_df))
