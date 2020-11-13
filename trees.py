import pandas as pd
import numpy as np
import random
from statistics import *
from scipy import stats
import sys

class Node:
    def __init__(self):
        self.split,self.feature_values,self.leaf,self.left,self.right, self.result =None, None,False,None,None,None
        self.gini,self.data,self.level=0,None,0
        self.excludeFeatures = []
    def __repr__(self):
        return '['+self.split+'  '+str(self.feature)+']'

class DecisionTree:
    # Takes in the data and the target column (as a string).    # Parameterized Constructor to intitalize the decision tree.
    def __init__(self, data, target, leaf_lim = 50, depth = None):
        self.data, self.target, self.root, self.depth,self.leaf_lim = data,target, None, depth, leaf_lim
    def giniIndex(dset, target):#
        return 1 - (len(dset[dset[target]==True])/len(dset))**2-(len(dset[dset[target]==False])/len(dset))**2
     
    # Method that builds the decision tree.
    def build(self, data,level = 0):
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

        if self.depth is not None and level == self.depth:  # When we reach the maximum permissible depth. (Have reached a leaf node).
            node.leaf = True
            # The resulting class of this leaf, is going to be the majority between of the classes between 0 and 1.
            node.result = len( data[data[self.target]==True] ) >= len( data[data[self.target]==False] )
            return node

        if gini == 0:   # If the calculated impurity for a node is zero, then it is homogeneous and hence we can make it a leaf   # node.
            node.gini, node.leaf = 0, True
            node.result = len( data[data[self.target] == True] ) >= len( data[data[self.target] == False] )
            return node

        for feature in data.columns:    # Otherwise, we need to determine the feature (attribute) on which to split.
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
            node.left = self.build(left_dataset,level+1)
        if right_dataset is not None:
            node.right = self.build(right_dataset,level+1)
        if node.left==None and node.right==None:    # If both the trees are not built, it has to be leaf.
            node.leaf = True
        return node        

    def fit(self):
        self.root = self.build(self.data)
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
   
def t(x):
    if x is None:
        return ""
    return str(x)

def build_tree(root):
    if root==None:
        return ""
    return {"split" : t(root.split)+' '+str(root.level), "left":build_tree(root.left), "right":build_tree(root.right) }

def test(dtree, test_df = None):
    score,preds = 0,[]
    for i in range(len(test_df)):
        pred = dtree.predict(test_df.loc[i])
        preds.append(pred)
        if pred == test_df.loc[i].decision:
            score+=1
    return preds, (score*100/len(test_df))

## (ii) bagging()
def bagging(trainingSet = None, testSet = None, bagging_num = 30, depth = 8):
    trainBagPreds,testBagPreds,trainAcc = [],[],0
    for iteration in range(bagging_num):
        trainSet = trainingSet.sample(frac = 1, replace = True).reset_index(drop = True)
        ytest = testSet["decision"]
        dtree = DecisionTree(trainSet, target = 'decision', leaf_lim = 50, depth = depth)  # Instantiate the decision tree.
        root = dtree.fit() # Train the decision tree.
        trainPreds, train_acc = test(dtree, trainingSet) # Test this decision tree on the training set.
        trainAcc += train_acc
        preds, test_acc = test(dtree, testSet)  # Test this decision tree on the testing set.
        trainBagPreds.append(trainPreds)
        testBagPreds.append(preds)
    trainBagPreds,testBagPreds,finalTrainPreds,trainPredDf = np.array(trainBagPreds),np.array(testBagPreds),[],pd.DataFrame(trainBagPreds)
    for i in range(trainBagPreds.shape[1]):
        finalTrainPreds.append( trainPredDf[i].value_counts().idxmax() )
    finalTestPreds,testPredDf = [],pd.DataFrame(testBagPreds)
    for i in range(testBagPreds.shape[1]):
        finalTestPreds.append( testPredDf[i].value_counts().idxmax() )
    return np.array(finalTrainPreds), np.array(finalTestPreds), (trainAcc / bagging_num)


## (iii)  Random Forests.

class Node:
    def __init__(self):
        self.split = None
        self.feature_values = None
        self.leaf = False
        self.left = None
        self.right = None
        self.result = None
        self.gini = 0
        self.data = None
        self.level=0
        
    def __repr__(self):
        return '['+self.split+'  '+str(self.feature)+']'

class DecisionTree_Random:
    
    # Takes in the data and the target column (as a string).

    # Parameterized Constructor to intitalize the decision tree.
    def __init__(self, data, target, leaf_lim = 50, depth = None):
        self.data = data
        self.target = target
        self.root = None
        self.depth = depth
        self.leaf_lim = leaf_lim
        
    # Helper function to calculate the gini index of the dataset.
    def giniIndex(dset, target):
#         # Since we're modeling a binary classification scenario. TODO : Make it more generic.
#         num_classes = list(dset[target].unique())
        
#         # For each class, compute the p^2 term.
        return 1 - (len(dset[dset[target]==True])/len(dset))**2-(len(dset[dset[target]==False])/len(dset))**2
     
    # Method that builds the decision tree.
    def build(self, data,level = 0):
        
        # Variable that keeps track of the dataset for the left child of a node.
        left_dataset = None
        
        # Variable that keeps track of the dataset for the right child of a node.
        right_dataset = None
        
        # Initialize the minimum gini index to a very high value.
        min_gini = np.inf
        
        # Variable to keep track of the best features.
        best_feature = None
        
        # Variable to keep track of the feature values.
        feature_values = None
        
        # Calculate the gini index of the entire dataset.
        gini = len(data)*DecisionTree_Random.giniIndex(data,self.target)
  
        # Create a node (to attach to our decision tree).
        node = Node()
        
        # Assign the data to the node, i.e. this variable keeps track of all the data that is present under this 
        # node.
        node.data = data
        
        # If the number of examples for this node is less than 50, then just make it a leaf node.
        if (data.shape[0] <= self.leaf_lim):
            # Make this node a leaf node.
            node.leaf = True
            node.gini = gini
            node.result = len( data[data[self.target]==True] ) >= len( data[data[self.target]==False] )
            return node
        
        node.level = level
        node.gini = gini
        
        # When we reach the maximum permissible depth. (Have reached a leaf node).
        if self.depth is not None and level == self.depth:
            node.leaf = True
            # The resulting class of this leaf, is going to be the majority between of the classes between 0 and 1.
            node.result = len( data[data[self.target]==True] ) >= len( data[data[self.target]==False] )
            return node
        
        # If the calculated impurity for a node is zero, then it is homogeneous and hence we can make it a leaf 
        # node.
        if gini == 0:
            node.gini = 0
            node.leaf = True
            node.result = len( data[data[self.target] == True] ) >= len( data[data[self.target] == False] )
            return node
            
        
        # Otherwise, we need to determine the feature (attribute) on which to split.
       
        '''
            Heart of the random forests, where I consider only sqrt(p) features, where p. : number of the features 
            that I have originally.
        '''
        p = list(range(data.columns.shape[0]))
        columns = data.columns
        setp = random.sample(p, int(np.sqrt(len(p))))
        new_columns = [columns[i] for i in setp]
        
        # Traverse the path from the root to the current node, and disregard all the features used from the root
        # to the that node.
        # TODO.
        
        for feature in new_columns:
            # We should not split on the target column.
            if feature == self.target:
                continue
            
            # Find all the unique values of this feature.
            unique = data[feature].unique()
                        
            # Initialize gini, left and right datasets and best feature values
            tmngini = np.inf
            tldset = None
            trdset = None
            tbftr = None

            # We can't split based on a single value. There must be atleast 2 unique values to be able to split on.
            if len(unique)==1:
                continue
            
            # Find the best values for the split on the given feature.
            for st in range(1,2**len(unique)-1):
                
                lvals = [unique[x] for x in [t[0] for t in enumerate(list(bin(st)[2:])[::-1]) if t[1]=='1']]

                # Find the left data set
                lset = data[data[feature].isin(lvals)]
                rvals = list(set(unique)-set(lvals))
                
                # Find the right data set
                rset = data[data[feature].isin(rvals)]
                
                # Avoid dealing with duplicate sets
                if len(lvals)>len(rvals):
                    continue
                    
                # Find gini index for left split
                lgini = DecisionTree_Random.giniIndex(lset,self.target)
                
                # Find gini impurity for right split
                rgini = DecisionTree_Random.giniIndex(rset,self.target)
                
                # Find the total weighted gini. 
                tgini = len(lset)*lgini+len(rset)*rgini

                # Update the minimum gini
                if tgini < tmngini:
                    tmngini=tgini
                    tldset = lset
                    trdset = rset
                    tbftr = lvals
                    
            #Update minimum gini
            if tmngini<min_gini:
                min_gini = tmngini
                left_dataset = tldset
                right_dataset = trdset
                feature_values = tbftr
                best_feature = feature
                
        # No improvement in gini value after split, Make it as leaf node.
        if min_gini>tmngini:
            node.leaf = True
            node.result = len( data[data[self.target]] ) > len( data[data[self.target]] )
            return node
            
        node.min_gini= min_gini
        node.feature_values = feature_values
        node.split = best_feature
        
        # Build tree for left dataset.
        if left_dataset is not None:
            node.left = self.build(left_dataset,level+1)
            
        # Build tree for right dataset.
        if right_dataset is not None:
            node.right = self.build(right_dataset,level+1)
            
        # If both the trees are not built, it has to be leaf.
        if node.left==None and node.right==None:
            node.leaf = True
        
        return node        
        
    
    def fit(self):
        self.root = self.build(self.data)
        return self.root
        
    def __predict__(self,s,root):
        if root is None:
            return False
        if root.leaf:
            return root.result
        if s[root.split] in root.feature_values:
            return self.__predict__(s,root.left)
        else:
            return self.__predict__(s,root.right)
        
    def predict(self,s):
        return self.__predict__(s,self.root)
   
def t(x):
    if x is None:
        return ""
    return str(x)

def build_tree(root):
    if root==None:
        return ""
    return {"split" : t(root.split)+' '+str(root.level), "left":build_tree(root.left), "right":build_tree(root.right) }
def randomForests(trainingSet = None, testSet = None, bagging_num = 30, depth = 8):
    
    trainBagPreds = []
    testBagPreds = []
    trainAcc = 0

    for iteration in range(bagging_num):
        
        trainSet = trainingSet.sample(frac = 1, replace = True).reset_index(drop = True)
        ytest = testSet["decision"]

        # Instantiate the decision tree.
        dtree = DecisionTree_Random(trainSet, target = 'decision', leaf_lim = 50, depth = depth)

        # Train the decision tree.
        root = dtree.fit()

        # Test this decision tree on the training set.
        #trainPreds, train_acc = trees.test(dtree, trainingSet)
        trainPreds, train_acc = test(dtree, trainingSet)
        trainAcc += train_acc

        # Test this decision tree on the testing set.
        preds, test_acc = test(dtree, testSet)

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
    
    trainFileName = sys.argv[1]
    testFileName = sys.argv[2]
    modelIdx = int(sys.argv[3])
    
    trainingSet = pd.read_csv(trainFileName)
    testSet = pd.read_csv(testFileName)
    bagPreds=[]
    if (modelIdx == 1):
        
        tree = DecisionTree(trainingSet, target = 'decision', leaf_lim = 50, depth = 8)
        root = tree.fit()
        # Test this decision tree.
        preds, test_acc = test(tree, testSet)
        bagPreds.append(preds)               
        preds, acc = test(tree, trainingSet)
        print('Training Accuracy DT: %.2f' % acc)
        print('Testing Accuracy DT: %.2f' % test_acc)
        
    elif (modelIdx == 2):
        
        trainBagPreds, testBagPreds, trainAcc = bagging(trainingSet, testSet, bagging_num = 30)
        print ("Training Accuracy BT: %.2f" % trainAcc)
        print ("Testing Accuracy BT: %.2f" %  (np.mean(testBagPreds == testSet["decision"].values)*100) )
        
    else:
        
        # modelIdx == 3.
        trainBagPreds, testBagPreds, trainAcc = randomForests(trainingSet, testSet, bagging_num = 30)
        print ("Training Accuracy RF: %.2f" % trainAcc)
        print ("Testing Accuracy RF: %.2f" % (np.mean(testBagPreds == testSet["decision"].values)*100) )

