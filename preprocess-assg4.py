import pandas as pd
import numpy as np
from data import *
import warnings
warnings.filterwarnings('ignore')

#--> Read the data, and discard the last 244 lines of the dataset.
#--> (i) Dropping the columns race, race_o and field, for simplicity.
#--> (ii) Label encoding on the categorical attribute "Gender" (like in Assignment 2).
dfc = df_class("dating-full.csv")
df = dfc.csvData
df = df.loc[:6499,:]
#print(df.shape)
df.drop(["race", "race_o", "field"], inplace = True, axis = 1)
uniq_gender=df["gender"].unique()
uniq_gender_dim=uniq_gender.shape[0]

integer_encode = list( range( uniq_gender_dim ) )
#print(integer_encode)
df.replace(uniq_gender, integer_encode, inplace = True)
#print(df["gender"])



# (iii) Repeat the preprocessing steps 1(iv) that you did in Assignment 2.
"""
'''
    1.  Normalization of :
        (1) preference scores of participant
        (2) preference scores of partner
'''

prefScoresParti = ["attractive_important", "sincere_important", "intelligence_important", "funny_important", "ambition_important", "shared_interests_important"]
prefScoresPartn = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests"]

partiTotal = df[prefScoresParti].sum(axis = 1)
partnTotal = df[prefScoresPartn].sum(axis = 1)

for column in range(len(prefScoresParti)):
        # Do the normalization for each column.
        df[prefScoresParti[column]] = df[prefScoresParti[column]] / partiTotal
    
for column in range(len(prefScoresPartn)):
        df[prefScoresPartn[column]] = df[prefScoresPartn[column]] / partnTotal

"""
for column in range(len(dfc.pref_scores_participant)):
    df[dfc.pref_scores_participant[column]] = df[dfc.pref_scores_participant[
        column]] / dfc.pref_scores_participant_total
    #print("Mean of %s: %.2f" % (dfc.pref_scores_participant[column], np.mean(df[dfc.pref_scores_participant[column]])))

for column in range(len(dfc.pref_scores_partner)):
    df[dfc.pref_scores_partner[column]] = df[dfc.pref_scores_partner[column]] / dfc.pref_scores_partner_total
    #print("Mean of %s: %.2f" % (dfc.pref_scores_partner[column], np.mean(df[dfc.pref_scores_partner[column]])))




# (iv) Discretize all the continuous-valued columns (the columns mentioned as continuous valued columns in Assignment 2)
# using 2 bins of equal widths.

# All columns other than [gender, race, race o, samerace, field, decision].


'''
    NOTE :  The range of [0,1] only applies to attributes that we normalized in Q1.(iv).
            For all the other attributes, get the range from the data and then bin it into num_bins categories.
'''

labels = list(range(dfc.num_bins))

# List of attributes that shouldn't be binned.


# Attributes that already have a range pre-defined.
#preRange = ["age","age_o", "interests_correlate"]

#prefScoresParti = ["attractive_important", "sincere_important", "intelligence_important", "funny_important", "ambition_important", "shared_interests_important"]
#prefScoresPartn = ["pref_o_attractive", "pref_o_sincere", "pref_o_intelligence", "pref_o_funny", "pref_o_ambitious", "pref_o_shared_interests"]

ctr = 0

for attribute in df.columns:
    if (attribute not in dfc.excludeList):
        if (attribute in dfc.pref_scores_participant) or (attribute in dfc.pref_scores_partner):
            # Then the range considered is [0,1].
            custom_bins = np.linspace(0,1,dfc.num_bins + 1)
            df[attribute] = pd.cut(df[attribute], custom_bins, include_lowest = True, labels = labels)
            df[attribute] = df[attribute].fillna(labels[-1])
        else:
            min_ = np.min(df[attribute])
            max_ = np.max(df[attribute])
            custom_bins = np.linspace(min_ , max_, dfc.num_bins + 1)
            df[attribute] = pd.cut(df[attribute], custom_bins, include_lowest = True, labels = labels)
            df[attribute] = df[attribute].fillna(labels[-1])

# (v) Use the sample function from pandas with the parameters initialized as 
# random state = 47, frac = 0.2 to take a random 20% sample from the entire dataset.

dfc.testSet = df.sample(random_state=47, frac = 0.2)
testIndices = dfc.testSet.index
dfc.trainSet = df.loc[~df.index.isin(testIndices)]
dfc.saveCsv()

# Save the output of this module as, trainingSet.csv and testSet.csv
#trainSet.to_csv("trainingSet.csv", index = False)
#testSet.to_csv("testSet.csv", index = False)
