
import pandas as pd


class df_class():
    def __init__(self,fileName):
        self.csvData= pd.read_csv(fileName)
       # self.csvData.rename(columns={"intelligence_parter": "intelligence_partner"}, inplace=True)
        self.pref_scores_participant=[col for col in self.csvData.columns if "important" in col]
        self.pref_scores_partner = [col for col in self.csvData.columns if "pref_o" in col]
        self.pref_scores_participant_total = self.csvData[self.pref_scores_partner].sum(axis=1)
        self.pref_scores_partner_total = self.csvData[self.pref_scores_participant].sum(axis=1)
        self.num_bins=2
        self.trainSet=""
        self.testSet=""
        self.excludeList = ["gender", "race", "race_o", "samerace", "field", "decision"]
    def saveCsv(self):
        self.trainSet.to_csv("trainingSet.csv", index = False)
        self.testSet.to_csv("testSet.csv", index = False)
