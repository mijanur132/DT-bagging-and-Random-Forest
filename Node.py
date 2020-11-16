class Node:
    def __init__(self):
        self.split,self.feature_values,self.leaf,self.left,self.right, self.result =None, None,False,None,None,None
        self.gini,self.data,self.level=0,None,0
        self.excludeFeatures = []
    def __repr__(self):
        return '['+self.split+'  '+str(self.feature)+']'
