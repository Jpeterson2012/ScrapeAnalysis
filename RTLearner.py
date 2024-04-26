import numpy as np  		  	   		 	   			  		 			     			  	   	   		 	   			  		 			     			  	   		  	   		 	   			  		 			     			  	   		  	   		 	   			  		 			     			  	   		  	   		 	   			  		 	

def mode(df):
    freq = {1: 0, 0: 0, -1: 0}
    for i in range(df.shape[0]):
        if df[i,-1] == 1:
            freq[1] += 1
        elif df[i,-1] == -1:
            freq[-1] += 1
        else:
            freq[0] += 1

    hf = max(freq.values())
    hflist = []

    for i, j in freq.items():
        if j == hf:
            hflist.append(i)

    if len(hflist) == 2:
        if hflist[0] == 0:
            hflist.remove(0)
    return hflist
    

def build_tree(data, leaf_size):
    leaf = -1
    
    if data.shape[0] <= leaf_size:
        if data.shape[0] == 1:
            return	np.array([[leaf,	data[0,-1],	-1,	-1]])
        else:
            temp = mode(data)
            # temp2 = stats.mode(data[:,-1])
            # print(temp)
            return np.array([[leaf, temp[0], -1, -1]])

    
    if len(np.unique(data[:,-1])) == 1:
        return	np.array([[leaf,	data[0,-1],	-1,	-1]])
    else:
        
        i = np.random.randint(0, data.shape[1] - 1)
        data = data[data[:,i].argsort()]
        
        SplitVal =	np.median(data[:,i])
        
        if data.shape[0] == data[data[:,i]<=SplitVal].shape[0]:
            # SplitVal = np.mean(data[:,i])
            temp = mode(data)
            return np.array([[leaf, temp[0], -1, -1]])
        
        lefttree =	build_tree(data[data[:,i]<=SplitVal], leaf_size)
        righttree =	build_tree(data[data[:,i]>SplitVal], leaf_size)
        root =	np.array([[i,	SplitVal, 1, lefttree.shape[0] + 1]])
        return np.concatenate((root,lefttree, righttree),axis=0)
    
class RTLearner(object):

    def __init__(self, leaf_size=5,verbose=False):

        self.leaf_size = leaf_size  
        self.verbose = verbose
        self.tree = []		  	   		 	   			  		 			     			  	 
        		  	   		 	   			  		 			     			  	 
        pass 


    def add_evidence(self,data_x, data_y):
        
        temp = np.array([data_y]).T
        data = np.concatenate((data_x,temp),axis=1)
        self.tree = build_tree(data,self.leaf_size)
     

    def author(self):
        return 'jpeterson93'
    
    def return_tree(self):
        return self.tree


    def query(self,points):
        predictions = np.zeros(points.shape[0])
        
        for j in range(points.shape[0]):
            index = 0
            for x in range(self.tree.shape[0]):
                if self.tree[index][0] == -1.:
                    break

                temp = int(self.tree[index][0])
                if points.iloc[j,temp] <= self.tree[index][1]:
                    index = index + int(self.tree[index][2])    
                else:
                    index = index + int(self.tree[index][3])

            predictions[j] = self.tree[index][1]
        return predictions
