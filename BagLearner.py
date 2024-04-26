import numpy as np
from scipy import stats

class BagLearner(object):

    def __init__(self, learner, kwargs = {}, bags = 20):

        self.learners = []
        self.kwargs = kwargs
        self.bags = bags
        for i in range(self.bags):
            self.learners.append(learner(**kwargs))
          		 	   			  		 			     			  	   		  	   		 	   			  		 			     			  	 
        pass

    def add_evidence(self,data_x, data_y):

        temp = np.array([data_y]).T
        data = np.concatenate((data_x,temp),axis=1)
        
        for i in range(self.bags):

            sizee = int(np.rint(data.shape[0] * .6))
            np.random.seed(903949363)
            sample = data[np.random.randint(data.shape[0], size=sizee), :]
            temp_y = np.array(sample[:,-1])

            self.learners[i].add_evidence(sample[:,:-1], temp_y)

    def author(self):
        return 'jpeterson93'


    def query(self, points):
        
        vals = np.empty([self.bags,points.shape[0]])
        for i in range(self.bags):
            vals[i] = self.learners[i].query(points)

        mode_vals = stats.mode(vals)
        # print(np.unique(mode_vals[0]))
        return mode_vals[0]
        
        

