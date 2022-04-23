#Alex Yeh
#HW4 Q2
#finds GMM model order that best fits the image data

import numpy as np
from sklearn.mixture import GaussianMixture

#dataset class for k-fold partitioning
class PartitionDataset():
    #assuming k>1, ki starts counting from partition 0
    #assuming k<N
    def __init__(self, name, ki, k, validate):
        #partition the dataset and store only a subset of the data
        all_samp = np.load(open(name+'.npy', 'rb'))
        N = all_samp.shape[0]
        idx_low = int(np.floor(ki/k*N))
        idx_high = int(np.floor((ki+1)/k*N))
        if validate:
            self.samples = all_samp[idx_low:idx_high]
        else:
            self.samples = np.concatenate([all_samp[0:idx_low],all_samp[idx_high:N]])

    #def __len__(self):
    #    return self.labels.shape[0]

    def getSamples(self):
        return self.samples

if __name__ == '__main__':
    k=10 #cross validation k 
    image = 'image'

    #model order candidates
    orders = np.array([2,3,4,5,6,7])

    #average validation log likelihood
    averageValLL = np.zeros(len(orders))

    #for each model
    for o in range(len(orders)):
        order = orders[o]
        LLsum = 0

        #10 fold cross validation
        #for each partition
        for ki in range(k):
            dataset_train = PartitionDataset(image,ki,k,False)
            dataset_validate = PartitionDataset(image,ki,k,True)

            #instantiate model    
            #sklearn mixture gaussian mixture for GMM implementation
            model = GaussianMixture(n_components=order)

            #train model using EM algorithm
            model.fit(dataset_train.getSamples())

            # per sample average log likelihood of validation samples given trained GMM
            LLsum += model.score(dataset_validate.getSamples())

        #per sample average log likelihood on validation samples
        averageValLL[o] = LLsum/k

        print("Model Order: "+str(order)+"\t Average Log Likelihood: "+str(LLsum/k))

    print("Greatest Log Likelihood: "+str(np.max(averageValLL)))
    print("Model Order Selected: "+str(orders[np.argmax(averageValLL)]))
