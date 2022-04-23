#Base Code from pytorch tutorial
#modified by Alex Yeh for HW4

import numpy as np
import pandas as pd
from sklearn import svm

class PartitionDataset():
    #assuming k>1, ki starts counting from partition 0
    #assuming k<N
    def __init__(self, name, ki, k, validate):
        #partition the dataset and store only a subset of the data
        all_samp = np.load(open(name+'_Samples.npy', 'rb'))
        all_lab = np.load(open(name+'_Labels.npy', 'rb'))
        N = all_lab.size
        idx_low = int(np.floor(ki/k*N))
        idx_high = int(np.floor((ki+1)/k*N))
        if validate:
            self.samples = all_samp[idx_low:idx_high]
            self.labels = all_lab[idx_low:idx_high]
        else:
            self.samples = np.concatenate([all_samp[0:idx_low],all_samp[idx_high:N]])
            self.labels = np.concatenate([all_lab[0:idx_low],all_lab[idx_high:N]])

    def __len__(self):
        return self.labels.size

    def getSamples(self):
        return self.samples
    def getLabels(self):
        return np.reshape(self.labels,(self.labels.shape[0]))

def test(model, test_dataset):
    correct = 0
    pred = model.predict(test_dataset.getSamples())
    label = test_dataset.getLabels()
    correct = (pred==label).sum()

    #return error
    return 1- correct/len(test_dataset)

def evaluate_C_gamma(C,gamma,train_set):
    #k-fold cross validation
    error_sum = 0
    for ki in range(k):
        dataset_train = PartitionDataset(train_set,ki,k,False)
        dataset_validate = PartitionDataset(train_set,ki,k,True)

        #instantiate model (radial basis/Gaussian kernel)
        model = svm.SVC(kernel='rbf',C=C,gamma=gamma)

        #train model
        model.fit(dataset_train.getSamples(),dataset_train.getLabels())

        #loss for fully trained model
        error = test(model, dataset_validate)
        error_sum+=error
        del model
    avg_error = error_sum/k
    return avg_error

if __name__ == '__main__':
    #model order selection
    k=10
    Cs = np.logspace(-3,3,7) #0.001 to 1000
    gammas = np.logspace(-3,3,7)

    results = pd.DataFrame(index = gammas)

    train_sets = ["D_train1000"]
    #check all training sets
    for t in range(len(train_sets)):
        train_set = train_sets[t]
        losses = np.zeros(gammas.size)

        #measure all C,gamma options
        for i in range(Cs.size):
            print("C: "+str(Cs[i]))
            for j in range(gammas.size):
                print("gamma: "+str(gammas[j]))
                losses[j] = evaluate_C_gamma(Cs[i],gammas[j],train_set)
            results[str(Cs[i])] = losses

    #choose perceptron option for lowest prediction error
    results.to_csv("Q1_SVM_Hyperparameter_Selection.csv")
    print("Min Error: "+str(results.min().min()))
    print("C: " +str(results.min().idxmin()))
    print("Gamma: "+str(results.idxmin()[results.min().idxmin()]))
