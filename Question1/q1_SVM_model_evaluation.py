# Alex Yeh
# HW3 Question 1 true pdf classifier

from q1_SVM_model_selection import *
from q1_visualize import doVisualization
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#main function
if __name__ == '__main__':
    train_sets = ["D_train1000"]
    #values copied from model selection script output
    optiimal_C = [0.1]
    optiimal_gamma = [0.1]

    test_error = np.zeros(len(train_sets))

    for t in range(len(train_sets)):
        train_set = train_sets[t]
        C = optiimal_C[t]
        gamma = optiimal_gamma[t]
        title1 = 'SVM Classifier Decision vs True Label'
        title2 = 'SVM Classifier Error vs True Label'

        train_samples = np.load(open(train_set+'_Samples.npy', 'rb'))
        train_labels = np.load(open(train_set+'_Labels.npy', 'rb'))
        train_labels = np.reshape(train_labels,(train_labels.shape[0]))

        #instantiate model (radial basis/Gaussian kernel)
        model = svm.SVC(kernel='rbf',C=C,gamma=gamma)

        #train model
        model.fit(train_samples,train_labels)

        #Do evaluation on test set
        test_set="D_test10000"
        test_samples = np.load(open(test_set+'_Samples.npy', 'rb'))
        test_labels = np.load(open(test_set+'_Labels.npy', 'rb'))
        test_labels = np.reshape(test_labels,(test_labels.shape[0]))

        predictions = model.predict(test_samples)
        correct = (predictions==test_labels).sum()/len(test_labels)
        test_error[t] = (1-correct)
        #data visualization
        doVisualization(test_samples, np.array([test_labels]), np.array([predictions]), title1, title2)
        
    #table to compare empirically measured test error
    output = pd.DataFrame()
    output["Training Sets"] = train_sets
    output["Test Errors"] = test_error
    output.to_csv("Q1_Test_Errors_SVM.csv")


