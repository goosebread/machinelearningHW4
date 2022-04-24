# Alex Yeh
# HW4 Question 1 evaluate MLP on test set

from q1_MLP_model_selection import *
from q1_visualize import doVisualization
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import StepLR

class CustomDatasetFull(Dataset):
    def __init__(self, name):
        self.samples = np.load(open(name+'_Samples.npy', 'rb'))
        self.labels = np.load(open(name+'_Labels.npy', 'rb'))

    def __len__(self):
        return self.labels.size

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = int(self.labels[idx][0])
        label_vec = np.zeros(2)#2 classes hard coded
        if label==-1:
            label_vec[0]=1
        else:
            label_vec[1]=1 #the label must be given in terms of a 2-d vector 
        return sample, label_vec

def predict(model, device, test_loader):
    model.eval()
    correct = 0
    samples = np.zeros((len(test_loader),2))
    true_labels = np.zeros(len(test_loader))
    predictions = np.zeros(len(test_loader))
    iter=0
    with torch.no_grad():
        for data, target in test_loader:
            #type casting to make things work
            data = data.float()
            target = target.float()

            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max probability
            label = target.argmax(dim=1, keepdim=True)  # get the index of the max probability
            correct += (pred==label).sum() #pred.eq(target.view_as(pred)).sum().item()
            predictions[iter] = int(pred)*2-1 #map [0,1] to [-1,1]
            true_labels[iter] = int(label)*2-1
            samples[iter,:] = data
            iter += 1

    return correct/len(test_loader), samples, true_labels, predictions

#main function
if __name__ == '__main__':
    train_sets = ["D_train1000"]
    #values copied from model selection script output
    optiimal_perceptrons = [256]

    test_error = np.zeros(len(train_sets))

    for t in range(len(train_sets)):
        train_set = train_sets[t]
        n = optiimal_perceptrons[t]
        title1 = 'MLP Classifier Decision vs True Label'
        title2 = 'MLP Classifier Error vs True Label'

        #train models
        use_cuda = False #cpu is faster on my laptop
        device = torch.device("cuda" if use_cuda else "cpu")
        train_kwargs = {'batch_size': 20}#args.batch_size}

        #training data only
        dataset_train = CustomDatasetFull(train_set)
        train_loader = torch.utils.data.DataLoader(dataset_train,**train_kwargs)

        #try 10 different random initial starting points
        #keep best model
        min_loss = np.inf
        best_model = 0

        for r in range(10):
            #instantiate model
            model = Net(n).to(device)
            optimizer = optim.Adadelta(model.parameters(), lr=1)#default lr = 1
            scheduler = StepLR(optimizer, step_size=1, gamma=0.7)#default gamma = 0.7
            loss_old = -1     

            for epoch in range(1, 100):#max 100 epochs. The model usually converges before this limit is reached
                loss = train([], model, device, train_loader, optimizer, epoch)
                if(np.abs(loss-loss_old)<1e-12):#check convergence
                    break
                loss_old = loss
                scheduler.step()
                if epoch==99:
                    print("Warning")

            #average cross entropy loss for fully trained model
            #evaluate on the full set of training data
            loss = test(model, device, train_loader)
            if loss<min_loss:
                best_model = model.state_dict()
            #compare/save the better model
            #print(loss)

            del model

        #for each training set, do evaluation on test set
        model = Net(n).to(device)
        model.load_state_dict(best_model)

        dataset_test = CustomDatasetFull("D_test10000")
        test_loader = torch.utils.data.DataLoader(dataset_test,shuffle=False)

        #run on test set
        [correct, samples, trueLabels, predictions] = predict(model, device, test_loader)
        test_error[t] = (1-correct)
        #data visualization
        doVisualization(samples, np.array([trueLabels]), np.array([predictions]), title1, title2)
        
    #table to compare empirically measured test error
    output = pd.DataFrame()
    output["Training Sets"] = train_sets
    output["Test Errors"] = test_error
    output.to_csv("Q1_Test_Errors_MLP.csv")


