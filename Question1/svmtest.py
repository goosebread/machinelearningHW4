import numpy as np
from sklearn import svm

name = "D_train10"
all_samp = np.load(open(name+'_Samples.npy', 'rb'))
all_lab = np.load(open(name+'_Labels.npy', 'rb'))
all_lab = np.reshape(all_lab,(all_lab.shape[0]))
model = svm.SVC(kernel='rbf',C=1,gamma=0.5)
model.fit(all_samp,all_lab)

print(all_samp)
print(all_lab.shape)