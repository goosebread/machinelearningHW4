# Alex Yeh
# HW4 Question 1 does visualizations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

#Visualizations
def doVisualization(samples, trueLabels, Decisions, title1, title2):
    #output measured error
    correctDecision = trueLabels==Decisions
    print("Measured Error = "+str(1-np.average(correctDecision)))
    data = np.concatenate((samples,trueLabels.T,Decisions.T,correctDecision.T),axis=1)

    #this filtering scheme requires a reshape to return to 2d matrix representation
    data1 = data[np.argwhere(data[:,2]==-1),:]
    data1 = np.reshape(data1,(data1.shape[0],data1.shape[2]))
    data2 = data[np.argwhere(data[:,2]==1),:]
    data2 = np.reshape(data2,(data2.shape[0],data2.shape[2]))

    #plot Decisions vs Actual Label
    fig,ax = plt.subplots()
    cmap1 = LinearSegmentedColormap.from_list('4class', [(1, 0.5, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0),(0, 0, 1)])
    legend_elements1 = [ax.scatter([0], [0], marker = 'o',c = -1, label='Label = -1',cmap=cmap1,vmin=-1,vmax=1),
                    ax.scatter([0], [0], marker = 's',c = 1,label='Label = 1',cmap=cmap1,vmin=-1,vmax=1)]
    l1=ax.scatter(data1[:,0],data1[:,1],marker = 'o',c = data1[:,3],label='Label = -1',cmap=cmap1,vmin=-1,vmax=1,alpha=0.3,zorder=3)
    l2=ax.scatter(data2[:,0],data2[:,1],marker = 's',c = data2[:,3],label='Label = 1',cmap=cmap1,vmin=-1,vmax=1,alpha=0.3,zorder=4)

    ax.set_title(title1)
    ax.legend(handles=legend_elements1,title="Shape = True Label\nColor = Classifier Decision")

    #plot Error vs Actual Label
    fig2,ax2 = plt.subplots()

    cmap2 = LinearSegmentedColormap.from_list('redTransparentGreen', [(1, 0, 0, 1), (0.5, 1, 0.5, 0.1)])
    legend_elements2 = [ax.scatter([0], [0], marker = 'o',c = 1, label='Label = -1',cmap=cmap2,vmin=0,vmax=1),
                    ax.scatter([0], [0], marker = 's',c = 1,label='Label = 1',cmap=cmap2,vmin=0,vmax=1)]

    #this filtering scheme requires a reshape to return to 2d matrix representation
    data11 = data1[np.argwhere(data1[:,4]==0),:]
    data11 = np.reshape(data11,(data11.shape[0],data11.shape[2]))
    data12 = data1[np.argwhere(data1[:,4]==1),:]
    data12 = np.reshape(data12,(data12.shape[0],data12.shape[2]))
    data21 = data2[np.argwhere(data2[:,4]==0),:]
    data21 = np.reshape(data21,(data21.shape[0],data21.shape[2]))
    data22 = data2[np.argwhere(data2[:,4]==1),:]
    data22 = np.reshape(data22,(data22.shape[0],data22.shape[2]))

    l11=ax2.scatter(data11[:,0],data11[:,1],marker = 'o',c = data11[:,4], label='Label = -1',cmap=cmap2,vmin=0,vmax=1,zorder=1)
    l12=ax2.scatter(data12[:,0],data12[:,1],marker = 'o',c = data12[:,4], label='Label = -1',cmap=cmap2,vmin=0,vmax=1,zorder=3)
    l21=ax2.scatter(data21[:,0],data21[:,1],marker = 's',c = data21[:,4], label='Label = 1',cmap=cmap2,vmin=0,vmax=1,zorder=2)
    l22=ax2.scatter(data22[:,0],data22[:,1],marker = 's',c = data22[:,4], label='Label = 1',cmap=cmap2,vmin=0,vmax=1,zorder=4)

    ax2.set_title(title2)
    lg = ax2.legend(handles=legend_elements2,title="Red Marker = Incorrect Classification")
    for i in range(2):
        lg.legendHandles[i].set_alpha(1)
    plt.show()
