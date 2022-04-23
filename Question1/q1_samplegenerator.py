#Alex Yeh
#HW4
#Question 1 Sample Generator

import numpy as np

def makeDistributions():
    rn1 = 2
    rp1 = 4
    sigma = 1
    with open('Q1_DistData.npz', 'wb') as f0:
        np.savez(f0,rn1=rn1,rp1=rp1,sigma=sigma)

def makeSamples(N,name):
    #N = number of Samples

    #assume uniform priors
    prior = 0.5
    distdata = np.load('Q1_DistData.npz')

    #distribution for theta
    theta = np.random.uniform(-np.pi,np.pi,N)

    #distribution for n
    n = np.random.multivariate_normal([0,0],distdata['sigma']**2*np.eye(2), N)

    #generate true labels and samples
    A = np.random.rand(N,1)
    class0 = A<=prior 
    class1 = A>prior

    trueClassLabels = class0*(-1)+class1*(1)
    print("Class Priors")
    print("p(L=-1) = "+str(np.sum(trueClassLabels==-1)/N))
    print("p(L=1) = "+str(np.sum(trueClassLabels==1)/N))

    x0 = distdata["rn1"] * np.concatenate([[np.cos(theta)],[np.sin(theta)]],axis=0).T+n
    x1 = distdata["rp1"] * np.concatenate([[np.cos(theta)],[np.sin(theta)]],axis=0).T+n

    #class0,class1 are mutually exclusive and collectively exhaustive
    samples = class1*x1 + class0*x0 

    print(samples)
    print(np.linalg.norm(samples,ord=2,axis=1))

    print(trueClassLabels)

    #store true labels and samples
    with open(name+str(N)+'_Labels.npy', 'wb') as f1:
        np.save(f1, trueClassLabels)
    with open(name+str(N)+'_Samples.npy', 'wb') as f2:
        np.save(f2, samples)

#run script
if __name__ == '__main__':
    makeDistributions()
    makeSamples(1000,"D_train")
    makeSamples(10000,"D_test")
