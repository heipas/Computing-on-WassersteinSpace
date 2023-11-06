import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
import torchvision
import torchvision.transforms as transforms


# Load the CIFAR 10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(),
                                        download=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor(),
                                       download=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, 
                                         shuffle=False, num_workers=2)

X_train=trainset.data
Y_train=trainset.targets
X_test=testset.data
Y_test=testset.targets

# Normalize the dataset
X_train=X_train/X_train.sum(axis=(1,2,3))[:,None,None,None]*32*32
X_train=np.reshape(X_train,(np.size(X_train,0),32*32*3))
X_test=X_test/X_test.sum(axis=(1,2,3))[:,None,None,None]*32*32
X_test=np.reshape(X_test,(np.size(X_test,0),32*32*3))

# Define the cost matrix M with m_(i,j)(k,l)=sqrt((i-k)^2+(j-l^2))
M = np.zeros((32*32*3, 32*32*3))
# i: Row index of first pixel
# j: Column index of first pixel
# k: Row index of second pixel
# l: Column index of second pixel
# m,n: RGB index
for m in range(0,3):
    for n in range(0,3):
        for i in range(0,32):
            for j in range(0,32):
                for k in range(0,32):
                    for l in range(0,32):
                        M[m+i*32*3+j*3,n+k*3*32+l*3]=np.sqrt((m-n)**2+(i-k)**2+(j-l)**2)

# Define the 34th picture in the training set as our reference measure
X_reference=X_train[34]
X_reference=np.reshape(X_reference,32*32*3)

# Compute the Wasserstein distance to the reference measure and the corresponding Kantorovich potentials
CIFAR_D_train=np.zeros(np.size(Y_train))
CIFAR_P_f=np.zeros((np.size(Y_train),32*32*3))
CIFAR_P_g=np.zeros((np.size(Y_train),32*32*3))
for i in range (0,np.size(Y_train)):
    TP=X_train[i,:]
    dist,dict=ot.emd(TP,X_reference,M,log=True)
    val=list(dict.values())
    d=val[0]
    u=val[1]
    v=val[2]
    CIFAR_D_train[i]=d
    CIFAR_P_f[i,:]=u
    CIFAR_P_g[i,:]=v

CIFAR_D_test=np.zeros(np.size(Y_test))
for i in range (0,np.size(Y_test)):
    TP=X_test[i,:]
    dist,dict=ot.emd(TP,X_reference,M,log=True)
    val=list(dict.values())
    d=val[0]
    u=val[1]
    v=val[2]
    CIFAR_D_test[i]=d

np.savetxt('CIFAR_D_test.gz',CIFAR_D_test,delimiter=',')
np.savetxt('CIFAR_D_train.gz',CIFAR_D_train,delimiter=',')
np.savetxt('CIFAR_P_f.gz',CIFAR_P_f,delimiter=',')
np.savetxt('CIFAR_P_g.gz',CIFAR_P_g,delimiter=',')
