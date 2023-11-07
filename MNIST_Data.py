from keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np
import ot
from ot.bregman import (convolutional_barycenter2d)

#Load the training set and normalize it
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
index=np.where(Y_train==0)[0]
X_train_0=X_train[index,:]
X_train_0=X_train_0/X_train_0.sum(axis=(1,2))[:,None,None]
X_train=X_train/X_train.sum(axis=(1,2))[:,None,None]
X_train=np.reshape(X_train,(np.size(X_train,0),28*28))
X_test=X_test/X_test.sum(axis=(1,2))[:,None,None]
X_test=np.reshape(X_test,(np.size(X_test,0),28*28))

# Generate the barycenter of all the images of the digit "0", which shall serve as our reference measure
epsilon=1e-3
bary0=convolutional_barycenter2d(X_train_0,epsilon)
np.savetxt('BarycenterZero.gz',bary0,delimiter=',')
bary0=np.reshape(bary0,28*28)

#Define the cost matrix M with m_(i,j)(k,l)=sqrt((i-k)^2+(j-l)^2)
M = np.zeros((28*28, 28*28))
# i: Row index of first pixel
# j: Column index of first pixel
# k: Row index of second pixel
# l: Column index of second pixel
for i in range(0,28):
    for j in range(0,28):
        for k in range(0,28):
            for l in range(0,28):
                M[i*28+j,k*28+l]=np.sqrt((i-k)**2+(j-l)**2)

# Compute the distance of the training set to the barycenter 0 as well as the corresponding Kantorovich potentials
MNIST_D_train=np.zeros(np.size(Y_train))
MNIST_P_f=np.zeros((np.size(Y_train),28*28))
MNIST_P_g=np.zeros((np.size(Y_train),28*28))
for i in range (0,np.size(Y_train)):
    TP=X_train[i,:]
    dist,dict=ot.emd(TP,bary0,M,log=True)
    val=list(dict.values())
    d=val[0]
    u=val[1]
    v=val[2]
    MNIST_D_train[i]=d
    MNIST_P_f[i,:]=u
    MNIST_P_g[i,:]=v

np.savetxt('MNIST_D_train.gz',MNIST_D_train,delimiter=',')
np.savetxt('MNIST_P_f.gz',MNIST_P_f,delimiter=',')
np.savetxt('MNIST_P_g.gz',MNIST_P_g,delimiter=',')

# Compute the distance of the test set to the barycenter 0
MNIST_D_test=np.zeros(np.size(Y_test))
for i in range (0,np.size(Y_test)):
    TP=X_test[i,:]
    dist,dict=ot.emd(TP,bary0,M,log=True)
    val=list(dict.values())
    d=val[0]
    MNIST_D_test[i]=d

np.savetxt('MNIST_D_test.gz',MNIST_D_test,delimiter=',')
