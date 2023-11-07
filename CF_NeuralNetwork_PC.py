import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
import numpy as np

class FFNNPC(nn.Module):
    def __init__(self,Amat,bec,dimension_space,number_potentials,non_trainable):
        super(FFNNPC,self).__init__()

        self.layers = nn.ModuleList()
        self.relu=nn.ReLU()
        helpA=nn.ModuleList()
        helpB=nn.ModuleList()
        Azero=nn.Linear(dimension_space,number_potentials)
        # Initialize the first hidden layer
        with torch.no_grad():
            Azero.weight.copy_(torch.from_numpy(Amat).float())
            Azero.bias.copy_(torch.from_numpy(bec).float())
        self.layers.append(Azero)

        # Recursively define the max operator as a neural network
        A1=torch.tensor(([1.0,-1.0],[0.0,1.0],[0.0,-1.0]))
        A2=torch.tensor(([[1,1,-1]]), dtype=float)
        with torch.no_grad():
                A2linear=nn.Linear(A2.shape[1],A2.shape[0], bias=False)
                A1linear=nn.Linear(A1.shape[1],A1.shape[0], bias=False)
                A1linear.weight.copy_(A1)
                A2linear.weight.copy_(A2)
                helpA.append(A1linear)
                helpB.append(A2linear)
        i=2
        while i<=int(number_potentials/2):
            A=A1
            B=A2
            j=1
            while j <= i-1:
                A=torch.block_diag(A,A1)
                B=torch.block_diag(B,A2)
                j=j+1
            with torch.no_grad():
                Alinear=nn.Linear(A.shape[1],A.shape[0],bias=False)
                Blinear=nn.Linear(B.shape[1],B.shape[0],bias=False)
                Alinear.weight.copy_(A)
                Blinear.weight.copy_(B)
                helpA.append(Alinear)
                helpB.append(Blinear)
            i=i*2
        k=number_potentials
        l=0
        self.layers.append(helpA[-1])
        while k>2:
            A=helpA[-2-l].weight
            B=helpB[-1-l].weight
            C=torch.matmul(A,B)
            Clinear=nn.Linear(C.shape[1],C.shape[0])
            with torch.no_grad():
                Clinear.weight.copy_(C)
            self.layers.append(Clinear)
            k=int(k/2)
            l=l+1
        self.layers.append(A2linear)

        # Make the parameters non-trainable
        if non_trainable:
            for i in range(1,len(self.layers)):
                for p in self.layers[i].parameters():
                    p.requires_grad=False

    def forward(self, x):
        out = self.layers[0](x)
        for i in range(1,len(self.layers) - 1):
            out = self.relu(self.layers[i](out))
        return self.layers[-1](out)

    # Output of the first hidden layer
    def modelPhi(self,x):
        return self.layers[0](x)
    
    # Output of the max-operator part
    def modelPsi(self,out):
        for i in range(1,len(self.layers)-1):
            out=self.relu(self.layers[i](out))
        return self.layers[-1](out)
    
    def initialWB(self,fW,fB):
        self.layers[0].bias.data=self.layers[0].bias.data*fB
        self.layers[0].weight.data=self.layers[0].weight.data*fW