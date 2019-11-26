import torch
from torch import nn
import torch.nn.functional as F
from math import pi as PI

class RBF(nn.Module):

    def __init__(self,
                input_features,
                output_features,
                centers,
                opt_centers=True,
                sigma = 1.0,
                opt_sigma= False ):

        '''Radial Basis Function Layer in N dimension

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF,self).__init__()

        # register dimension
        self.input_features = input_features
        self.output_features = output_features

        # make the centers optmizable or not
        self.centers = nn.Parameter(torch.Tensor(centers))
        self.ncenter = len(self.centers)
        self.centers.requires_grad = opt_centers


        # get the standard deviation
        self.sigma = nn.Parameter(sigma*torch.ones(self.ncenter))
        self.sigma.requires_grad = opt_sigma
        
    def forward(self,input):
        '''Compute the output of the RBF layer'''

        # get the distancese of each point to each RBF center
        # (Nbatch,Nrbf,Ndim)
        delta =  (input[:,None,:] - self.centers[None,...])

        # Compute (INPUT-MU).T x Sigma^-1 * (INPUT-MU)-> (Nbatch,Nrbf)
        X = ( delta**2 ).sum(2)

        # divide by the determinant of the cov mat
        X = torch.exp(-X/self.sigma)

        return X.view(-1,self.ncenter)




