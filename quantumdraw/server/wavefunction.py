import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

from scipy import interpolate

from schrodinet.wavefunction.wf_base import WaveFunction
from schrodinet.wavefunction.wf_potential import Potential
from schrodinet.wavefunction.rbf import RBF_Gaussian
import time

class RBF(RBF_Gaussian):

    def __init__(self, input_features, output_features, centers,
                opt_centers=True, sigma = 1.0, opt_sigma= True):

        '''Radial Basis Function Layer in N dimension

        Args:
            input_features: input side
            output_features: output size
            centers : position of the centers
            opt_centers : optmize the center positions
            sigma : strategy to get the sigma
            opt_sigma : optmize the std or not
        '''

        super(RBF,self).__init__(input_features, output_features, centers, opt_centers, sigma , opt_sigma)

    
    def forward(self,input,der=0):
        '''Compute the output of the RBF layer'''
        
        delta =  (input[:,None,:] - self.centers[None,...])
        X = ( delta**2 ).sum(2)
        eps = 1E-6

        if der == 0:
            X = torch.exp(-X/(self.sigma+eps))
        elif der == 2 :
            X = (4*X/(self.sigma+eps)**2-2./(self.sigma+eps)) * torch.exp(-X/(self.sigma+eps))

        return X.view(-1,self.ncenter)

class NeuralWaveFunction(Potential):

    def __init__(self,fpot,domain,ncenter,fcinit=0.1,sigma=1.):
        """neural network RBF wave function."""

        Potential.__init__(self,fpot, domain, ncenter)
        self.domain = domain
        self.kinetic_energy = self.kinetic_energy_analytical

        self.rbf = RBF(self.ndim_tot, self.ncenter,
                       centers=self.centers, sigma=sigma,
                       opt_centers=True,
                       opt_sigma=True)

    def kinetic_energy_analytical(self,pos,out=None):
        """Fast calculation of the kinetic energy."""
        x = self.rbf(pos,der=2)
        x = self.fc(x)
        return -0.5*x.view(-1,1)

class UserWaveFunction(WaveFunction):

    def __init__(self ,fpot, domain, xpts=None, ypts=None):
        super(UserWaveFunction,self).__init__(1,1)
        self.user_potential = fpot
        self.domain = domain
        
        # book the potential function
        self.load_data(xpts,ypts)
        self.get_interp()

    def __call__(self,pos):
        return self.forward(pos)

    def get_interp(self):
        """Creates a function that interpolate the data points.
        """
        if self.data['x'] is not None:
            self.finterp = interpolate.interp1d(self.data['x'],
                                                self.data['y'],
                                                fill_value='extrapolate')

    def get_spline(self):
        if self.data['x'] is not None:
            self.finterp = interpolate.UnivariateSpline(self.data['x'],self.data['y'],k=5)
            self.finterp_kin = self.finterp.derivative(n=2)

    def load_data(self,x,y):
        """load data points in the class
        
        Args:
            x (array): x coordinates of the points
            y (array): y values of the points
        """
        
        x = np.insert(x,0,1.25*self.domain['min'])
        y = np.insert(y,0,0)

        x = np.insert(x,len(x),1.25*self.domain['max'])
        y = np.insert(y,len(y),0)

        self.data = {'x':[],'y':[]}
        self.data['x'] = x
        self.data['y'] = y

    def forward(self,pos):
        """Compute the value of the wave function.
        for a multiple conformation of the electrons
        
        Args:
            pos (torch.tensor): positions of the particle
        
        Returns:
            torch.tensor: value of the wave function
        """

        x = pos.detach().numpy()
        x = self.finterp(x)
        return torch.tensor(x).view(-1,1)


    def kinetic_energy(self,pos,out=None):
        """Compute the second derivative of the network
           output w.r.t the value of the input. 
        
        Args:
            pos (torch.tensor): position of the particle
            out (torch.tensor, optional): precomputed values of the wf
                Defaults to None.
        
        Returns:
            torch.tensor: values of the kinetic energy
        """
        _spl_ = False
        if _spl_:
            K = torch.tensor(-0.5*self.finterp_kin(pos.detach().numpy()))
        else:
            eps = 5*1E-2
            if out is None:
                out = self.forward(pos)

            xp = self.forward(pos+eps)
            xm = self.forward(pos-eps)
            K = -0.5 / eps / eps * (xm+xp-2.*out)

        return K.view(-1,1)

    def nuclear_potential(self,pos):
        """Compute the potential of the wf points.

        Args:
            pos (torch.tensor): position of the electron

        Returns: 
            torch.tensor: values of V 
        """
        return self.user_potential(pos).flatten().view(-1,1)       



