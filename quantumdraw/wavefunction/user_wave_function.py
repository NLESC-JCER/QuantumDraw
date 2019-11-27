import numpy as np 
import torch
from scipy import interpolate
from quantumdraw.wave_function_base import WaveFunction

class UserWaveFunction(WaveFunction):

    def __init__(self,fpot,domain,xpts=None,ypts=None):
        super(UserWaveFunction,self).__init__(fpot,domain)

        # book the potential function
        self.load_data(xpts,ypts)
        self.get_interp()

    def get_interp(self):
        """Creates a function that interpolate the data points.
        """
        if self.data['x'] is not None:
            self.finterp = interpolate.interp1d(self.data['x'],
                                                self.data['y'])
    def load_data(self,x,y):
        """load data points in the class
        
        Args:
            x (array): x coordinates of the points
            y (array): y values of the points
        """
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

        eps = 1E-6
        if out is None:
            out = self.forward(pos)
        xp = self.forward(pos+eps)
        xm = self.forward(pos-eps)

        return -0.5/eps/eps * (xm+xp-2.*out)

    def nuclear_potential(self,pos):
        """Compute the potential of the wf points.

        Args:
            pos (torch.tensor): position of the electron

        Returns: 
            torch.tensor: values of V 
        """
        return self.user_potential(pos).flatten().view(-1,1)       