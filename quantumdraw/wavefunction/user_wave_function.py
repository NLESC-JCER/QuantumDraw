import numpy as np 
import torch
from scipy import interpolate

class UserWaveFunction(object):

    def __init__(self,fpot,domain,data=None):

        # book the potential function
        self.user_potential = fpot
        self.domain = domain
        self.data = data

        self.get_interp()

    def get_interp(self):

        if self.data is not None:
            self.finterp = interpolate.interp1d(self.data['x'],
                                                self.data['y'])

    def forward(self,pos):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            pos: position of the electrons

        Returns: values of psi
        '''

        x = pos.detach().numpy()
        x = self.finterp(x)
        return torch.tensor(x).view(-1,1)


    def kinetic_energy(self,pos,out=None):
        '''Compute the second derivative of the network
        output w.r.t the value of the input. 

        This is to compute the value of the kinetic operator.

        Args:
            pos: position of the electron
            eps : argument for the finite difference calc

        Returns:
            values of nabla^2 * Psi
        '''
        eps = 1E-6
        if out is None:
            out = self.forward(pos)
        xp = self.forward(pos+eps)
        xm = self.forward(pos-eps)

        return -0.5/eps/eps * (xm+xp-2.*out)

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi
        '''
        return self.user_potential(pos).flatten().view(-1,1)       