import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

from quantumdraw.wavefunction.rbf import RBF


class WaveFunction(object):

    def __init__(self,fpot,domain):
        
        self.user_potential = fpot
        self.domain = domain
        

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            pos: position of the electrons

        Returns: values of psi
        '''
        raise NotImplementedError('forward method not implemented yet')

    def kinetic_energy(self,pos,out=None):
        '''Compute the second derivative of the network
        output w.r.t the value of the input. 

        This is to compute the value of the kinetic operator.

        Args:
            pos: position of the electron
            out : preomputed values of the wf at pos

        Returns:
            values of nabla^2 * Psi
        '''
        raise NotImplementedError('forward method not implemented yet')


    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi
        '''
        return self.user_potential(pos).flatten().view(-1,1)

    def local_energy(self,pos):
        ''' local energy of the sampling points.'''
        
        wf = self.forward(pos)
        ke = self.kinetic_energy(pos,out=wf)
        
        return ke/wf + self.nuclear_potential(pos)

    def energy(self,pos):
        '''Total energy for the sampling points.'''
        return torch.mean(self.local_energy(pos)) 

    def variance(self, pos):
        '''Variance of the energy at the sampling points.'''
        return torch.var(self.local_energy(pos))

    def _energy_variance(self,pos):
        el = self.local_energy(pos)
        return torch.mean(el), torch.var(el)

    def pdf(self,pos):
        '''density of the wave function.'''
        return (self.forward(pos)**2).reshape(-1)








