import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad, Variable

from quantumdraw.wavefunction.wave_function_base import WaveFunction
from quantumdraw.wavefunction.rbf import RBF


class NeuralWaveFunction(nn.Module,WaveFunction):

    def __init__(self,fpot,domain,ncenter,fcinit=0.1,sigma=1.):

        #super(WaveFunction,self).__init__()
        WaveFunction.__init__(self,fpot,domain)
        nn.Module.__init__(self)

        self.ndim = 1
        self.nelec = 1
        self.ndim_tot = self.nelec*self.ndim
        

        # get the RBF centers 
        if not isinstance(ncenter,list):
            ncenter = [ncenter]
        self.centers = torch.linspace(domain['xmin'],domain['xmax'],ncenter[0]).view(-1,1)
        self.ncenter = ncenter[0]

        # define the RBF layer
        self.rbf = RBF(self.ndim_tot, self.ncenter,
                      centers=self.centers, sigma = sigma,
                      opt_centers=True,
                      opt_sigma = True)
        
        # define the fc layer
        self.fc = nn.Linear(self.ncenter, 1, bias=False)
        self.fc.clip = True

        # initiaize the fc layer
        if fcinit == 'random':
            nn.init.uniform_(self.fc.weight,0,1)
        elif isinstance(fcinit,float):  
            self.fc.weight.data.fill_(fcinit)

    def forward(self,x):
        ''' Compute the value of the wave function.
        for a multiple conformation of the electrons

        Args:
            parameters : variational param of the wf
            pos: position of the electrons

        Returns: values of psi
        '''
        x = self.rbf(x)
        x = self.fc(x)
        return x.view(-1,1)

    def nuclear_potential(self,pos):
        '''Compute the potential of the wf points
        Args:
            pos: position of the electron

        Returns: values of V * psi
        '''
        return self.user_potential(pos).flatten().view(-1,1)

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

        if out is None:
            out = self.forward(pos)
        
        # compute the jacobian            
        z = Variable(torch.ones(out.shape))
        jacob = grad(out,pos,
                     grad_outputs=z,
                     only_inputs=True,
                     create_graph=True)[0]
        
        # compute the diagonal element of the Hessian
        z = Variable(torch.ones(jacob.shape[0]))
        hess = torch.zeros(jacob.shape[0])
        
        for idim in range(jacob.shape[1]):

            tmp = grad(jacob[:,idim],pos,
                      grad_outputs=z,
                      only_inputs=True,
                      #retain_graph=True)[0]
                      create_graph=True)[0] # create_graph is REQUIRED and is causing memory issues for large systems
                      #allow_unused=True)[0]    
              
            hess += tmp[:,idim]
        
        return -0.5 * hess.view(-1,1) 
    
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







