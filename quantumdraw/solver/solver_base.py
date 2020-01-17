import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from types import SimpleNamespace
import inspect

from quantumdraw.solver.torch_utils import DataSet, Loss, ZeroOneClipper

from tqdm import tqdm
import time


class Solver(object):

    def __init__(self,wf=None, sampler=None):

        self.wf = wf
        self.sampler = sampler

        # observalbe
        self.observable(['local_energy'])

        # get the solution via fd
        self.solution = self.get_solution()

    def observable(self,obs):
        '''Create the observalbe we want to track.'''
        self.obs_dict = {}
        
        for k in obs:
            self.obs_dict[k] = []

        required_obs = ['local_energy']
        for k in required_obs:
            if k not in self.obs_dict:
                self.obs_dict[k] = []
            
    def sample(self, ntherm=-1,with_tqdm=True,pos=None):
        ''' sample the wave function.'''
        
        pos = self.sampler.generate(self.wf.pdf,ntherm=ntherm,with_tqdm=with_tqdm,pos=pos)
        pos.requires_grad = True
        return pos.float()

    def get_observable(self,obs_dict,pos,**kwargs):
        '''compute all the required observable.

        Args :
            obs_dict : a dictionanry with all keys 
                        corresponding to a method of self.wf
            **kwargs : the possible arguments for the methods
        TODO : match the signature of the callables
        '''

        for obs in self.obs_dict.keys():

            # get the method
            func = self.wf.__getattribute__(obs)
            data = func(pos)
            if isinstance(data,torch.Tensor):
                data = data.detach().numpy()
            self.obs_dict[obs].append(data)

    def get_wf(self,x):
        '''Get the value of the wave functions at x.'''
        vals = self.wf(x)
        return vals.detach().numpy().flatten()

    def energy(self,pos=None):
        '''Get the energy of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)
        return self.wf.energy(pos)

    def variance(self,pos):
        '''Get the variance of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)
        return self.wf.variance(pos)

    def single_point(self,pos=None,prt=True):
        '''Performs a single point calculation.'''
        if pos is None:
            pos = self.sample(ntherm=-1)

        e,s = self.wf._energy_variance(pos)
        if prt:
            print('Energy   : ',e)
            print('Variance : ',s)
        return pos, e, s

    def get_solution(self,npts=100):

        """Computes the solution using finite difference
        
        Args:
            npts (int, optional): number of discrete points. Defaults to 100.

        Returns:
            dict: position and numerical value of the wave function 
        """

        x = torch.linspace(self.wf.domain['xmin'],self.wf.domain['xmax'],npts)
        dx2 = (x[1]-x[0])**2
        Vx = np.diag(self.wf.nuclear_potential(x).detach().numpy().flatten())
        K = -0.5 / dx2.numpy() * ( np.eye(npts,k=1) + np.eye(npts,k=-1) - 2. * np.eye(npts))
        l, U = np.linalg.eigh(K+Vx)
        sol = np.abs(U[:,0])
        return {'x':x.detach().numpy(),'y':sol,'max':np.max(sol)}

    def get_score(self):
        
        with torch.no_grad():

            ywf = self.wf(torch.tensor(self.solution['x'])).clone().detach().numpy()
            ywf = (ywf/np.max(ywf) * self.solution['max']).flatten()
            return self._score(ywf)

    def _score(self,yvals):
    
        d = np.sqrt(np.sum((self.solution['y']-yvals)**2))
        return np.exp(-d)



