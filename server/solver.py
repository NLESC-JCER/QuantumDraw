import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from scipy import interpolate

from types import SimpleNamespace
import inspect

from tqdm import tqdm
import time

from schrodinet.solver.solver_base import SolverBase
from schrodinet.solver.solver_potential import SolverPotential

class QuantumDrawSolver(SolverBase):

    def __init__(self, wf=None, sampler=None, optimizer=None,scheduler=None):
        super(QauntumDrawSolver,self).__init__(wf,sampler,optimizer,scheduler)

        # observalbe
        self.observable(['local_energy'])

        # get the solution via fd
        self.solution = self.get_solution()

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
        """Get the score of the current solution."""

        with torch.no_grad():

            ywf = self.wf(torch.tensor(self.solution['x'])).clone().detach().numpy()
            ywf = (ywf/np.max(ywf) * self.solution['max']).flatten()
            return self._score(ywf)

    def _score(self,yvals):
        """Scoring function."""
        d = np.sqrt(np.sum((self.solution['y']-yvals)**2))
        return np.exp(-d)

class NeuralSolver(QuantumDrawSolver, SolverPotential):

    def __init__(self,wf=None, sampler=None, optimizer=None,scheduler=None):
        """Solver for the neural network wave function."""
        QuantumDrawSolver.__init__(wf,sampler,optimizer,scheduler)
        SolverPotential.__init__(wf,sampler,optimizer,scheduler)


class UserSolver(QuantumDrawSolver):

    def __init__(self,wf=None, sampler=None, optimizer=None,scheduler=None):
        """Solver for the user defined wave function."""
        super(UserSolver,self).__init__(wf,sampler)
        self.interpolate_solution()

    def interpolate_solution(self):
        self.solinterp = interpolate.interp1d(self.solution['x'], self.solution['y'],
                                              fill_value='extrapolate')

    def feedback(self):
        """Returns the feedback to the user
        
        Returns:
            dict: x and y value of the feedback curve
        """
        yuser = self.wf.finterp(self.solution['x'])

        yuser_scale = np.copy(yuser)
        yuser_scale /= np.max(yuser_scale)
        yuser_scale *= self.solution['max']

        delta = (self.solution['y']-yuser_scale)
        scale = 1.
        
        return {'x':self.solution['x'].tolist(), 'y' : (scale * delta).tolist()}
