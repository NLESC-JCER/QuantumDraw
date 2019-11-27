import numpy as np 
import torch
from quantumdraw.solver.solver_base import Solver


class UserSolver(Solver):

    def __init__(self,wf=None, sampler=None, 
                      optimizer=None,scheduler=None):
        super(UserSolver,self).__init__(wf,sampler)
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
        K = -0.5 / dx2 * ( np.eye(npts,k=1) + np.eye(npts,k=-1) - 2. * np.eye(npts))
        l, U = np.linalg.eigh(K+Vx)
        return {'x':x.detach().numpy(),'y':U[:,0],'max':np.max(U[:,0])}

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
        delta /= np.max(np.abs(delta))
        scale = 0.25

        return {'x':self.solution['x'],'y': yuser, 'delta' : scale*delta}



