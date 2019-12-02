import numpy as np 
import torch
from quantumdraw.solver.solver_base import Solver


class UserSolver(Solver):

    def __init__(self,wf=None, sampler=None, 
                      optimizer=None,scheduler=None):
        super(UserSolver,self).__init__(wf,sampler)
        
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
        dmax = np.max(np.abs(delta))
        print(dmax)
        if dmax > 0.25:
            delta /= dmax
        scale = 1

        return {'x':self.solution['x'],'y': yuser, 'delta' : scale*delta}



