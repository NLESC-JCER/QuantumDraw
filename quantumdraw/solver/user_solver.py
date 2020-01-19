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
        
        # thr = 0.15
        # delta[delta>thr] = thr
        # delta[delta<-thr] = -thr
        
        dmax = np.max(np.abs(delta))
        delta /= dmax
        scale = 0.15
        
        return {'x':self.solution['x'].tolist(), 'y' : (scale * delta).tolist()}



