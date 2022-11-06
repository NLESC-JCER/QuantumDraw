import numpy as np 
import torch
from quantumdraw.solver.solver_base import Solver
from scipy import interpolate

class UserSolver(Solver):

    def __init__(self,wf=None, sampler=None, 
                      optimizer=None,scheduler=None):
        super(UserSolver,self).__init__(wf,sampler)

        self.interpolate_solution()

    def interpolate_solution(self):
        self.solinterp = interpolate.interp1d(self.solution['x'],
                                                self.solution['y'],
                                                fill_value='extrapolate')
    def get_score(self):
        
        with torch.no_grad():

            ywf = self.wf.data['y'][1:-1]
            ywf = ( ywf/np.max(ywf) * self.solution['max']).flatten()

            ysol = self.solinterp(self.wf.data['x'][1:-1])
            fill_factor = (self.wf.data['x'][-2]-self.wf.data['x'][1])/10
            return fill_factor*self._score(ywf, ysol)

    def _score(self,yvals, ysol):
    
        d = np.sqrt(np.sum((ysol-yvals)**2))
        return np.exp(-d)

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
        
        # dmax = np.max(np.abs(delta))
        # if dmax > 0.05:
        #     delta /= dmax
        scale = 1.
        
        return {'x':self.solution['x'].tolist(), 'y' : (scale * delta).tolist()}



