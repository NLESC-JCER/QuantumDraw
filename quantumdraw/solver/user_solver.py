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
        scale = 0.75
        
        return {'x':self.solution['x'].tolist(), 'y' : (scale * delta).tolist()}

    def feedback_v2(self):
        '''returns the segment of pos/neg values'''

        # user data
        x = self.wf.data['x'][1:-2].copy()
        y = self.wf.data['y'][1:-2].copy()
        try:
            yuser_scale = y / np.max(y)
        except:
            yuser_scale = y

        # solution at those points
        ysol = self.solinterp(x)

        # get index of +/- segments
        idx_pos = np.where(yuser_scale > ysol)
        idx_neg = np.where(yuser_scale < ysol)

        return {'x':x[idx_pos].tolist(), 'y': y[idx_pos].tolist()}, {'x':x[idx_neg].tolist(), 'y': y[idx_neg].tolist()} 


