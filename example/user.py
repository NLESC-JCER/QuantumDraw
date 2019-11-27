import torch
from torch import optim
import numpy as np

from quantumdraw.sampler.metropolis import  Metropolis
from quantumdraw.wavefunction.user_wave_function import UserWaveFunction
from quantumdraw.solver.user_solver import UserSolver
from quantumdraw.solver.plot_utils import plot_wf_1d

def pot_func(pos):
    '''Potential function desired.'''
    return  0.5*pos**2

def gen_pts(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2) + 2*torch.exp(-0.75*(pos-2.)**2)

# box
domain = {'xmin':-5.,'xmax':5.}

#user wave function
xpts = torch.tensor(np.sort(np.random.rand(25)*10-5))
ypts = gen_pts(xpts).detach().numpy()
xpts = xpts.detach().numpy()
uwf = UserWaveFunction(pot_func,domain,xpts=xpts,ypts=ypts)

#sampler
sampler = Metropolis(nwalkers=100, nstep=100, 
                     step_size = 0.5, domain = domain)

usolver = UserSolver(wf=uwf,sampler=sampler)
plot_wf_1d(usolver,domain,51,feedback=usolver.feedback())
pos,e,v = usolver.single_point()


