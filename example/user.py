import torch
import numpy as np

from schrodinet.sampler.metropolis import  Metropolis
from quantumdraw.server.wavefunction import UserWaveFunction
from quantumdraw.server.solver import UserSolver


def pot_func(pos):
    '''Potential function desired.'''
    return  0.5*pos**2

def gen_pts(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2) #+ 2*torch.exp(-0.75*(pos-2.)**2)

# box
domain = {'min':-5.,'max':5.}

#user wave function
npts = 501
xpts = torch.tensor(np.sort(np.random.rand(npts)*10-5))
ypts = gen_pts(xpts).detach().numpy() + 0.1*np.random.rand(npts)
xpts = xpts.detach().numpy()
uwf = UserWaveFunction(pot_func,domain,xpts=xpts,ypts=ypts)

#sampler
sampler = Metropolis(nwalkers=100, nstep=100, 
                     step_size = 0.5, init = domain)

usolver = UserSolver(wf=uwf, sampler=sampler)
#plot_wf_1d(usolver,domain,51,pot=False,feedback=usolver.feedback())
score = usolver.get_score()
print(score)
