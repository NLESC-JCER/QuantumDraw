import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from quantumdraw.sampler.metropolis import  Metropolis
from quantumdraw.wavefunction.neural_wave_function import NeuralWaveFunction
from quantumdraw.solver.neural_solver import NeuralSolver
from quantumdraw.solver.plot_utils import plot_results_1d, plotter1d, plot_wf_1d

def pot_func(pos):
    '''Potential function desired.'''
    return  0.5*pos**2

def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2)

# box
domain, ncenter = {'xmin':-5.,'xmax':5.}, 11

#sampler
sampler = Metropolis(nwalkers=1000, nstep=2000, 
                     step_size = 0.5, domain = domain)

# wavefunction
wf = NeuralWaveFunction(pot_func,domain,ncenter,fcinit='random',sigma=0.5)

# optimizer
opt = optim.Adam(wf.parameters(),lr=0.05)

# scheduler
scheduler = optim.lr_scheduler.StepLR(opt,step_size=100,gamma=0.75)

# define solver
solver = NeuralSolver(wf=wf,sampler=sampler,optimizer=opt,scheduler=scheduler)
#plot_wf_1d(solver,domain,51,sol=ho1d_sol)
#pos,e,v = solver.single_point()



# train the wave function
plotter = plotter1d(wf,domain,100,sol=ho1d_sol)#,save='./image/')
solver.run(75,loss = 'variance', plot = None, save='model.pth' )

# plot the final wave function 
plot_results_1d(solver,domain,100,ho1d_sol,e0=0.5,load='model.pth')

