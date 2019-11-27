import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from quantumdraw.sampler.metropolis import  Metropolis
from quantumdraw.wavefunction.neural_wave_function import NeuralWaveFunction
from quantumdraw.wavefunction.user_wave_function import UserWaveFunction
from quantumdraw.solver.neural_solver import NeuralSolver
from quantumdraw.solver.user_solver import UserSolver
from quantumdraw.solver.plot_utils import plot_results_1d, plotter1d, plot_wf_1d

def pot_func(pos):
    '''Potential function desired.'''
    return  0.5*pos**2

def ho1d_sol(pos):
    '''Analytical solution of the 1D harmonic oscillator.'''
    return torch.exp(-0.5*pos**2) + 2*torch.exp(-0.75*(pos-2.)**2)

# box
domain, ncenter = {'xmin':-5.,'xmax':5.}, 11


#user wave function
xpts = torch.tensor(np.sort(np.random.rand(25)*10-5))
ypts = ho1d_sol(xpts).detach().numpy()
xpts = xpts.detach().numpy()
uwf = UserWaveFunction(pot_func,domain,xpts=xpts,ypts=ypts)

# wavefunction
#wf = NeuralWaveFunction(pot_func,domain,ncenter,fcinit='random',sigma=0.5)

#sampler
sampler = Metropolis(nwalkers=1000, nstep=2000, 
                     step_size = 0.5, domain = domain)

usolver = UserSolver(wf=uwf,sampler=sampler)
plot_wf_1d(usolver,domain,51,grad=False,sol=None,feedback=usolver.feedback())
#pos,e,v = usolver.single_point()

#plt.hist(pos.detach().numpy())
#plt.show()  



# # optimizer
# opt = optim.Adam(wf.parameters(),lr=0.05)

# # scheduler
# scheduler = optim.lr_scheduler.StepLR(opt,step_size=100,gamma=0.75)

# # define solver
# solver = NeuralSolver(wf=wf,sampler=sampler,optimizer=opt,scheduler=scheduler)

# # train the wave function
# plotter = plotter1d(wf,domain,100,sol=ho1d_sol)#,save='./image/')
# solver.run(300,loss = 'variance', plot = plotter,save='model.pth' )

# # plot the final wave function 
# plot_results_1d(solver,domain,100,ho1d_sol,e0=0.5,load='model.pth')

