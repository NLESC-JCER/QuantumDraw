import torch
from torch import optim
import time
import numpy as np

from schrodinet.sampler.metropolis import Metropolis
from quantumdraw.server.solver import NeuralSolver, UserSolver
from quantumdraw.server.wavefunction import NeuralWaveFunction, UserWaveFunction

domain = {'min': -5., 'max': 5.}


def get_user_score(user_guess, current_pot):
    uwf = UserWaveFunction(current_pot, domain, xpts=list(map(lambda g: g[0], user_guess)), ypts=list(map(lambda g: g[1], user_guess)))

    # sampler
    sampler = Metropolis(nwalkers=100, nstep=100,
                         step_size=0.5, init=domain)

    usolver = UserSolver(wf=uwf, sampler=sampler)
    data = usolver.feedback()
    points = list(zip(data['x'],data['y']))
        
    return points, usolver.get_score()
    
def get_solution(current_pot):
    uwf = UserWaveFunction(current_pot, domain, xpts=[], ypts=[])

    # sampler
    sampler = Metropolis(nwalkers=100, nstep=100,
                         step_size=0.5, init=domain)

    usolver = UserSolver(wf=uwf, sampler=sampler)
    data = usolver.get_solution()
    data['y'] /= np.max(data['y'])
    points = list(zip(data['x'].tolist(),data['y'].tolist()))
        
    return points

def get_ai_score(current_pot, max_iterations=100, duration=30):
    sampler = Metropolis(nwalkers=500, nstep=2000,
                         step_size=0.5, init=domain)

    # wavefunction
    wf = NeuralWaveFunction(current_pot, domain, 11, fcinit='random', sigma=0.5)

    # optimizer
    opt = optim.Adam(wf.parameters(), lr=0.05)

    # scheduler
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=100, gamma=0.75)

    # define solver
    solver = NeuralSolver(wf=wf, sampler=sampler, optimizer=opt, scheduler=scheduler)
    # solver = NeuralSolver(wf=wf, sampler=sampler, optimizer=None, scheduler=None)

    pos = solver.run(1)
    solver.sampler.nstep = solver.resample.resample

    end_time = time.time() + duration
    while time.time() < end_time:
        pos = solver.run(1, pos=pos, with_tqdm=False)
        num_samples = 50
        low_x = -5
        high_x = 5
        x_points = [low_x + sample_num * (high_x - low_x) / num_samples for sample_num in range(0, num_samples)]
        x = torch.tensor(x_points).view(-1,1)
        y = solver.wf(x)
        y /= y.max()
        y_points = y.detach().numpy().T[0].tolist()
        points = list(zip(x_points, y_points))
        score = solver.get_score()
        yield points, score

    
