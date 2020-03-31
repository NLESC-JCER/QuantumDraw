import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from scipy import interpolate

from types import SimpleNamespace
import inspect

from tqdm import tqdm
import time

from schrodinet.solver.torch_utils import DataSet, Loss, ZeroOneClipper
from schrodinet.solver.solver_base import Solver


class NeuralSolver(Solver):

    def __init__(self,wf=None, sampler=None, 
                      optimizer=None,scheduler=None):
        super(NeuralSolver,self).__init__(wf,sampler)
        self.opt = optimizer  
        self.scheduler = scheduler

        #esampling
        self.resampling(ntherm=-1, resample=100,
                        resample_from_last=True, 
                        resample_every=1)

        # observalbe
        self.observable(['local_energy'])

    def resampling(self,ntherm=-1, resample=100, 
                        resample_from_last=True, 
                        resample_every=1):
        '''Configure the resampling options.'''
        self.resample = SimpleNamespace()
        self.resample.ntherm = ntherm
        self.resample.resample = resample
        self.resample.resample_from_last = resample_from_last
        self.resample.resample_every = resample_every

    def save_checkpoint(self,epoch,loss,filename):
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : self.wf.state_dict(),
            'optimzier_state_dict' : self.opt.state_dict(),
            'loss' : loss
            }, filename)
        return loss

    def run(self, nepoch, pos = None, batchsize=None, 
            save='model.pth',  loss='variance', plot = None,
            with_tqdm=True):

        '''Train the model.

        Arg:
            nepoch : number of epoch
            batchsize : size of the minibatch, if None take all points at once
            pos : presampled electronic poition
            obs_dict (dict, {name: []} ) : quantities to be computed during the training
                                           'name' must refer to a method of the Solver instance
            ntherm : thermalization of the MC sampling. If negative (-N) takes the last N entries
            resample : number of MC step during the resampling
            resample_from_last (bool) : if true use the previous position as starting for the resampling
            resample_every (int) : number of epch between resampling
            loss : loss used ('energy','variance' or callable (for supervised)
            plot : None or plotter instance from plot_utils.py to interactively monitor the training
        '''

        # checkpoint file
        self.save_model = save

        # sample the wave function
        pos = self.sample(pos=pos, ntherm=self.resample.ntherm, with_tqdm=with_tqdm)

        # determine the batching mode
        if batchsize is None:
            batchsize = len(pos)

        # change the number of steps
        _nstep_save = self.sampler.nstep
        self.sampler.nstep = self.resample.resample

        # create the data loader
        self.dataset = DataSet(pos)
        self.dataloader = DataLoader(self.dataset,batch_size=batchsize)

        # get the loss
        self.loss = Loss(self.wf,method=loss)
                
        # clipper for the fc weights
        clipper = ZeroOneClipper()
    
        cumulative_loss = []
        

        for n in range(nepoch):
            #print('----------------------------------------')
            #print('epoch %d' %n)

            cumulative_loss = 0
            for data in self.dataloader:
                
                lpos = Variable(data).float()
                lpos.requires_grad = True

                loss = self.loss(lpos)
                cumulative_loss += loss

                if torch.isnan(loss):
                    print('ooops ran into an issue')
                    continue

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                self.check_parameters()

                if self.wf.fc.clip:
                    self.wf.fc.apply(clipper)
                
            if plot is not None:
                plot.drawNow()

            # if cumulative_loss < min_loss:
            #     min_loss = self.save_checkpoint(n,cumulative_loss,self.save_model)
                 
            # get the observalbes
            # self.get_observable(self.obs_dict,pos)
            # print('loss %f' %(cumulative_loss))
            # print('variance : %f' %np.var(self.obs_dict['local_energy'][-1]))
            # print('energy : %f' %np.mean(self.obs_dict['local_energy'][-1]) )   
            print('score : %f' %self.get_score() )   
            # print('----------------------------------------')
            
            # resample the data
            if (n%self.resample.resample_every == 0): # or (n == nepoch-1):
                if self.resample.resample_from_last:
                    pos = pos.clone().detach()
                else:
                    pos = None
                pos = self.sample(pos=pos,ntherm=self.resample.ntherm,with_tqdm=False)
                self.dataloader.dataset.data = pos

            if self.scheduler is not None:
                self.scheduler.step()

        #restore the sampler number of step
        self.sampler.nstep = _nstep_save

        return pos

    def check_parameters(self):

        check_nan =  torch.isnan(self.wf.rbf.centers.data)
        self.wf.rbf.centers.data[check_nan] = 0.

        check_nan =  torch.isnan(self.wf.rbf.sigma.data)
        self.wf.rbf.sigma.data[check_nan] = 1.0

        check_nan =  torch.isnan(self.wf.fc.weight.data)
        self.wf.fc.weight.data[check_nan] = 0.


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
        scale = 1.
        
        return {'x':self.solution['x'].tolist(), 'y' : (scale * delta).tolist()}
