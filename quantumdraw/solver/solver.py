import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader

from types import SimpleNamespace
import inspect

from quantumdraw.solver.torch_utils import DataSet, Loss, ZeroOneClipper

from tqdm import tqdm
import time

class Solver(object):

    def __init__(self,wf=None, sampler=None, optimizer=None,scheduler=None):

        self.wf = wf
        self.sampler = sampler
        self.opt = optimizer  
        self.scheduler = scheduler

        #esampling
        self.resampling(ntherm=-1, resample=100,
                        resample_from_last=True, 
                        resample_every=1)

        # observalbe
        self.observable(['local_energy'])

    def resampling(self,ntherm=-1, resample=100,resample_from_last=True, resample_every=1):
        '''Configure the resampling options.'''
        self.resample = SimpleNamespace()
        self.resample.ntherm = ntherm
        self.resample.resample = resample
        self.resample.resample_from_last = resample_from_last
        self.resample.resample_every = resample_every

    def observable(self,obs):
        '''Create the observalbe we want to track.'''
        self.obs_dict = {}
        
        for k in obs:
            self.obs_dict[k] = []

        if 'local_energy' not in self.obs_dict:
            self.obs_dict['local_energy'] = []
            
    def sample(self, ntherm=-1,with_tqdm=True,pos=None):
        ''' sample the wave function.'''
        
        pos = self.sampler.generate(self.wf.pdf,ntherm=ntherm,with_tqdm=with_tqdm,pos=pos)
        pos.requires_grad = True
        return pos.float()

    def get_observable(self,obs_dict,pos,**kwargs):
        '''compute all the required observable.

        Args :
            obs_dict : a dictionanry with all keys 
                        corresponding to a method of self.wf
            **kwargs : the possible arguments for the methods
        TODO : match the signature of the callables
        '''

        for obs in self.obs_dict.keys():

            # get the method
            func = self.wf.__getattribute__(obs)
            data = func(pos)
            if isinstance(data,torch.Tensor):
                data = data.detach().numpy()
            self.obs_dict[obs].append(data)

    def get_wf(self,x):
        '''Get the value of the wave functions at x.'''
        vals = self.wf(x)
        return vals.detach().numpy().flatten()

    def energy(self,pos=None):
        '''Get the energy of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)
        return self.wf.energy(pos)

    def variance(self,pos):
        '''Get the variance of the wave function.'''
        if pos is None:
            pos = self.sample(ntherm=-1)
        return self.wf.variance(pos)

    def single_point(self,pos=None,prt=True):
        '''Performs a single point calculation.'''
        if pos is None:
            pos = self.sample(ntherm=-1)

        e,s = self.wf._energy_variance(pos)
        if prt:
            print('Energy   : ',e)
            print('Variance : ',s)
        return pos, e, s

    def save_checkpoint(self,epoch,loss,filename):
        torch.save({
            'epoch' : epoch,
            'model_state_dict' : self.wf.state_dict(),
            'optimzier_state_dict' : self.opt.state_dict(),
            'loss' : loss
            }, filename)
        return loss

    def run(self,nepoch, batchsize=None, save='model.pth',  loss='variance', plot = None):

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
        pos = self.sample(ntherm=self.resample.ntherm)

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
        min_loss = 1E3

        for n in range(nepoch):
            print('----------------------------------------')
            print('epoch %d' %n)

            cumulative_loss = 0
            for data in self.dataloader:
                
                lpos = Variable(data).float()
                lpos.requires_grad = True

                loss = self.loss(lpos)
                cumulative_loss += loss

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if self.wf.fc.clip:
                    self.wf.fc.apply(clipper)
                
            if plot is not None:
                plot.drawNow()

            if cumulative_loss < min_loss:
                min_loss = self.save_checkpoint(n,cumulative_loss,self.save_model)
                 
            # get the observalbes
            self.get_observable(self.obs_dict,pos)
            print('loss %f' %(cumulative_loss))
            print('variance : %f' %np.var(self.obs_dict['local_energy'][-1]))
            print('energy : %f' %np.mean(self.obs_dict['local_energy'][-1]) )   
            print('----------------------------------------')
            
            # resample the data
            if (n%self.resample.resample_every == 0) or (n == nepoch-1):
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
