import torch
import numpy as np
from quantumdraw.sampler.walkers import Walkers
from tqdm import tqdm 

class Metropolis(object):

    def __init__(self, nwalkers=1000, nstep=1000,
                 step_size = 3, domain = {'min':-2,'max':2}):

        ''' METROPOLIS HASTING SAMPLER
        Args:
            f (func) : function to sample
            nstep (int) : number of mc step
            nwalkers (int) : number of walkers
            eps (float) : size of the mc step
            boudnary (float) : boudnary of the space
        '''

        self.nwalkers = nwalkers
        self.nstep = nstep
        self.step_size = step_size
        self.domain = domain
        self.nelec = 1
        self.ndim = 1
        
        self.walkers = Walkers(nwalkers,self.nelec,self.ndim,domain)

    def set_ndim(self,ndim):
        self.ndim = ndim

    def set_initial_guess(self,guess):
        self.initial_guess = guess

    def generate(self,pdf,ntherm=10,with_tqdm=True,pos=None,init='uniform'):

        ''' perform a MC sampling of the function f
        Returns:
            X (list) : position of the walkers
        '''
        with torch.no_grad():
            
            if ntherm < 0:
                ntherm = self.nstep+ntherm

            self.walkers.initialize(method=init,pos=pos)

            #fx = pdf(torch.tensor(self.walkers.pos).float())
            fx = pdf(self.walkers.pos)
            fx[fx==0] = 1E-6
            POS = []
            rate = 0

            if with_tqdm:
                rng = tqdm(range(self.nstep))
            else:
                rng = range(self.nstep)

            for istep in rng:

                # new positions
                #Xn = torch.tensor(self.walkers.move(self.step_size,method=self.move)).float()
                Xn = self.walkers.move(self.step_size)

                # new function
                fxn = pdf(Xn)
                df = (fxn/(fx)).double()
                
                # accept the moves
                index = self._accept(df)
                
                # acceptance rate
                rate += index.byte().sum().float()/self.walkers.nwalkers
                
                # update position/function value
                self.walkers.pos[index,:] = Xn[index,:]
                fx[index] = fxn[index]
                fx[fx==0] = 1E-6
            
                if istep>=ntherm:
                    POS.append(self.walkers.pos.clone().detach())

            if with_tqdm:
                print("Acceptance rate %1.3f %%" % (rate/self.nstep*100) )
            
        return torch.cat(POS)

    
    def _accept(self,P):
        ones = torch.ones(self.nwalkers)
        P[P>1]=1.0
        tau = torch.rand(self.nwalkers).double()
        index = (P-tau>=0).reshape(-1)
        return index.type(torch.bool)

