import torch
import numpy as np
from quantumdraw.sampler.walkers import Walkers
from tqdm import tqdm 

class Metropolis(object):

    def __init__(self, nwalkers=1000, nstep=1000,
                 step_size = 3, domain = {'min':-2,'max':2}):

        """Metropolis hasting sampling.

        Args:
            nwalkers (int) : number of walkers
            nstep (int) : number of mc step
            step_size (float) : size of the mc step
            domain (dict) : boundary of the space
        """

        self.nwalkers = nwalkers
        self.nstep = nstep
        self.step_size = step_size
        self.domain = domain
        self.nelec = 1
        self.ndim = 1
        
        self.walkers = Walkers(nwalkers,self.nelec,self.ndim,domain)

    def generate(self,pdf,ntherm=10,with_tqdm=True,pos=None,init='uniform'):


        """Generates a series of point sampling the function pdf
        
        Args :
            pdf (function) : pdf of the function to sample
            ntherm (int) : number of step to skip
            with_tqdm (bool) : use tqdm progess bar
            pos (torch.tensor) : starting position of the walkers
            init (str) : method to initialize the walkers

        Returns:
            X (list) : position of the walkers
        """
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

