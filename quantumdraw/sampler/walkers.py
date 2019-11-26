import torch

class Walkers(object):

    def __init__(self,nwalkers,nelec,ndim,domain):

        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nelec = nelec
        self.domain = domain

        self.pos = None
        self.status = None

    def initialize(self, method='uniform', pos=None):

        if pos is not None:
            if len(pos) > self.nwalkers:
                pos = pos[-self.nwalkers:,:]
            self.pos = pos
        
        else:
            options = ['center','uniform']
            if method not in options:
                raise ValueError('method %s not recognized. Options are : %s ' %(method, ' '.join(options)) )

            if method == options[0]:
                self.pos = torch.zeros((self.nwalkers, self.nelec*self.ndim ))

            elif method == options[1]:
                self.pos = torch.rand(self.nwalkers, self.nelec*self.ndim) 
                self.pos *= (self.domain['max'] - self.domain['min'])
                self.pos += self.domain['min']

        self.status = torch.ones((self.nwalkers,1))

    def move(self,step_size):
        return self.pos + self.status * self._random(step_size,(self.nwalkers,self.nelec * self.ndim))

    def _random(self,step_size,size):
        return step_size * (2 * torch.rand(size) - 1)










