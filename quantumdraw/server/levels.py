import numpy as np
import torch

# make sure the functions can be called either with a single value, or
# with a Torch tensor similar to an array of x-values.

def level1(pos):
    '''harmonic oscillator.'''
    return pos**2

def level2(pos):
    '''morse oscillator.'''
    if isinstance(pos, torch.Tensor):
        return 0.25*(torch.exp(-2.*(pos+2)) - 2.*torch.exp(-(pos+2))).view(-1, 1) 
    else:
        return 0.25*(np.exp(-2.*(pos+2)) - 2.*np.exp(-(pos+2))) 
        
def level3(pos):
    '''Double well'''
    h = 0.75
    c = 0.05
    return -0.25*pos**2*h + 0.5 * c*pos**4

def level4(pos):
    ''' infinite well.'''
    if isinstance(pos, torch.Tensor):
        v = 0.1*torch.ones_like(pos)
        v[pos<-2.5] = 5.
        v[pos>2.5] = 5.

        return v
    else:
        return .1 if np.abs(pos) < 2.5 else 5.


def level5(pos):
    '''H2+ type of thing.'''
    k=0.75*1E-1
    if isinstance(pos,torch.Tensor):
        return -k/torch.abs(pos-2.5) + -k/torch.abs(pos+2.5)
    else:
        return -k/np.abs(pos-2.5) + -k/np.abs(pos+2.5)

potentials = [
    level1,
    level2,
    level5
]