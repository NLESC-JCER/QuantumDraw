import numpy as np
import torch

# make sure the functions can be called either with a single value, or
# with a Torch tensor similar to an array of x-values.

def level1(pos):
    if isinstance(pos, torch.Tensor):
        v = 0.1*torch.ones_like(pos)
        v[pos<-2.5] = 5.
        v[pos>2.5] = 5.

        return v
    else:
        return .1 if np.abs(pos) < 2.5 else 5.

potentials = [
    level1,
    lambda pos: 0.5*pos**2
]