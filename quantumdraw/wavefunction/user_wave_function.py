import numpy as np 
import torch
from scipy import interpolate

from quantumdraw.wavefunction.wave_function_base import WaveFunction

class UserWaveFunction(WaveFunction):

    def __init__(self,fpot,domain,xpts=None,ypts=None):
        super(UserWaveFunction,self).__init__(fpot,domain)

        # book the potential function
        self.load_data(xpts,ypts)
        self.get_interp()
        #self.get_spline()

    def __call__(self,pos):
        return self.forward(pos)

    def get_interp(self):
        """Creates a function that interpolate the data points.
        """
        if self.data['x'] is not None:
            self.finterp = interpolate.interp1d(self.data['x'],
                                                self.data['y'],
                                                fill_value='extrapolate')

    def get_spline(self):
        if self.data['x'] is not None:
            # self.finterp = interpolate.CubicSpline(self.data['x'],self.data['y'],extrapolate='True')
            # self.finterp_kin = self.finterp.derivative(nu=2)

            #self.finterp = interpolate.InterpolatedUnivariateSpline(self.data['x'],self.data['y'],k=2)
            self.finterp = interpolate.UnivariateSpline(self.data['x'],self.data['y'],k=5)
            self.finterp_kin = self.finterp.derivative(n=2)

    def load_data(self,x,y):
        """load data points in the class
        
        Args:
            x (array): x coordinates of the points
            y (array): y values of the points
        """
        
        x = np.insert(x,0,1.25*self.domain['xmin'])
        y = np.insert(y,0,0)

        x = np.insert(x,len(x),1.25*self.domain['xmax'])
        y = np.insert(y,len(y),0)

        self.data = {'x':[],'y':[]}
        self.data['x'] = x
        self.data['y'] = y

    def forward(self,pos):
        """Compute the value of the wave function.
        for a multiple conformation of the electrons
        
        Args:
            pos (torch.tensor): positions of the particle
        
        Returns:
            torch.tensor: value of the wave function
        """

        x = pos.detach().numpy()
        x = self.finterp(x)
        return torch.tensor(x).view(-1,1)


    def kinetic_energy(self,pos,out=None):
        """Compute the second derivative of the network
           output w.r.t the value of the input. 
        
        Args:
            pos (torch.tensor): position of the particle
            out (torch.tensor, optional): precomputed values of the wf
                Defaults to None.
        
        Returns:
            torch.tensor: values of the kinetic energy
        """
        _spl_ = False
        if _spl_:
            K = torch.tensor(-0.5*self.finterp_kin(pos.detach().numpy()))
        else:
            eps = 5*1E-2
            if out is None:
                out = self.forward(pos)

            xp = self.forward(pos+eps)
            xm = self.forward(pos-eps)
            K = -0.5 / eps / eps * (xm+xp-2.*out)

        return K.view(-1,1)

    def nuclear_potential(self,pos):
        """Compute the potential of the wf points.

        Args:
            pos (torch.tensor): position of the electron

        Returns: 
            torch.tensor: values of V 
        """
        return self.user_potential(pos).flatten().view(-1,1)       