import os
import torch
from torch.autograd import Variable
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm


def plot_observable(obs_dict,e0=None,ax=None):
    '''Plot the observable selected.

    Args:
        obs_dict : dictioanry of observable
    '''
    show_plot = False
    if ax is None:    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        show_plot = True

    if isinstance(obs_dict,dict):
        data = obs_dict['local_energy']
    else:
        data = np.hstack(np.squeeze(np.array(obs_dict)))
        
    n = len(data)
    epoch = np.arange(n)

    # get the variance
    emax = [np.quantile(e,0.75) for e in data ]
    emin = [np.quantile(e,0.25) for e in data ]

    # get the mean value
    energy = np.mean(data,1)

    # plot
    ax.fill_between(epoch,emin,emax,alpha=0.5,color='#4298f4')
    ax.plot(epoch,energy,color='#144477')
    if e0 is not None:
        ax.axhline(e0,color='black',linestyle='--')

    ax.grid()
    ax.set_xlabel('Number of epoch')
    ax.set_ylabel('Energy')

    if show_plot:
        plt.show()

class plotter1d(object):

    def __init__(self, wf, domain, res=51, sol = None, 
                 plot_weight=False, plot_grad=False, save=None):
        '''Dynamic plot of a 1D-wave function during the optimization

        Args:
            wf : wave function object
            domain : dict containing the boundary
            res : number of points in each direction
            sol : callabale solution of the problem
            plot_weight : plot the weight of the fc
            plot_grad : plot the grad of the weight
        '''
        plt.ion()
        self.wf = wf
        self.res = res
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot( 111 )

        self.plot_weight = plot_weight
        self.plot_grad = plot_grad
        self.save = save
        self.iter = 0

        self.POS = Variable(torch.linspace(domain['xmin'],domain['xmax'],res).view(res,1))
        pos = self.POS.detach().numpy().flatten()  

        if callable(sol):
            v = sol(self.POS).detach().numpy()
            self.ax.plot(pos,v,color='#b70000',linewidth=4,linestyle='--',label='solution')

        vpot = wf.nuclear_potential(self.POS).detach().numpy()
        self.ax.plot(pos,vpot,color='black',linestyle='--')

        vp = self.wf(self.POS).detach().numpy()
        vp/=np.max(vp)
        self.lwf, = self.ax.plot(pos,vp,linewidth=2,color='black')

        if self.plot_weight:
            self.pweight, = self.ax.plot(self.wf.rbf.centers.detach().numpy(),
                                         self.wf.fc.weight.detach().numpy().T,'o')
        if self.plot_grad:
            if self.wf.fc.weight.requires_grad:
                self.pgrad, = self.ax.plot(self.wf.rbf.centers.detach().numpy(),
                                           np.zeros(self.wf.ncenter),'X')

        self.ax.set_ylim((np.min(vpot),1))
        plt.grid()
        plt.draw()
        self.fig.canvas.flush_events()

        if self.save is not None:
            self._save_pic()

    def drawNow(self):
        '''Update the plot.'''

        vp = self.wf(self.POS).detach().numpy()
        vp/=np.max(vp)
        self.lwf.set_ydata(vp)

        if self.plot_weight:
            self.pweight.set_xdata(self.wf.rbf.centers.detach().numpy())
            self.pweight.set_ydata(self.wf.fc.weight.detach().numpy().T)

        if self.plot_grad:
            if self.wf.fc.weight.requires_grad:
                self.pgrad.set_xdata(self.wf.rbf.centers.detach().numpy())
                data = (self.wf.fc.weight.grad.detach().numpy().T)**2
                data /= np.linalg.norm(data)
                self.pgrad.set_ydata(data)

        #self.fig.canvas.draw() 
        plt.draw()
        self.fig.canvas.flush_events()

        if self.save is not None:
            self._save_pic()

    def _save_pic(self):
        fname = 'image_%03d.png' %self.iter
        fname = os.path.join(self.save,fname)
        plt.savefig(fname)
        self.iter += 1

def plot_wf_1d(net,domain,res,grad=False,hist=False,pot=True,sol=None,ax=None,load=None,feedback=None):
        '''Plot a 1D wave function.

        Args:
            net : network object
            grad : plot gradient
            hist : plot histogram of the data points
            sol : callabale of the solution
        '''

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot( 111 )
            show_plot = True
        else:
            show_plot = False


        if load is not None:
            checkpoint = torch.load(load)
            net.wf.load_state_dict(checkpoint['model_state_dict'])
            epoch = checkpoint['epoch']

        X = Variable(torch.linspace(domain['xmin'],domain['xmax'],res).view(res,1))
        X.requires_grad = True
        xn = X.detach().numpy().flatten()

        if callable(sol):
            vs = sol(X).detach().numpy()
            ax.plot(xn,vs,color='#b70000',linewidth=4,linestyle='--',label='solution')

        if isinstance(sol,dict):
            ax.plot(sol['x'],sol['y'],color='#b70000',linewidth=4,linestyle='--',label='solution')

        vals = net.wf(X)
        vn = vals.detach().numpy().flatten()
        ax.plot(xn,vn,color='black',linewidth=2,label='DeepQMC')

        if pot:
            pot = net.wf.nuclear_potential(X).detach().numpy()
            ax.plot(xn,pot,color='black',linestyle='--')

        if grad:
            kin = net.wf.kinetic_energy(X)
            g = np.gradient(vn,xn)
            h = -0.5*np.gradient(g,xn)
            ax.plot(xn,kin.detach().numpy(),label='kinetic')
            ax.plot(xn,h,label='hessian')

        if hist:
            pos = net.sample(ntherm=-1)
            ax.hist(pos.detach().numpy(),density=False)
        
        if feedback is not None:
            x = feedback['x']
            y1 = feedback['y']
            y2 = feedback['y']+feedback['delta']
            ax.fill_between(x,y1,y2,where=y2>y1,facecolor="#5286c7")
            ax.fill_between(x,y1,y2,where=y1>y2,facecolor="#8e1796")

        plt.scatter(net.wf.data['x'],net.wf.data['y'])
        #ax.set_ylim((-0.1,2.5))
        ax.grid()
        ax.set_xlabel('X')
        if load is None:
            ax.set_ylabel('Wavefuntion')
        else:
            ax.set_ylabel('Wavefuntion %d epoch' %epoch)
        #ax.legend()

        if show_plot:
            plt.show()

def plot_results_1d(net,domain,res,sol=None,e0=None,load=None):
    ''' Plot the summary of the results for a 1D problem.

    Args: 
        net : network object
        obs_dict : dict containing the obserable
        sol : callable of the solutions
        e0 : energy of the solution
        domain : boundary of the plot
        res : number of points in the x axis
    '''
    plt.ioff()
    fig = plt.figure()
    ax0 = fig.add_subplot(211)
    ax1 = fig.add_subplot(212)

    plot_wf_1d(net,domain,res,sol=sol,hist=False,ax=ax0,load=load)
    plot_observable(net.obs_dict,e0=e0,ax=ax1)

    plt.show()
 



