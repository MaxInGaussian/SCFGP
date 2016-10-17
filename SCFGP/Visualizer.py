################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.animation as anm

class Visualizer(object):
    
    " Visualizer (Data Visualization) "
    
    model, fig = None, None
    
    def __init__(self, model, fig, eval='NMSE'):
        self.model = model
        self.fig = fig
        self.fig.figsize = (8, 6)
        self.fig.facecolor = 'white'
        self.eval = eval
    
    def train_with_plot(self):
        if(self.model.D == 1):
            self.train_with_1d_plot()
        else:
            self.train_with_eval_plot()
    
    def train_with_1d_plot(self):
        self.fig.clf()
        self.fig.suptitle(self.model.NAME, fontsize=15)
        ax = self.fig.add_subplot(111)
        def animate(i):
            ax.cla()
            pts = 300
            errors = [0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64, 2.2]
            Xplot = np.linspace(-0.1, 1.1, pts)[:, None]
            mu, std = self.model.pred_func(
                Xplot, self.model.hyper, self.model.alpha, self.model.Ri)
            mu = mu.ravel()
            std = std.ravel()
            for er in errors:
                ax.fill_between(Xplot[:, 0], mu-er*std, mu+er*std,
                                alpha=((3-er)/5.5)**1.7, facecolor='blue',
                                linewidth=0.0)
            ax.plot(Xplot[:, 0], mu, alpha=0.8, c='black')
            ax.errorbar(self.model.X[:, 0],
                self.model.y.ravel(), fmt='r.', markersize=5, alpha=0.6)
            yrng = self.model.y.max()-self.model.y.min()
            ax.set_ylim([
                self.model.y.min()-0.5*yrng, self.model.y.max() + 0.5*yrng])
            ax.set_xlim([-0.1, 1.1])
        ani = anm.FuncAnimation(self.fig, animate, interval=1000)
    
    def train_with_eval_plot(self):
        self.fig.clf()
        self.fig.suptitle(self.model.NAME, fontsize=15)
        ax1 = self.fig.add_subplot(211)
        ax2 = self.fig.add_subplot(212)
        plt.xlabel('# iteration', fontsize=13)
        def animate(i):
            iter_axis = np.arange(len(self.model.evals['COST'][1]))
            ax1.cla()
            ax1.plot(iter_axis, self.model.evals['COST'][1],
                color='r', linewidth=2.0, label='COST')
            handles, labels = ax1.get_legend_handles_labels()
            ax1.legend(handles, labels, loc='upper center',
                bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True)
            ax2.cla()
            ax2.plot(iter_axis, self.model.evals[self.eval.upper()][1],
                color='b', linewidth=2.0, label=self.eval.upper())
            handles, labels = ax2.get_legend_handles_labels()
            ax2.legend(handles, labels, loc='upper center',
                bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True)
        ani = anm.FuncAnimation(self.fig, animate, interval=500)