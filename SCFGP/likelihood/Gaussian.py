################################################################################
#  Sparsely Correlated Fourier Features Based Generalized Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from __future__ import absolute_import

import numpy as np
import theano.tensor as T;

from SCFGP.likelihood import ExponentialFamily

class Gaussian(ExponentialFamily):
    
    def __init__(self):
        super().__init__("Gaussian")
    
    def aIdI(self, d):
        return d
    
    def cIy_dI(self, y, d):
        return -0.5*(y/d)**2-T.log(2*np.pi*d)/2
    
    def TIyI(self, y):
        return y
    
    def theta(self, f):
        return f
    
    def dtheta(self, f):
        return T.ones_like(f)
    
    def d2theta(self, f):
        return T.zeros_like(f)
    
    def d3theta(self, f):
        return T.zeros_like(f)
    
    def Atheta(self, f):
        return 1./2*f**2
    
    def dAtheta(self, f):
        return f
    
    def d2Atheta(self, f):
        return T.ones_like(f)
    
    def d3Atheta(self, f):
        return T.zeros_like(f)
    
