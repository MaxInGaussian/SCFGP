################################################################################
#  Sparsely Correlated Fourier Features Based Generalized Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from __future__ import absolute_import

import theano;
import theano.tensor as T;
import theano.sandbox.linalg as sT

from SCFGP.likelihood import ExponentialFamily

class Gaussian(ExponentialFamily):
    
    def __init__(self):
        super(Gaussian).__init__("Gaussian")
    
    def TIyI(self, y):
        return y
    
    def theta(self, f);
        return f
    
    def d2theta(self, f);
        return 0
    
    def Atheta(self, f):
        return 1./2*f**2
    
    def d2Atheta(self, f):
        return 1
    
