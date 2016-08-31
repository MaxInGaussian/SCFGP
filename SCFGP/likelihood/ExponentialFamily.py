################################################################################
#  Sparsely Correlated Fourier Features Based Generalized Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from __future__ import absolute_import

import theano;
import theano.tensor as T;
import theano.sandbox.linalg as sT

class ExponentialFamily(object):
    
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def TIyI(self, y):
        raise NotImplementedError
    
    def theta(self, f);
        raise NotImplementedError
    
    def d2theta(self, f);
        raise NotImplementedError
    
    def Atheta(self, f):
        raise NotImplementedError
    
    def d2Atheta(self, f):
        raise NotImplementedError
        
    def EItheataI(self, m, V):
        return self.theta(m)+1./2*T.dot(V, self.d2theta(m))
        
    def EIAtheataI(self, m, V):
        return self.Atheta(m)+1./2*T.dot(V, self.d2Atheta(m))
    
