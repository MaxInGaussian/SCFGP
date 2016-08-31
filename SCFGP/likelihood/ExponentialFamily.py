################################################################################
#  Sparsely Correlated Fourier Features Based Generalized Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from __future__ import absolute_import

import numpy as np
import theano.tensor as T;

class ExponentialFamily(object):
    
    def __init__(self, name):
        self.name = name
    
    def __str__(self):
        return self.name
    
    def aIdI(self, d):
        raise NotImplementedError
    
    def cIy_dI(self, y, d):
        raise NotImplementedError
    
    def TIyI(self, y):
        raise NotImplementedError
    
    def theta(self, f):
        raise NotImplementedError
    
    def dtheta(self, f):
        raise NotImplementedError
    
    def d2theta(self, f):
        raise NotImplementedError
    
    def d3theta(self, f):
        raise NotImplementedError
    
    def Atheta(self, f):
        raise NotImplementedError
    
    def dAtheta(self, f):
        raise NotImplementedError
    
    def d2Atheta(self, f):
        raise NotImplementedError
    
    def d3Atheta(self, f):
        raise NotImplementedError
        
    def EItheataI(self, m, v):
        return self.theta(m)+1./2*v*self.d2theta(m)
        
    def dEItheataI_dm(self, m, v):
        return self.dtheta(m)+1./2*v*self.d3theta(m)
        
    def dEItheataI_dv(self, m):
        return 1./2*self.d2theta(m)
        
    def EIAtheataI(self, m, v):
        return self.Atheta(m)+1./2*v*self.d2Atheta(m)
        
    def dEIAtheataI_dm(self, m, v):
        return self.dAtheta(m)+1./2*v*self.d3Atheta(m)
        
    def dEIAtheataI_dv(self, m):
        return 1./2*self.d2Atheta(m)
        
    def fIm_v_y_dI(self, m, v, y, d):
        return (self.TIyI(y)*self.EItheataI(m, v)-self.EIAtheataI(m, v))/\
            self.aIdI(d)+self.cIy_dI(y, d)
    
    def df_dm_optimal(self, m, v, y, d):
        return (self.TIyI(y)*self.dEItheataI_dm(m, v)-\
            self.dEIAtheataI_dm(m, v))/self.aIdI(d)
    
    def df_dv_optimal(self, m, y, d):
        return (self.TIyI(y)*self.dEItheataI_dv(m)-\
            self.dEIAtheataI_dv(m))/self.aIdI(d)
    
