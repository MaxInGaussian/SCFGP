################################################################################
#  Sparsely Correlated Fourier Features Based Generalized Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from __future__ import absolute_import

import theano;
import theano.tensor as T;
import theano.sandbox.linalg as sT

from SCFGP.likelihood import ExponentialFamily

class Bernoulli(ExponentialFamily):
    
    def __init__(self):
        super(Bernoulli).__init__("Bernoulli")
    
    def TIyI(self, y):
        return y
        
    def logistic(self, f):
        return 1./(1+T.exp(-f))
    
    def dlogistic(self, f):
        logistic_f = self.logistic(f)
        return logistic_f*(1-logistic_f)
    
    def d2logistic(self, f):
        logistic_f = self.logistic(f)
        d_logistic_f = self.dlogistic(f)
        return d_logistic_f-2*logistic_f*d_logistic_f
    
    def sigmoid(self, f):
        return logistic(f)
    
    def dsigmoid(self, f):
        return dlogistic(f)
    
    def d2sigmoid(self, f):
        return d2logistic(f)
        
    def theta(self, f):
        return -T.log(1./self.sigmoid(f)-1)
    
    def dtheta(self, f):
        sigmoid_f = self.sigmoid(f)
        d_sigmoid_f = self.dsigmoid(f)
        return d_sigmoid_f/sigmoid_f/(1-sigmoid_f)
    
    def d2theta(self, f):
        sigmoid_f = self.sigmoid(f)
        d_sigmoid_f = self.dsigmoid(f)
        isig = 1./sigmoid_f/(1-sigmoid_f)
        return d2sigmoid*isig-(d_sigmoid_f**2)*(1-2*sigmoid_f)*(isig**2)
    
    def Atheta(self, f):
        return T.log(1+T.exp(self.theta(f)))
    
    def d2Atheta(self, f):
        theta_f = self.theta(f)
        logistic_theta = self.logistic(theta_f)
        d_logistic_theta = self.dlogistic(theta_f)
        return self.d2theta(f)*logistic_theta+self.dtheta(f)*d_logistic_theta
    
