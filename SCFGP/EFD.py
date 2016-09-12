################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import theano
import theano.tensor as T

class EFD(object):
    
    " Exponential Family Distribution for Generalized SCFGP "

    available_efds = ["gaussian", "bernoulli", "poisson", "beta"]
    
    efd = None
    params_size, params_vec = None, None
    
    def __init__(self, efd):
        assert efd.lower() in self.available_efds, "Invalid Distribution!"
        self.efd = efd.lower()
    
    def theta(self, f):
        if(self.efd == "gaussian"):
            return f
        elif(self.efd == "bernoulli"):
            return f
        elif(self.efd == "poisson"):
            return f
        elif(self.efd == "beta"):
            return 1./(1+T.exp(-f))
    
    def T(self, y):
        if(self.efd == "gaussian"):
            return y
        elif(self.efd == "bernoulli"):
            return y
        elif(self.efd == "poisson"):
            return y
        elif(self.efd == "beta"):
            return T.log(y/(1-y))
    
    def a(self, disper):
        if(self.efd == "gaussian"):
            return disper
        elif(self.efd == "bernoulli"):
            return disper
        elif(self.efd == "poisson"):
            return disper
        elif(self.efd == "beta"):
            return disper
    
    def b(self, f, disper):
        theta = self.theta(f)
        if(self.efd == "gaussian"):
            return 0.5*theta**2.
        elif(self.efd == "bernoulli"):
            return T.log(1+T.exp(theta))
        elif(self.efd == "poisson"):
            return T.exp(theta)
        elif(self.efd == "beta"):
            return disper*(T.gammaln(theta/disper)+T.gammaln((1-theta)/disper))
    
    def c(self, y, disper):
        if(self.efd == "gaussian"):
            return -0.5*T.log(2*np.pi*disper)-0.5*y**2/disper
        elif(self.efd == "bernoulli"):
            return 0.
        elif(self.efd == "poisson"):
            return -T.log(T.gamma(y+1))
        elif(self.efd == "beta"):
            return T.gammaln(1./disper)+(1./disper-1)*T.log(1-y)-T.log(y)
    
    def NegLogLik(self, y, f, disper):
        return (self.b(f, disper)-self.T(y)*self.theta(f))/\
            self.a(disper)-self.c(y, disper)
            
    def ExpectedNegLogLik(self, y, mu_f, var_f, disper):
        hermgauss = np.polynomial.hermite.hermgauss(50)
        x = theano.shared(hermgauss[0])[None, None, :]
        w = theano.shared(hermgauss[1]/np.sqrt(np.pi))[None, None, :]
        enll = w*self.NegLogLik(y[:, :, None], T.sqrt(2*var_f[:, :, None])*x+mu_f[:, :, None], disper)
        return enll.sum()