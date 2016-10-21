################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
from scipy.stats import norm, shapiro
from scipy.optimize import minimize

class Scaler(object):
    
    " Scaler (Data Preprocessing) "

    algos = [
        "min-max",
        "normal",
        "inv-normal",
        "auto-normal",
        "auto-inv-normal",
    ]
    
    data = {}
    
    def __init__(self, algo):
        assert algo.lower() in self.algos, "Invalid Scaling Algorithm!"
        self.algo = algo.lower()
        if(self.algo == "min-max"):
            self.data = {"cols": None, "min": 0, "max":0}
        elif(self.algo == "normal"):
            self.data = {"cols": None, "std": 0, "mu":0}
        elif(self.algo == "inv-normal"):
            self.data = {"cols": None, "std": 0, "mu":0}
        elif(self.algo == "auto-normal"):
            self.data = {"cols": None, "std": 0, "mu":0, "ihs1":0, "ihs2":0}
        elif(self.algo == "auto-inv-normal"):
            self.data = {"cols": None, "std": 0, "mu":0, "ihs1":0, "ihs2":0}
    
    def fit(self, X):
        self.data["cols"] = list(set(range(X.shape[1])).difference(
            np.where(np.all(X == X[0,:], axis = 0))[0]))
        tX = X[:, self.data["cols"]]
        if(self.algo == "min-max"):
            self.data['min'] = np.min(tX, axis=0)
            self.data['max'] = np.max(tX, axis=0)
        elif(self.algo == "normal"):
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
        elif(self.algo == "inv-normal"):
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
        elif(self.algo == "auto-normal"):
            ihs = lambda y, a, b: np.arcsinh(a*(y+b))/a
            self.data['ihs1'] = np.zeros(tX.shape[1])
            self.data['ihs2'] = np.zeros(tX.shape[1])
            for d in range(tX.shape[1]):
                if(np.unique(tX[:, d]).shape[0] < 10):
                    self.data['ihs1'][d] = 1e-3
                    self.data['ihs2'][d] = 0
                    continue
                test = lambda y: 1-shapiro(y.ravel())[0]
                fun = lambda x: test(ihs(tX[:, d], np.exp(x[0]), x[1]))
                std = np.std(tX[:, d])
                bounds = [(-2, 2), (-2*std, 2*std)]
                x = minimize(fun, [0., 0.], method='SLSQP', bounds=bounds,
                    options={'ftol': 1e-8, 'maxiter':100, 'disp':False})['x']
                self.data['ihs1'][d] = np.exp(x[0])
                self.data['ihs2'][d] = x[1]
            tX = ihs(tX, self.data['ihs1'][None, :], self.data['ihs2'][None, :])
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
        elif(self.algo == "auto-inv-normal"):
            ihs = lambda y, a, b: np.arcsinh(a*(y+b))/a
            self.data['ihs1'] = np.zeros(tX.shape[1])
            self.data['ihs2'] = np.zeros(tX.shape[1])
            for d in range(tX.shape[1]):
                if(np.unique(tX[:, d]).shape[0] < 10):
                    self.data['ihs1'][d] = 1e-3
                    self.data['ihs2'][d] = 0
                    continue
                test = lambda y: 1-shapiro(y.ravel())[0]
                fun = lambda x: test(ihs(tX[:, d], np.exp(x[0]), x[1]))
                std = np.std(tX[:, d])
                bounds = [(-2, 2), (-2*std, 2*std)]
                x = minimize(fun, [0., 0.], method='SLSQP', bounds=bounds,
                    options={'ftol': 1e-8, 'maxiter':100, 'disp':False})['x']
                self.data['ihs1'][d] = np.exp(x[0])
                self.data['ihs2'][d] = x[1]
            tX = ihs(tX, self.data['ihs1'][None, :], self.data['ihs2'][None, :])
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
    
    def forward_transform(self, X):
        tX = X[:, self.data["cols"]]
        if(self.algo == "min-max"):
            return (tX-self.data["min"])/(self.data["max"]-self.data["min"])
        elif(self.algo == "normal"):
            return (tX-self.data["mu"])/self.data["std"]
        elif(self.algo == "inv-normal"):
            return norm.cdf((tX-self.data["mu"])/self.data["std"])
        elif(self.algo == "auto-normal"):
            ihs = lambda y, a, b: np.arcsinh(a*(y+b))/a
            tX = ihs(tX, self.data['ihs1'][None, :], self.data['ihs2'][None, :])
            return (tX-self.data["mu"])/self.data["std"]
        elif(self.algo == "auto-inv-normal"):
            ihs = lambda y, a, b: np.arcsinh(a*(y+b))/a
            tX = ihs(tX, self.data['ihs1'][None, :], self.data['ihs2'][None, :])
            return norm.cdf((tX-self.data["mu"])/self.data["std"])
    
    def backward_transform(self, X):
        assert len(self.data["cols"]) == X.shape[1], "Backward Transform Error"
        if(self.algo == "min-max"):
            return X*(self.data["max"]-self.data["min"])+self.data["min"]
        elif(self.algo == "normal"):
            return X*self.data["std"]+self.data["mu"]
        elif(self.algo == "inv-normal"):
            return (norm.ppf(X)-self.data["mu"])/self.data["std"]
        elif(self.algo == "auto-normal"):
            hs = lambda y, a, b: np.sinh(a*y)/a-b
            tX = X*self.data["std"]+self.data["mu"]
            tX = hs(tX, self.data['ihs1'][None, :], self.data['ihs2'][None, :])
            return tX
        elif(self.algo == "auto-inv-normal"):
            hs = lambda y, a, b: np.sinh(a*y)/a-b
            tX = norm.ppf(X)*self.data["std"]+self.data["mu"]
            tX = hs(tX, self.data['ihs1'][None, :], self.data['ihs2'][None, :])
            return tX
    
