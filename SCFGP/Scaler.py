################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
from scipy.stats import norm, kstest, boxcox_normmax
from scipy.special import boxcox, inv_boxcox
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
            self.data = {"cols": None, "std": 0, "mu":0,
                "lmb":0, "min": 0, "max":0, "bias":0}
        elif(self.algo == "auto-inv-normal"):
            self.data = {"cols": None, "std": 0, "mu":0,
                "lmb":0, "min": 0, "max":0, "bias":0}
    
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
            mu = np.mean(tX, axis=0)
            std = np.std(tX, axis=0)
            self.data['min'] = mu-3*std
            self.data['max'] = mu+3*std
            tX = (tX-self.data["min"])/(self.data["max"]-self.data["min"])
            tX = np.maximum(np.zeros_like(tX), tX)
            tX = np.minimum(np.ones_like(tX), tX)
            self.data['bias'] = np.zeros(tX.shape[1])
            self.data['lmb'] = np.zeros(tX.shape[1])
            for d in range(tX.shape[1]):
                lmb_func = lambda b: boxcox_normmax(tX[:, d]+b[0]**2)
                ks_func = lambda x: kstest(x.ravel(), 'norm')[0]
                fun = lambda b: ks_func(boxcox(tX[:, d]+b[0]**2, lmb_func(b)))
                b = minimize(fun, [.3], method='SLSQP', bounds=[(0.01, 2)],
                    options={'xtol': 1e-4, 'maxiter':100, 'disp': True})['x']
                self.data['bias'][d] = b[0]**2
                self.data['lmb'][d] = boxcox_normmax(tX[:, d]+b[0]**2)
            tX += self.data['bias'][None, :]
            tX = boxcox(tX, self.data['lmb'])
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
        elif(self.algo == "auto-inv-normal"):
            mu = np.mean(tX, axis=0)
            std = np.std(tX, axis=0)
            self.data['min'] = mu-3*std
            self.data['max'] = mu+3*std
            tX = (tX-self.data["min"])/(self.data["max"]-self.data["min"])
            tX = np.maximum(np.zeros_like(tX), tX)
            tX = np.minimum(np.ones_like(tX), tX)
            self.data['bias'] = np.zeros(tX.shape[1])
            self.data['lmb'] = np.zeros(tX.shape[1])
            for d in range(tX.shape[1]):
                if(np.unique(tX[:, d]).shape[0] < 10):
                    self.data['bias'][d] = 1.
                    self.data['lmb'][d] = 1.
                    continue
                lmb_func = lambda b: boxcox_normmax(tX[:, d]+b[0]**2)
                ks_func = lambda x: kstest(norm.cdf(
                    (x-np.mean(x))/np.std(x)), 'uniform')[0]
                fun = lambda b: ks_func(boxcox(tX[:, d]+b[0]**2, lmb_func(b)))
                b = minimize(fun, [.3], method='SLSQP', bounds=[(0.01, 2)],
                    options={'xtol': 1e-4, 'maxiter':100, 'disp': True})['x']
                self.data['bias'][d] = b[0]**2
                self.data['lmb'][d] = boxcox_normmax(tX[:, d]+b[0]**2)
            tX += self.data['bias'][None, :]
            tX = boxcox(tX, self.data['lmb'])
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
            tX = (tX-self.data["min"])/(self.data["max"]-self.data["min"])
            tX = np.maximum(np.zeros_like(tX), tX)
            tX = np.minimum(np.ones_like(tX), tX)
            tX += self.data['bias'][None, :]
            tX = boxcox(tX, self.data['lmb'])
            return (tX-self.data["mu"])/self.data["std"]
        elif(self.algo == "auto-inv-normal"):
            tX = (tX-self.data["min"])/(self.data["max"]-self.data["min"])
            tX = np.maximum(np.zeros_like(tX), tX)
            tX = np.minimum(np.ones_like(tX), tX)
            tX += self.data['bias'][None, :]
            tX = boxcox(tX, self.data['lmb'])
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
            tX = X*self.data["std"]+self.data["mu"]
            tX = inv_boxcox(tX, self.data['lmb'])
            tX -= self.data['bias'][None, :]
            return tX*(self.data["max"]-self.data["min"])+self.data["min"]
        elif(self.algo == "auto-inv-normal"):
            tX = norm.ppf(X)*self.data["std"]+self.data["mu"]
            tX = inv_boxcox(tX, self.data['lmb'])
            tX -= self.data['bias'][None, :]
            return tX*(self.data["max"]-self.data["min"])+self.data["min"]
    
