################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np

class Scaler(object):
    
    " Scaler (Data Preprocessing) "

    scaling_methods = [
        "min-max",
        "normal",
        "log-normal",
        "sigmoid",
        "log-sigmoid"
    ]
    
    data = {}
    
    def __init__(self, method):
        assert method.lower() in self.scaling_methods, "Invalid Scaling Method!"
        self.method = method.lower()
        if(self.method == "min-max"):
            self.data = {"cols": None, "min": 0, "max":0}
        elif(self.method == "normal"):
            self.data = {"cols": None, "mu":0}
        elif(self.method == "log-normal"):
            self.data = {"cols": None, "log-std": 0, "log-mu":0}
        elif(self.method == "sigmoid"):
            self.data = {"cols": None, "std": 0, "mu":0}
        elif(self.method == "log-sigmoid"):
            self.data = {"cols": None, "log-std": 0, "log-mu":0}
    
    def fit(self, X):
        self.data["cols"] = list(set(range(X.shape[1])).difference(
            np.where(np.all(X == X[0,:], axis = 0))[0]))
        tX = X[:, self.data["cols"]]
        if(self.method == "min-max"):
            self.data['min'] = np.min(tX, axis=0)
            self.data['max'] = np.max(tX, axis=0)
        elif(self.method == "normal"):
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
        elif(self.method == "log-normal"):
            tX = np.log(tX)
            self.data['log-mu'] = np.mean(tX, axis=0)
            self.data['log-std'] = np.std(tX, axis=0)
        elif(self.method == "sigmoid"):
            self.data['mu'] = np.mean(tX, axis=0)
            self.data['std'] = np.std(tX, axis=0)
        elif(self.method == "log-sigmoid"):
            tX = np.log(tX)
            self.data['log-mu'] = np.mean(tX, axis=0)
            self.data['log-std'] = np.std(tX, axis=0)
    
    def forward_transform(self, X):
        tX = X[:, self.data["cols"]]
        if(self.method == "min-max"):
            return (tX-self.data["min"])/(self.data["max"]-self.data["min"])
        elif(self.method == "normal"):
            return (tX-self.data["mu"])/self.data["std"]
        elif(self.method == "log-normal"):
            return (np.log(tX)-self.data["log-mu"])/self.data["log-std"]
        elif(self.method == "sigmoid"):
            return np.tanh((tX-self.data["mu"])/self.data["std"])
        elif(self.method == "log-sigmoid"):
            return np.tanh((np.log(tX)-self.data["log-mu"])/self.data["log-std"])
    
    def backward_transform(self, X):
        assert len(self.data["cols"]) == X.shape[1], "Backward Transform Error"
        if(self.method == "min-max"):
            return X*(self.data["max"]-self.data["min"])+self.data["min"]
        elif(self.method == "normal"):
            return X*self.data["std"]+self.data["mu"]
        elif(self.method == "log-normal"):
            return np.exp(X*self.data["log-std"]+self.data["log-mu"])
        elif(self.method == "sigmoid"):
            return (np.log(1+X)-np.log(1-X))/2.*self.data["std"]+self.data["mu"]
        elif(self.method == "log-sigmoid"):
            return np.exp((np.log(1+X)-np.log(1-X))/2.*self.data["log-std"]+self.data["log-mu"])
    
