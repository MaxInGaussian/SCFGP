################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
######################################################################################

import numpy as np

class Normalizer(object):
    
    " Normalizer (Data Preprocessing) "

    methods = ["linear", "centralize", "standardize", "sigmoid", "uniformize"]
    
    data = {}
    
    def __init__(self, meth):
        assert meth in self.methods, "Invalid Normalization Method!"
        if(meth == "linear"):
            self.data = {"cols": None, "min": 0, "max":0}
        elif(meth == "centralize"):
            self.data = {"cols": None, "mu":0}
        elif(meth == "standardize"):
            self.data = {"cols": None, "std": 0, "mu":0}
        elif(meth == "sigmoid"):
            self.data = {"cols": None, "std": 0, "mu":0}
        elif(meth == "uniformize"):
            self.data = {"cols": None, "X_sort": None, "min": 0, "max":0}
        self.meth = meth
    
    def fit(self, X):
        self.data["cols"] = list(set(range(X.shape[1])).difference(
            np.where(np.all(X == X[0,:], axis = 0))[0]))
        tX = X[:, self.data["cols"]]
        for key in self.data.keys():
            if(key == "mu"):
                self.data[key] = np.mean(tX, axis=0)
            elif(key == "std"):
                self.data[key] = np.std(tX, axis=0)
            elif(key == "min"):
                self.data[key] = np.min(tX, axis=0)
            elif(key == "max"):
                self.data[key] = np.max(tX, axis=0)
    
    def forward_transform(self, X):
        tX = X[:, self.data["cols"]]
        if(self.meth == "linear"):
            return (tX-self.data["min"])/(self.data["max"]-self.data["min"])
        elif(self.meth == "centralize"):
            return tX-self.data["mu"]
        elif(self.meth == "standardize"):
            return (tX-self.data["mu"])/self.data["std"]
        elif(self.meth == "sigmoid"):
            return np.tanh((tX-self.data["mu"])/self.data["std"])
        elif(self.meth == "uniformize"):
            N, D = tX.shape
            _X = (tX-self.data["min"])/(self.data["max"]-self.data["min"])
            self.data["X_sort"] = np.sort(_X, axis=0).copy()
            for d in range(D):
                _sort_Xd = self.data["X_sort"][:, d]
                _X[:, d] = np.searchsorted(_sort_Xd, _X[:, d], 'right')/N
            return np.hstack((np.ones((_X.shape[0], 1)), _X))
    
    def backward_transform(self, X):
        assert len(self.data["cols"]) == X.shape[1], "Backward Transform Error"
        if(self.meth == "linear"):
            return X*(self.data["max"]-self.data["min"])+self.data["min"]
        elif(self.meth == "centralize"):
            return X+self.data["mu"]
        elif(self.meth == "standardize"):
            return X*self.data["std"]+self.data["mu"]
        elif(self.meth == "sigmoid"):
            return (np.log(1+X)-np.log(1-X))/2.*self.data["std"]+self.data["mu"]
        elif(self.meth == "uniformize"):
            N, D = X.shape
            _X = X.copy()
            for d in range(D):
                _sort_Xd = self.data["X_sort"][:, d]
                _X[:, d] = np.searchsorted(_sort_Xd, _X[:, d], 'right')/N
            return _X[:, 1:]*(self.data["max"]-self.data["min"])+self.data["min"]
    
