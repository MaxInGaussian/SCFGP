################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
try:
    from SCFGP import Regressor, Optimizer
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SCFGP import Regressor, Optimizer
    print("done.")

def load_co2_data(proportion=0.1):
    from sklearn.datasets import fetch_mldata
    from sklearn import cross_validation
    data = fetch_mldata('mauna-loa-atmospheric-co2').data
    X = data[:, [1]]
    y = data[:, 0]
    y = y[:, None]
    X = X.astype(np.float64)
    return X, y

Ms = [10, 20, 30]
model_types = ["phz", "fz", "ph", "f"]
X, y = load_co2_data()
opt = Optimizer("adam", int(1e10), 50, 1e-5, [0.05, 0.9, 0.999])
for fftype in model_types:
    for rank in ["full", 50, 100]:
        for M in Ms:
            model = Regressor(rank, M, fftype=fftype, msg=False)
            model.fit(X, y, opt=opt, plot_1d_function=True)
            plt.savefig('co2_%s_%d_%s.png'%(str(rank), M, fftype))
            plt.close()