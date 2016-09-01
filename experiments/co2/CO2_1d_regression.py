################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
try:
    from SCFGP import SCFGP, Optimizer
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SCFGP import SCFGP, Optimizer
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

Ms = [10]
model_types = ["ph"]
X, y = load_co2_data()
opt = Optimizer("adam", [0.05, 0.9, 0.999], int(1e10), 20, 1e-6, True)
for fftype in model_types:
    for rank in [50]:
        for M in Ms:
            model = SCFGP(rank, M, fftype=fftype, msg=False)
            model.fit(X, y, opt=opt, plot_1d_function=True)
            plt.savefig('co2_%s_%d_%s.png'%(str(rank), M, fftype))
            plt.close()