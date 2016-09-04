################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
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

Ms = [30]
X, y = load_co2_data()
opt = Optimizer("adam", [1e-3, 0.9, 0.9], 500, 30, 1e-5, False)
for rank in [50]:
    for M in Ms:
        model = SCFGP(rank, M, msg=True)
        model.fit(X, y, opt=opt, plot_1d_function=True)