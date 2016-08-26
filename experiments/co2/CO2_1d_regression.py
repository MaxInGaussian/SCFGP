################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
try:
    from SCFGP import SCFGP
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SCFGP import SCFGP
    print("done.")

def load_1d_function_data(proportion=0.1):
    from sklearn.datasets import fetch_mldata
    from sklearn import cross_validation
    data = fetch_mldata('mauna-loa-atmospheric-co2').data
    X = data[:, [1]]
    y = data[:, 0]
    y = y[:, None]
    X = X.astype(np.float64)
    return X, y

Ms = [10, 20, 30]
fourier_feature_types = ["f", "ph", "zf", "zph"]
X, y = load_1d_function_data()
fig_num = 1
for fftype in fourier_feature_types:
    rank = "full"
    for M in Ms:
        model = SCFGP(rank, M, fftype=fftype)
        model.fit(X, y, plot_1d_function=True)
