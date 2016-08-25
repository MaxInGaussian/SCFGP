################################################################################
#  Optimized Fourier Features Based Gaussian Process Regression
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
try:
    from OffGPR import OffGPR
except:
    print("OffGPR is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../")
    from OffGPR import OffGPR
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
fourier_feature_types = ["f", "flr", "ph", "phlr", "zf", "zflr", "zph", "zphlr"]
X, y = load_1d_function_data()
fig_num = 1
for ftype in feature_types:
    tf1, tf2, tf3 = False, False, False
    if("z" in fftype):
        tf1 = True
    if("ph" in fftype):
        tf2 = True
    if("lr" in fftype):
        tf3 = True
        rank = 3
    else:
        rank = "full"
    for M in Ms:
        model = OffGPR(rank, M, tf1, tf2, tf3)
        model.fit(X, y, plot_1d_function=True)
