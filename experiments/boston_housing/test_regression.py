################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import matplotlib.pyplot as plt
try:
    from SCFGP import *
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SCFGP import *
    print("done.")

def load_boston_data(proportion=106./506):
    from sklearn import datasets
    from sklearn import cross_validation
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    y = y[:, None]
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=proportion)
    return X_train, y_train, X_test, y_test

repeats = 3
feature_size_choices = [50]
scores = []
nmses = []
mnlps = []
X_train, y_train, X_test, y_test = load_boston_data()
for feature_size in feature_size_choices:
    for _ in range(repeats):
        model = SCFGP(-1, feature_size, False)
        model.fit(X_train, y_train, X_test, y_test, plot_training=True)
        nmses.append(model.TsNMSE)
        mnlps.append(model.TsMNLP)
        scores.append(model.SCORE)
        print("\n>>>", model.NAME)
        print("    NMSE = %.4f | Avg = %.4f | Std = %.4f"%(
            model.TsNMSE, np.mean(nmses), np.std(nmses)))
        print("    MNLP = %.4f | Avg = %.4f | Std = %.4f"%(
            model.TsMNLP, np.mean(mnlps), np.std(mnlps)))
        print("    Score = %.4f | Avg = %.4f | Std = %.4f"%(
            model.SCORE, np.mean(scores), np.std(scores)))