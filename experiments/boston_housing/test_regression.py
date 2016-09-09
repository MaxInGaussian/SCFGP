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
kerns = ["dot", "lin", "rbf", 'exp', 'per']
feature_size_choices = [20]
scores = [[] for _ in kerns]
nmses = [[] for _ in kerns]
mnlps = [[] for _ in kerns]
X_train, y_train, X_test, y_test = load_boston_data()
for i, kern in enumerate(kerns):
    for feature_size in feature_size_choices:
        for _ in range(repeats):
            model = SCFGP(-1, feature_size, kern, kern, False)
            model.fit(X_train, y_train, X_test, y_test, plot_training=True)
            nmses[i].append(model.TsNMSE)
            mnlps[i].append(model.TsMNLP)
            scores[i].append(model.SCORE)
            print("\n>>>", model.NAME, kern)
            print("    NMSE = %.4f | Avg = %.4f | Std = %.4f"%(
                model.TsNMSE, np.mean(nmses[i]), np.std(nmses[i])))
            print("    MNLP = %.4f | Avg = %.4f | Std = %.4f"%(
                model.TsMNLP, np.mean(mnlps[i]), np.std(mnlps[i])))
            print("    Score = %.4f | Avg = %.4f | Std = %.4f"%(
                model.SCORE, np.mean(scores[i]), np.std(scores[i])))