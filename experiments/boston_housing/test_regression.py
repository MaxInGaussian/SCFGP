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

repeats = 20
kerns = ["dot", "wht", "lin", "rbf", "per", "exp"]
rank_choices = [2, 4, 6]
feature_size_choices = [150, 100, 50]
for rank in rank_choices:
    scores = [[] for _ in kerns]
    for feature_size in feature_size_choices:
        for _ in range(repeats):
            X_train, y_train, X_test, y_test = load_boston_data()
            for i, kern in enumerate(kerns):
                model = SCFGP(rank, feature_size, kern, kern, False)
                model.fit(X_train, y_train, X_test, y_test, plot_training=True)
                scores[i].append(model.SCORE)
                print("\n>>>", model.NAME, kern)
                print("    Score = %.4f | Avg = %.4f | Std = %.4f"%(
                    model.SCORE, np.mean(scores[i]), np.std(scores[i])))