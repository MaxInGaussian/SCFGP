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
rank_choices = [5, 10]
feature_size_choices = [10, 20]
sum_score, count_score = 0, 0
for feature_size in feature_size_choices:
    for rank in rank_choices:
        for _ in range(repeats):
            X_train, y_train, X_test, y_test = load_boston_data()
            model = SCFGP(rank, feature_size, False)
            model.fit(X_train, y_train, X_test, y_test)
            sum_score += model.SCORE
            count_score += 1
            print("\n>>>", model.NAME)
            print("    Model Selection Score = %.4f | Avg = %.4f"%(
                model.SCORE, sum_score/count_score))