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

def load_abalone_data(proportion=1044./4177):
    from sklearn import datasets
    from sklearn import preprocessing
    from sklearn import cross_validation
    abalone = datasets.fetch_mldata('regression-datasets abalone')
    X_cate = np.array([abalone.target[i].tolist()
        for i in range(abalone.target.shape[0])])
    X_cate = preprocessing.label_binarize(X_cate, np.unique(X_cate))
    X = np.hstack((X_cate, abalone.data))
    y = abalone.int1[0].T.astype(np.float64)
    y = y[:, None]
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=proportion)
    return X_train, y_train, X_test, y_test

repeats = 20
kerns = ["dot", "rbf", "per", "exp"]
rank_choices = [3]
feature_size_choices = [70]
scores = [[[] for _ in kerns] for _ in kerns]
for i, kern1 in enumerate(kerns):
    for j, kern2 in enumerate(kerns):
        for rank in rank_choices:
            for feature_size in feature_size_choices:
                for _ in range(repeats):
                    X_train, y_train, X_test, y_test = load_abalone_data()
                    model = SCFGP(rank, feature_size, kern1, kern2, False)
                    model.fit(X_train, y_train, X_test, y_test, plot_training=True)
                    scores[i][j].append(model.SCORE)
                    print("\n>>>", model.NAME, kern1, kern2)
                    print("    Score = %.4f | Avg = %.4f | Std = %.4f"%(
                        model.SCORE, np.mean(scores[i][j]), np.std(scores[i][j])))