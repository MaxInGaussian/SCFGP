################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
try:
    from SCFGP import SCFGP, Optimizer
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../")
    from SCFGP import SCFGP, Optimizer
    print("done.")

    
def load_mnist_data(proportion=0.1):
    from sklearn import datasets
    from sklearn import cross_validation
    digits = datasets.load_digits()
    X, y = digits.data, digits.target
    labels = np.unique(y).tolist()
    y = y[:, None]
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=proportion)
    return X_train, y_train, X_test, y_test, labels

X_train, y_train, X_test, y_test, labels = load_mnist_data()
_y_train = label_binarize(y_train, classes=labels)
_y_test = label_binarize(y_test, classes=labels)
Ms = [10, 30, 100]
opt = Optimizer("smorms3", 10000000, 10, 1e-3, [1e-2])
for M in Ms:
    fig = plt.figure(1, figsize=(8, 6), facecolor='white', dpi=120)
    ax = fig.add_subplot(111)
    plot = (fig, ax)
    model = SCFGP(rank=2, feature_size=10)
    model.fit(X_train, _y_train, X_test, _y_test, opt=opt)
    plt.title('Mapping 784 to 2 dimension By SCFGP (M=%d)'%(M), fontsize=20)
    plt.savefig('dimensionality_reduction_%d.png'%(M))
    plt.show()