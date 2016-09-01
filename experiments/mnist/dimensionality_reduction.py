################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anm
from sklearn.preprocessing import label_binarize
try:
    from SCFGP import SCFGP
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SCFGP import SCFGP
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
Ms = [10, 30, 50]
model_types = ["phz", "fz", "ph", "f"]
for M in Ms:
    for fftype in model_types:
        model = SCFGP(rank=2, feature_size=M, fftype=fftype)
        def callback():
            if(model.iter == 1):
                fig = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
                fig.suptitle('Mapping 784 to 2 dimensions using %d Fourier '%(M)+\
                    'features (feature type=%s)'%(fftype.upper()), fontsize=15)
                ax = fig.add_subplot(111)
            def animate(i):
                ax.cla()
                c = np.reshape(model.hyper[2:2+model.D*model.R], (model.D, model.R))
                tX = model.X_nml.forward_transform(X_test).dot(c)
                for i in range(tX.shape[0]):
                    ax.plot(tX[i, 0], tX[i, 1], 'o',
                        color=plt.cm.Set1(int(y_test[i])/10.))
                minx, maxx = min(tX[:, 0]), max(tX[:, 0])
                miny, maxy = min(tX[:, 1]), max(tX[:, 1])
                ax.set_xlim([minx-(maxx-minx)*0.05,maxx+(maxx-minx)*0.05])
                ax.set_ylim([miny-(maxy-miny)*0.05,maxy+(maxy-miny)*0.05])
                ax.figure.canvas.draw()
            if(model.iter == 1):
                ani = anm.FuncAnimation(fig, animate, interval=500)
            plt.pause(0.001)
        model.fit(X_train, _y_train, callback=callback)
        plt.savefig('visualize_mnist_%s_%d.png'%(fftype, M))