################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import time
import numpy as np
import matplotlib.pyplot as plt
try:
    from SCFGP import Regressor, Optimizer
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SCFGP import Regressor, Optimizer
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

Ms = [10, 20, 30]
model_types = ["phz", "fz", "ph", "f"]
X, y = load_co2_data()
opt = Optimizer("adam", int(1e10), 50, 1e-4, [0.05, 0.9, 0.999])
for fftype in model_types:
    for rank in ["full", 50, 100]:
        for M in Ms:
            model = Regressor(rank, M, fftype=fftype, msg=False)
            model.fit(X, y, opt=opt, plot_1d_function=True)
            plt.close()
            plot_1d_fig = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
            plot_1d_fig.suptitle(model.NAME, fontsize=15)
            plot_1d_ax = plot_1d_fig.add_subplot(111)
            pts = 300
            errors = [0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64, 2.2]
            Xplot = np.linspace(-0.1, 1.1, pts)[:, None]
            mu, std = model.pred_func(
                Xplot, model.hyper, model.invL, model.AiPhiTY)
            mu = mu.ravel()
            std = std.ravel()
            for er in errors:
                plot_1d_ax.fill_between(Xplot[:, 0], mu-er*std, mu+er*std,
                                alpha=((3-er)/5.5)**1.7, facecolor='blue',
                                linewidth=0.0)
            plot_1d_ax.plot(Xplot[:, 0], mu, alpha=0.8, c='black')
            plot_1d_ax.errorbar(model.X[:, 0],
                model.y.ravel(), fmt='r.', markersize=5, alpha=0.6)
            yrng = model.y.max()-model.y.min()
            plot_1d_ax.set_ylim([
                model.y.min()-0.5*yrng, model.y.max() + 0.5*yrng])
            plot_1d_ax.set_xlim([-0.1, 1.1])
            plot_1d_fig.savefig('co2_%s_%d_%s.png'%(str(rank), M, fftype))