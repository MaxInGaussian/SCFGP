################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import os, sys
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
try:
    from SCFGP import SCFGP, Visualizer
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SCFGP import SCFGP, Visualizer
    print("done.")

def test_xgb_regressor():
    import xgboost as xgb
    from sklearn.grid_search import GridSearchCV
    xgb_params = {
        'max_depth': [3, 4, 5, 6, 7],
        'n_estimators': [200, 400, 600]
    }
    clf = GridSearchCV(xgb.XGBRegressor(), xgb_params, verbose=1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    xgb_mae = np.mean(np.abs(y_pred-y_valid))
    xgb_nmae = xgb_mae/np.std(y_valid)
    xgb_mse = np.mean((y_pred-y_valid)**2.)
    xgb_nmse = xgb_mse/np.var(y_valid)
    print("\n>>> XGBRegressor", clf.best_params_)
    print("    Mean Absolute Error\t\t\t\t= %.4f"%(xgb_mae))
    print("    Normalized Mean Absolute Error\t= %.4f"%(xgb_nmae))
    print("    Mean Square Error\t\t\t\t= %.4f"%(xgb_mse))
    print("    Normalized Mean Square Error\t\t= %.4f"%(xgb_nmse))

def load_boston_data(prop=400/506):
    from sklearn import datasets
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    y = y[:, None]
    ntrain = y.shape[0]
    train_inds = npr.choice(range(ntrain), int(prop*ntrain), replace=False)
    valid_inds = np.setdiff1d(range(ntrain), train_inds)
    X_train, y_train = X[train_inds].copy(), y[train_inds].copy()
    X_valid, y_valid = X[valid_inds].copy(), y[valid_inds].copy()
    return X_train, y_train, X_valid, y_valid

trials_per_model = 50
X_train, y_train, X_valid, y_valid = load_boston_data()
feature_size_choices = [int(X_train.shape[0]**0.5*(i+1)) for i in range(5)]
evals = {
    "SCORE": ["Model Selection Score", []],
    "COST": ["Hyperparameter Selection Cost", []],
    "MAE": ["Mean Absolute Error", []],
    "NMAE": ["Normalized Mean Absolute Error", []],
    "MSE": ["Mean Square Error", []],
    "NMSE": ["Normalized Mean Square Error", []],
    "MNLP": ["Mean Negative Log Probability", []],
    "TIME(s)": ["Training Time", []],
}
for feature_size in feature_size_choices:
    funcs = None
    results = {en:[] for en in evals.keys()}
    for round in range(trials_per_model):
        X_train, y_train, X_valid, y_valid = load_boston_data()
        model = SCFGP(-1, feature_size, y_scaling_method='log-normal', verbose=True)
        plt.close()
        vis = Visualizer(plt.figure(figsize=(8, 6), facecolor='white'))
        if(funcs is None):
            model.fit(X_train, y_train, X_valid, y_valid, vis=vis)
            funcs = (model.train_func, model.pred_func)
        else:
            model.fit(X_train, y_train, X_valid, y_valid, funcs, vis=vis)
        if(not os.path.exists("boston_scfgp.pkl")):
            model.save("boston_scfgp.pkl")
            best_model = model
        else:
            best_model = SCFGP(verbose=False)
            best_model.load("boston_scfgp.pkl")
            best_model.predict(X_valid, y_valid)
            if(model.evals['SCORE'][1][-1] > best_model.evals['SCORE'][1][-1]):
                model.save("boston_scfgp.pkl")
                best_model = model
        for metric in evals.keys():
            results[metric].append(model.evals[metric][1][-1])
        print("\n>>>", model.NAME)
        print("    Model Selection Score\t\t\t= %.4f%s| Best = %.4f"%(
            model.evals['SCORE'][1][-1], "  ", best_model.evals['SCORE'][1][-1]))
        print("    Hyperparameter Selection Cost\t= %.4f%s| Best = %.4f"%(
            model.evals['COST'][1][-1], "  ", best_model.evals['COST'][1][-1]))
        print("    Mean Absolute Error\t\t\t\t= %.4f%s| Best = %.4f"%(
            model.evals['MAE'][1][-1], "  ", best_model.evals['MAE'][1][-1]))
        print("    Normalized Mean Absolute Error\t= %.4f%s| Best = %.4f"%(
            model.evals['NMSE'][1][-1], "  ", best_model.evals['NMSE'][1][-1]))
        print("    Mean Square Error\t\t\t\t= %.4f%s| Best = %.4f"%(
            model.evals['MSE'][1][-1], "  ", best_model.evals['MSE'][1][-1]))
        print("    Normalized Mean Square Error\t\t= %.4f%s| Best = %.4f"%(
            model.evals['NMSE'][1][-1], "  ", best_model.evals['NMSE'][1][-1]))
        print("    Mean Negative Log Probability\t= %.4f%s| Best = %.4f"%(
            model.evals['MNLP'][1][-1], "  ", best_model.evals['MNLP'][1][-1]))
        print("    Training Time\t\t\t\t\t= %.4f%s| Best = %.4f"%(
            model.evals['TIME(s)'][1][-1], "  ", best_model.evals['TIME(s)'][1][-1]))
    for en in evals.keys():
        evals[en][1].append((np.mean(results[en]), np.std(results[en])))

import os
if not os.path.exists('plots'):
    os.mkdir('plots')
for en, (metric_name, metric_result) in evals.items():
    f = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
    ax = f.add_subplot(111)
    maxv, minv = 0, 1e5
    for i in range(len(feature_size_choices)):
        maxv = max(maxv, metric_result[i][0])
        minv = min(minv, metric_result[i][0])
        ax.text(feature_size_choices[i], metric_result[i][0], '%.2f' % (
            metric_result[i][0]), fontsize=5)
    line = ax.errorbar(feature_size_choices, [metric_result[i][0] for i in
        range(len(feature_size_choices))], fmt='-o')
    ax.set_xlim([min(feature_size_choices)-10, max(feature_size_choices)+10])
    ax.set_ylim([minv-(maxv-minv)*0.15,maxv+(maxv-minv)*0.45])
    plt.title(metric_name, fontsize=18)
    plt.xlabel('Number of Fourier Features', fontsize=13)
    plt.ylabel(en, fontsize=13)
    plt.savefig('plots/'+en.lower()+'.png')