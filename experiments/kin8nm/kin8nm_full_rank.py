################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import matplotlib.pyplot as plt
try:
    from SCFGP import SCFGP
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../../")
    from SCFGP import SCFGP
    print("done.")

def load_kin8nm_data(proportion=3192./8192):
    from sklearn import datasets
    from sklearn import cross_validation
    kin8nm = datasets.fetch_mldata('regression-datasets kin8nm')
    X, y = kin8nm.data[:, :-1], kin8nm.data[:, -1]
    y = y[:, None]
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=proportion)
    return X_train, y_train, X_test, y_test

trials_per_model = 50
X_train, y_train, X_test, y_test = load_kin8nm_data()
rank = "full"
Ms = [int(np.log(X_train.shape[0])/np.log(8)+1)*(i+1) for i in range(10)]
try:
    best_model = SCFGP(msg=False)
    best_model.load("kin8nm_best_model.pkl")
    best_model_score = best_model.SCORE
except (FileNotFoundError, IOError):
    best_model = None
model_types = ["zph", "zf", "ph", "f"]
num_models = len(model_types)
metrics = {
    "MAE": ["Mean Absolute Error", [[] for _ in range(num_models)]],
    "MSE": ["Mean Square Error", [[] for _ in range(num_models)]],
    "RMSE": ["Root Mean Square Error", [[] for _ in range(num_models)]],
    "NMSE": ["Normalized Mean Square Error", [[] for _ in range(num_models)]],
    "MNLP": ["Mean Negative Log Probability", [[] for _ in range(num_models)]],
    "TIME": ["Training Time", [[] for _ in range(num_models)]],
}
for M in Ms:
    for i, fftype in enumerate(model_types):
        funcs = None
        results = {en:[] for en in metrics.keys()}
        for round in range(trials_per_model):
            X_train, y_train, X_test, y_test = load_kin8nm_data()
            model = SCFGP(rank, M, fftype=fftype, msg=False)
            if(funcs is None):
                model.fit(X_train, y_train, X_test, y_test)
                funcs = (model.train_func, model.pred_func)
            else:
                model.fit(X_train, y_train, X_test, y_test, funcs)
            if(best_model is None):
                model.save("kin8nm_best_model.pkl")
                best_model = model
            else:
                best_model.predict(X_test, y_test)
                if(model.SCORE > best_model.SCORE):
                    model.save("kin8nm_best_model.pkl")
                    best_model = model
            results["MAE"].append(model.TsMAE)
            results["MSE"].append(model.TsMSE)
            results["RMSE"].append(model.TsRMSE)
            results["NMSE"].append(model.TsNMSE)
            results["MNLP"].append(model.TsMNLP)
            results["TIME"].append(model.TrTime)
            print("\n>>>", model.NAME)
            print("    Model Selection Score\t\t\t\t= %.4f%s(Best %.4f)"%(
                model.SCORE, "  ", best_model.SCORE))
            print("    Mean Absolute Error\t\t\t\t= %.4f%s(Best %.4f)"%(
                model.TsMAE, "  ", best_model.TsMAE))
            print("    Mean Square Error\t\t\t\t\t= %.4f%s(Best %.4f)"%(
                model.TsMSE, "  ", best_model.TsMSE))
            print("    Root Mean Square Error\t\t\t= %.4f%s(Best %.4f)"%(
                model.TsRMSE, "  ", best_model.TsRMSE))
            print("    Normalized Mean Square Error \t= %.4f%s(Best %.4f)"%(
                model.TsNMSE, "  ", best_model.TsNMSE))
            print("    Mean Negative Log Probability\t= %.4f%s(Best %.4f)"%(
                model.TsMNLP, "  ", best_model.TsMNLP))
            print("    Training Time\t\t\t\t\t\t= %.4f%s(Best %.4f)"%(
                model.TrTime, "  ", best_model.TrTime))
        for en in metrics.keys():
            metrics[en][1][i].append((np.mean(results[en]), np.std(results[en])))
            
import os
if not os.path.exists('full_rank_plots'):
    os.mkdir('full_rank_plots')
for en, (metric_name, metric_results) in metrics.items():
    f = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
    ax = f.add_subplot(111)
    maxv, minv = 0, 1e5
    lines = []
    for j in range(num_models):
        for i in range(len(Ms)):
            maxv = max(maxv, metric_results[j][i][0])
            minv = min(minv, metric_results[j][i][0])
            ax.text(Ms[i], metric_results[j][i][0], '%.2f' % (
                metric_results[j][i][0]), fontsize=5)
        line = ax.errorbar(Ms,
            [metric_results[j][i][0] for i in range(len(Ms))], fmt='-o')
        lines.append(line)
    ax.set_xlim([min(Ms)-10, max(Ms)+10])
    ax.set_ylim([minv-(maxv-minv)*0.15,maxv+(maxv-minv)*0.45])
    plt.title(metric_name, fontsize=20)
    plt.xlabel('# Fourier features', fontsize=13)
    plt.ylabel(en, fontsize=13)
    legend = f.legend(handles=lines, labels=model_types, loc=1, shadow=True)
    plt.savefig('full_rank_plots/'+en.lower()+'.png')