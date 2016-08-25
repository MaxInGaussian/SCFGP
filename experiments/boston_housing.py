################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
try:
    from SCFGP import SCFGP
except:
    print("SCFGP is not installed yet! Trying to call directly from source...")
    from sys import path
    path.append("../")
    from SCFGP import SCFGP
    print("done.")

def load_low_vs_full_data(proportion=0.1):
    from sklearn import datasets
    from sklearn import cross_validation
    boston = datasets.load_boston()
    X, y = boston.data, boston.target
    y = y[:, None]
    X = X.astype(np.float64)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X, y, test_size=proportion)
    return X_train, y_train, X_test, y_test

X_train, Y_train, X_test, Y_test = load_low_vs_full_data()
rank = 3
trials_per_model = 3
Ms = [10, 20, 30, 40, 50, 60, 70]
fourier_feature_types = ["f", "flr"]
MSE = [[] for _ in range(len(fourier_feature_types))]
NMSE = [[] for _ in range(len(fourier_feature_types))]
MNLP = [[] for _ in range(len(fourier_feature_types))]
TIME = [[] for _ in range(len(fourier_feature_types))]
for M in Ms:
    sum_mse = [0]*len(fourier_feature_types)
    sum_nmse = [0]*len(fourier_feature_types)
    sum_mnlp = [0]*len(fourier_feature_types)
    sum_time = [0]*len(fourier_feature_types)
    for i, fftype in enumerate(fourier_feature_types):
        tf1, tf2, tf3 = False, False, False
        if("z" in fftype):
            tf1 = True
        if("ph" in fftype):
            tf2 = True
        if("lr" in fftype):
            tf3 = True
            rank = 4
        else:
            rank = "full"
        funcs = None
        for round in range(trials_per_model):
            model = SCFGP(rank, M, tf1, tf2, tf3, msg=False)
            if(funcs is None):
                model.fit(X_train, Y_train, X_test, Y_test)
                funcs = (model.train_func, model.pred_func)
            else:
                model.fit(X_train, Y_train, X_test, Y_test, funcs)
            sum_mse[i] += model.TsMSE
            sum_nmse[i] += model.TsNMSE
            sum_mnlp[i] += model.TsMNLP
            sum_time[i] += model.TrTime
            print("\n>>>", model.NAME)
            print("    Mean Square Error\t\t\t\t= %.3f%s(Avg. %.4f)"%(
                model.TsMSE, (2-int(np.log10(abs(model.TsMSE))))*"\t",
                sum_mse[i]/(round+1)))
            print("    Normalized Mean Square Error\t= %.3f%s(Avg. %.4f)"%(
                model.TsNMSE, (2-int(np.log10(abs(model.TsNMSE))))*"\t",
                sum_nmse[i]/(round+1)))
            print("    Mean Negative Log Probability\t= %.3f%s(Avg. %.4f)"%(
                model.TsMNLP, (2-int(np.log10(abs(model.TsMNLP))))*"\t",
                sum_mnlp[i]/(round+1)))
            print("    Training Time\t\t\t\t\t= %.3f%s(Avg. %.4f)"%(
                model.TrTime, (2-int(np.log10(abs(model.TrTime))))*"\t",
                sum_time[i]/(round+1)))
            plt.close()
        MSE[i].append(sum_mse[i]/trials_per_model)
        NMSE[i].append(sum_nmse[i]/trials_per_model)
        MNLP[i].append(sum_mnlp[i]/trials_per_model)
        TIME[i].append(sum_time[i]/trials_per_model)
f = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
ax = f.add_subplot(111)
maxv, minv = 0, 1e5
labels = ["Full Rank", "Rank="+str(rank)]
lines = []
for j in range(len(MSE)):
    for i in range(len(Ms)):
        maxv = max(maxv, MSE[j][i])
        minv = min(minv, MSE[j][i])
        ax.text(Ms[i], MSE[j][i], '%.2f' % (MSE[j][i]), fontsize=5)
    line, = ax.plot(Ms, MSE[j], marker=Line2D.filled_markers[0], label=labels[j])
    lines.append(line)
ax.set_ylim([minv-(maxv-minv)*0.15,maxv+(maxv-minv)*0.45])
plt.title('Mean Square Error', fontsize=20)
plt.xlabel('M', fontsize=13)
plt.ylabel('MSE', fontsize=13)
legend = f.legend(handles=lines, labels=labels, loc=1, shadow=True)
plt.savefig('low_vs_full_housing_mse.png')
f = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
ax = f.add_subplot(111)
maxv, minv = 0, 1e5
lines = []
for j in range(len(MNLP)):
    for i in range(len(Ms)):
        maxv = max(maxv, MNLP[j][i])
        minv = min(minv, MNLP[j][i])
        ax.text(Ms[i], MNLP[j][i], '%.2f' % (MNLP[j][i]), fontsize=5)
    line, = plt.plot(Ms, MNLP[j], marker=Line2D.filled_markers[0], label=labels[j])
    lines.append(line)
ax.set_ylim([minv-(maxv-minv)*0.15,maxv+(maxv-minv)*0.45])
plt.title('Mean Negative Log Probablity', fontsize=20)
plt.xlabel('M', fontsize=13)
plt.ylabel('MNLP', fontsize=13)
legend = f.legend(handles=lines, labels=labels, loc=1, shadow=True)
plt.savefig('low_vs_full_housing_mnlp.png')
f = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
ax = f.add_subplot(111)
maxv, minv = 0, 1e5
lines = []
for j in range(len(TIME)):
    for i in range(len(Ms)):
        maxv = max(maxv, TIME[j][i])
        minv = min(minv, TIME[j][i])
        ax.text(Ms[i], TIME[j][i], '%.2f' % (TIME[j][i]), fontsize=5)
    line, = plt.plot(Ms, TIME[j], marker=Line2D.filled_markers[0], label=labels[j])
    lines.append(line)
ax.set_ylim([minv-(maxv-minv)*0.15,maxv+(maxv-minv)*0.45])
plt.title('Training Time', fontsize=20)
plt.xlabel('M', fontsize=13)
plt.ylabel('Time (s)', fontsize=13)
legend = f.legend(handles=lines, labels=labels, loc=1, shadow=True)
plt.savefig('low_vs_full_housing_time.png')
plt.show()