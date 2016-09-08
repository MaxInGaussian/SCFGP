################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys, os, string, time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import theano
import theano.tensor as T
import theano.sandbox.linalg as sT

from .FrequencyKernels import FrequencyKernel
from .Optimizers import Optimizer
from .Normalizers import Normalizer

theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False

class SCFGP(object):
    
    """
    Sparsely Correlated Fourier Features Based Gaussian Process
    """

    ID, NAME, seed, verbose = "", "", None, True
    freq_kern, iduc_kern, X_nml, y_nml = [None]*4
    R, M, N, D, FKP, IKP = -1, -1, -1, -1, -1, -1
    X, y, Xs, ys, hyper, Ri, alpha, Omega, train_func, pred_func = [None]*10
    SCORE, COST, TrMSE, TrNMSE, TsMAE, TsMSE, TsRMSE, TsNMSE, TsMNLP = [0]*9
    
    def __init__(self, rank=-1, feature_size=-1,
                 freq_kern="rbf", iduc_kern="per", verbose=True):
        self.R = rank
        self.M = feature_size
        self.freq_kern = FrequencyKernel(freq_kern, self.R)
        self.FKP = self.freq_kern.params_size
        self.iduc_kern = FrequencyKernel(iduc_kern, self.R)
        self.IKP = self.iduc_kern.params_size
        self.X_nml = Normalizer("linear")
        self.y_nml = Normalizer("standardize")
        self.verbose = verbose
        self.generate_ID()
    
    def message(self, *arg):
        if(self.verbose):
            print(" ".join(map(str, arg)))
            sys.stdout.flush()
    
    def generate_ID(self):
        self.ID = ''.join(
            chr(np.random.choice([ord(c) for c in (
                string.ascii_uppercase+string.digits)])) for _ in range(5))
        self.NAME = "SCFGP (Rank=%s, Feature Size=%d)"%(str(self.R), self.M)
        self.seed = np.prod([ord(c) for c in self.ID])%4294967291
        npr.seed(self.seed)
    
    def build_theano_model(self):
        epsilon = 1e-8
        kl = lambda mu, sig: sig+mu**2-T.log(sig)
        self.message("Compiling SCFGP theano model...")
        X, y, Xs, alpha, Ri = T.dmatrices('X', 'Y', 'Xs', 'alpha', 'Ri')
        N, S = X.shape[0], Xs.shape[0]
        hyper = T.dvector('hyper')
        t_ind = 0
        a = hyper[0];t_ind+=1
        b = hyper[1];t_ind+=1
        sig_n, sig_f = T.exp(a), T.exp(b)
        sig2_n, sig2_f = sig_n**2, sig_f**2
        l = hyper[t_ind:t_ind+2*self.D*self.R];t_ind+=2*self.D*self.R
        L = T.reshape(l[self.D*self.R:], (self.D, self.R))
        Z_L = T.reshape(l[:self.D*self.R], (self.D, self.R))
        f = hyper[t_ind:t_ind+2*self.M*self.R];t_ind+=2*self.M*self.R
        F = T.reshape(f[:self.M*self.R], (self.M, self.R))
        Z_F = T.reshape(f[self.M*self.R:], (self.M, self.R))
        fkp = hyper[t_ind:t_ind+self.FKP];t_ind+=self.FKP
        self.freq_kern.set_params(fkp)
        Omega = self.freq_kern.fit(L, F)
        ikp = hyper[t_ind:t_ind+self.IKP];t_ind+=self.IKP
        self.iduc_kern.set_params(ikp)
        Z = self.iduc_kern.fit(Z_L, Z_F)
        theta = hyper[t_ind:t_ind+self.M+self.R];t_ind+=self.M+self.R
        Theta = T.reshape(theta[:self.M], (1, self.M))
        Theta_L = T.reshape(theta[self.M:], (1, self.R))
        FF = T.dot(X, Omega)+(Theta-T.sum(Z*Omega, 0)[None, :])
        FF_L = T.dot(X, L)+(Theta_L-T.sum(Z_L*L, 0)[None, :])
        Phi = sig_f*T.sqrt(2./self.M)*T.cos(T.concatenate((FF, FF_L), 1))
        PhiTy = T.dot(Phi.T, y)
        PhiTPhi = T.dot(Phi.T, Phi)
        A = PhiTPhi+(sig2_n+epsilon)*T.identity_like(PhiTPhi)
        R = sT.cholesky(A)
        t_Ri = sT.matrix_inverse(R)
        GTy = T.dot(t_Ri, PhiTy)
        t_alpha = T.dot(t_Ri.T, GTy)
        mu_f = T.dot(Phi, t_alpha)
        mu_w = T.sum(T.mean(Omega, axis=1))
        sig_w = T.sum(T.std(Omega, axis=1))
        cost = 2*T.log(T.diagonal(R)).sum()/N+\
            1./sig2_n/N*((y**2).sum()-(GTy**2).sum())+(1-self.M/N)*a
        penelty = kl(mu_w, sig_w)
        cost += penelty/N
        dhyper = T.grad(cost, hyper)
        train_input = [X, y, hyper]
        train_input_name = ['X', 'y', 'hyper']
        train_output = [t_alpha, t_Ri, Omega, mu_f, cost, dhyper]
        train_output_name = ['alpha', 'Ri', 'Omega', 'mu_f', 'cost', 'dhyper']
        self.train_func = theano.function(
            train_input, train_output, on_unused_input='ignore')
        FFs = T.dot(Xs, Omega)+(Theta-T.sum(Z*Omega, 0)[None, :])
        FFs_L = T.dot(Xs, L)+(Theta_L-T.sum(Z_L*L, 0)[None, :])
        Phis = sig_f*T.sqrt(2./self.M)*T.cos(T.concatenate((FFs, FFs_L), 1))
        mu_pred = T.dot(Phis, alpha)
        std_pred = sig_n*(1+(T.dot(Phis, Ri.T)**2).sum(1))**0.5
        pred_input = [Xs, hyper, alpha, Ri]
        pred_input_name = ['Xs', 'hyper', 'alpha', 'Ri']
        pred_output = [mu_pred, std_pred]
        pred_output_name = ['mu_pred', 'std_pred']
        self.pred_func = theano.function(
            pred_input, pred_output, on_unused_input='ignore')
        self.message("done.")

    def init_model(self):
        best_hyper, min_cost = None, np.inf
        for _ in range(20):
            a_and_b = np.random.randn(2)
            l = np.random.rand(2*self.D*self.R)
            f = np.random.rand(2*self.M*self.R)
            kps = np.random.rand(self.FKP+self.IKP)
            theta = 2*np.pi*np.random.rand(self.M+self.R)
            hyper = np.concatenate((a_and_b, l, f, kps, theta))
            alpha, Ri, Omega, mu_f, cost, _ =\
                self.train_func(self.X, self.y, hyper)
            self.message("Random parameters yield cost:", cost)
            if(cost < min_cost):
                min_cost = cost
                best_hyper = hyper.copy()
        self.hyper = best_hyper.copy()
        self.alpha, self.Ri, self.Omega, self.mu_f, self.cost, _ =\
            self.train_func(self.X, self.y, self.hyper)

    def fit(self, X, y, Xs=None, ys=None, funcs=None, opt=None, callback=None,
        plot_matrices=False, plot_training=False, plot_1d_function=False):
        self.X_nml.fit(X)
        self.y_nml.fit(y)
        self.X = self.X_nml.forward_transform(X)
        self.y = self.y_nml.forward_transform(y)
        self.N, self.D = self.X.shape
        if(funcs is None):
            self.build_theano_model()
        else:
            self.train_func, self.pred_func = funcs
        if(Xs is not None and ys is not None):
            self.Xs = self.X_nml.forward_transform(Xs)
            self.ys = self.y_nml.forward_transform(ys)
        else:
            plot_training = False
        train_start_time = time.time()
        self.init_model()
        if(opt is None):
            opt = Optimizer("adam", [0.01/self.R, 0.9, 0.9], 500, 8, 1e-4, True)
        plt.close()
        if(plot_matrices):
            plot_mat_fig = plt.figure(facecolor='white', dpi=120)
            plot_mat_ax = plot_mat_fig.add_subplot(111)
            plot_mat_ax.set_title('Omega')
        if(plot_training):
            iter_list = []
            cost_list = []
            train_mse_list = []
            test_mse_list = []
            plot_train_fig, plot_train_axarr = plt.subplots(
                2, figsize=(8, 6), facecolor='white', dpi=120)
            plot_train_fig.suptitle(self.NAME, fontsize=15)
            plt.xlabel('# iteration', fontsize=13)
        if(plot_1d_function):
            plot_1d_fig = plt.figure(facecolor='white', dpi=120)
            plot_1d_fig.suptitle(self.NAME, fontsize=15)
            plot_1d_ax = plot_1d_fig.add_subplot(111)
            plot_1d_ax.set_title('Omega')
        def animate(i):
            if(plot_matrices):
                plot_mat_ax.cla()
                plot_mat_ax.imshow(self.Omega, origin='lower')
            if(plot_training):
                if(len(iter_list) > 100):
                    iter_list.pop(0)
                    cost_list.pop(0)
                    train_mse_list.pop(0)
                    test_mse_list.pop(0)
                plot_train_axarr[0].cla()
                plot_train_axarr[0].plot(iter_list, cost_list,
                    color='r', linewidth=2.0, label='Training cost')
                plot_train_axarr[1].cla()
                plot_train_axarr[1].plot(iter_list, train_mse_list,
                    color='b', linewidth=2.0, label='Training MSE')
                plot_train_axarr[1].plot(iter_list, test_mse_list,
                    color='g', linewidth=2.0, label='Testing MSE')
                handles, labels = plot_train_axarr[0].get_legend_handles_labels()
                plot_train_axarr[0].legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(0.5, 1.05), ncol=1, fancybox=True)
                handles, labels = plot_train_axarr[1].get_legend_handles_labels()
                plot_train_axarr[1].legend(handles, labels, loc='upper center',
                    bbox_to_anchor=(0.5, 1.05), ncol=2, fancybox=True)
            if(plot_1d_function):
                plot_1d_ax.cla()
                pts = 300
                errors = [0.25, 0.39, 0.52, 0.67, 0.84, 1.04, 1.28, 1.64, 2.2]
                Xplot = np.linspace(-0.1, 1.1, pts)[:, None]
                mu, std = self.pred_func(
                    Xplot, self.hyper, self.alpha, self.Ri)
                mu = mu.ravel()
                std = std.ravel()
                for er in errors:
                    plot_1d_ax.fill_between(Xplot[:, 0], mu-er*std, mu+er*std,
                                    alpha=((3-er)/5.5)**1.7, facecolor='blue',
                                    linewidth=0.0)
                plot_1d_ax.plot(Xplot[:, 0], mu, alpha=0.8, c='black')
                plot_1d_ax.errorbar(self.X[:, 0],
                    self.y.ravel(), fmt='r.', markersize=5, alpha=0.6)
                yrng = self.y.max()-self.y.min()
                plot_1d_ax.set_ylim([
                    self.y.min()-0.5*yrng, self.y.max() + 0.5*yrng])
                plot_1d_ax.set_xlim([-0.1, 1.1])
        def train(iter, hyper):
            self.iter = iter
            self.hyper = hyper.copy()
            self.alpha, self.Ri, self.Omega, mu_f, self.COST, dhyper =\
                self.train_func(self.X, self.y, hyper)
            self.mu_f = self.y_nml.backward_transform(mu_f)
            self.TrMSE = np.mean((self.mu_f-y)**2.)
            self.TrNMSE = self.TrMSE/np.var(y)
            self.message("="*20, "TRAINING ITERATION", iter, "="*20)
            self.message(self.NAME, " COST = %.4f"%(self.COST))
            self.message(self.NAME, "  TrMSE = %.4f"%(self.TrMSE))
            self.message(self.NAME, " TrNMSE = %.4f"%(self.TrNMSE))
            if(Xs is not None and ys is not None):
                self.predict(Xs, ys)
            if(callback is not None):
                callback()
            if(iter == -1):
                return
            if(plot_matrices):
                plt.pause(0.1)
            if(plot_training):
                iter_list.append(iter)
                cost_list.append(self.COST)
                train_mse_list.append(self.TrMSE)
                test_mse_list.append(self.TsMSE)
                plt.pause(0.01)
            if(plot_1d_function):
                plt.pause(0.05)
            return self.COST, self.TrNMSE, dhyper
        if(plot_matrices):
            ani = anm.FuncAnimation(plot_mat_fig, animate, interval=500)
        if(plot_training):
            ani = anm.FuncAnimation(plot_train_fig, animate, interval=500)
        if(plot_1d_function):
            ani = anm.FuncAnimation(plot_1d_fig, animate, interval=500)
        opt.run(train, self.hyper)
        train_finish_time = time.time()
        self.TrTime = train_finish_time-train_start_time

    def predict(self, Xs, ys=None):
        self.Xs = self.X_nml.forward_transform(Xs)
        mu_pred, std_pred = self.pred_func(
            self.Xs, self.hyper, self.alpha, self.Ri)
        self.mu_pred = self.y_nml.backward_transform(mu_pred)
        self.std_pred = std_pred[:, None]*self.y_nml.data["std"]
        if(ys is not None):
            self.ys = self.y_nml.forward_transform(ys)
            self.TsMAE = np.mean(np.abs(self.mu_pred-ys))
            self.TsMSE = np.mean((self.mu_pred-ys)**2.)
            self.TsRMSE = np.sqrt(np.mean((self.mu_pred-ys)**2.))
            self.TsNMSE = self.TsMSE/np.var(ys)
            self.TsMNLP = 0.5*np.mean(((ys-self.mu_pred)/\
                self.std_pred)**2+np.log(2*np.pi*self.std_pred**2))
            self.SCORE = np.exp(-self.TsMNLP)/self.TsNMSE
            self.message(self.NAME, "  TsMAE = %.4f"%(self.TsMAE))
            self.message(self.NAME, "  TsMSE = %.4f"%(self.TsMSE))
            self.message(self.NAME, " TsRMSE = %.4f"%(self.TsRMSE))
            self.message(self.NAME, " TsNMSE = %.4f"%(self.TsNMSE))
            self.message(self.NAME, " TsMNLP = %.4f"%(self.TsMNLP))
            self.message(self.NAME, "  SCORE = %.4f"%(self.SCORE))
        return self.mu_pred, self.std_pred

    def save(self, path):
        import pickle
        prior_setting = (self.seed, self.R, self.M)
        init_objects = (self.freq_kern, self.iduc_kern, self.X_nml, self.y_nml)
        train_data = (self.X, self.y)
        matrices = (self.hyper, self.alpha, self.Ri)
        metrics = (self.SCORE, self.COST, self.TrMSE, self.TrNMSE, self.TrTime, 
            self.TsMAE, self.TsMSE, self.TsRMSE, self.TsNMSE, self.TsMNLP)
        save_pack = [prior_setting, init_objects, train_data, matrices, metrics]
        with open(path, "wb") as save_f:
            pickle.dump(save_pack, save_f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle
        with open(path, "rb") as load_f:
            load_pack = pickle.load(load_f)
        self.seed, self.R, self.M = load_pack[0]
        self.generate_ID()
        npr.seed(self.seed)
        self.freq_kern, self.iduc_kern, self.X_nml, self.y_nml = load_pack[1]
        self.FKP = self.freq_kern.params_size
        self.IKP = self.iduc_kern.params_size
        self.X, self.y = load_pack[2]
        self.N, self.D = self.X.shape
        self.build_theano_model()
        self.hyper, self.alpha, self.Ri = load_pack[3]
        [self.SCORE, self.COST, self.TrMSE, self.TrNMSE,
            self.TrTime, self.TsMAE, self.TsMSE, self.TsRMSE,
                self.TsNMSE, self.TsMNLP] = load_pack[4]






