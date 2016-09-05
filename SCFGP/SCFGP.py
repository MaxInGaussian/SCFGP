################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

from __future__ import absolute_import

import sys, os, string, time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.animation as anm
import theano;
import theano.tensor as T;
import theano.sandbox.linalg as sT

from SCFGP.util import Optimizer, Normalizer

theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False

class SCFGP(object):
    
    """
    Sparsely Correlated Fourier Features Based Gaussian Process
    """

    ID, NAME, seed, opt, msg, SCORE = "", "", None, None, True, 0
    use_inducing_inputs = True
    use_optimized_phases = True
    add_low_rank_freq = True
    precompute_c_method = None
    R, M, N, D, P = -1, -1, -1, -1, -1
    pre_c, X, y, Xs, ys, hyper, Ri, alpha, train_func, pred_func = [None]*10
    TrCost, TrMSE, TrNMSE, TsMAE, TsMSE, TsRMSE, TsNMSE, TsMNLP = [np.inf]*8
    
    def __init__(self, rank=-1, feature_size=-1, msg=True):
        self.R = rank
        self.M = feature_size
        self.X_nml = Normalizer("linear")
        self.y_nml = Normalizer("standardize")
        self.msg = msg
        self.generate_ID()
    
    def message(self, *arg):
        if(self.msg):
            print(" ".join(map(str, arg)))
            sys.stdout.flush()
    
    def generate_ID(self):
        self.ID = ''.join(
            chr(np.random.choice([ord(c) for c in (
                string.ascii_uppercase+string.digits)])) for _ in range(5))
        self.fftype = "PH" if self.use_optimized_phases else "F"
        self.fftype += "Z" if self.use_inducing_inputs else ""
        self.NAME = "SCFGP (Rank=%s, Feature Size=%d)"%(str(self.R), self.M)
        self.seed = np.prod([ord(c) for c in self.ID])%4294967291
        npr.seed(self.seed)
    
    def build_theano_model(self):
        epsilon = 1e-8
        self.message("Compiling SCFGP theano model...")
        X, y, Xs, Ri, alpha = T.dmatrices('X', 'Y', 'Xs', 'Ri', 'alpha')
        N, S = X.shape[0], Xs.shape[0]
        hyper = T.dvector('hyper')
        t_ind = 0
        a = hyper[0];t_ind+=1
        b = hyper[1];t_ind+=1
        sig_n, sig_f = T.exp(a), T.exp(b)
        sig2_n, sig2_f = sig_n**2, sig_f**2
        kp = hyper[t_ind:t_ind+2*self.R];t_ind+=2*self.R
        kp1 = kp[:self.R][None, None, :]
        kp2 = kp[self.R:][None, None, :]
        l = hyper[t_ind:t_ind+2*self.D*self.R];t_ind+=2*self.D*self.R
        L = T.reshape(l[self.D*self.R:], (self.D, self.R))
        Z_L = T.reshape(l[:self.D*self.R], (self.D, self.R))
        f = hyper[t_ind:t_ind+2*self.M*self.R];t_ind+=2*self.M*self.R
        F = T.reshape(f[:self.M*self.R], (self.M, self.R))
        Z_F = T.reshape(f[self.M*self.R:], (self.M, self.R))
        Omega = T.sum((L[:, None, :])*(F[None, :, :]+kp1), 2)
        Z = T.sum((Z_L[:, None, :])*(Z_F[None, :, :]+kp2), 2)
        theta = hyper[t_ind:t_ind+self.M+self.R];t_ind+=self.M+self.R
        Theta = T.reshape(theta[:self.M], (1, self.M))
        Theta_L = T.reshape(theta[self.M:], (1, self.R))
        FF = T.dot(X, Omega)+(Theta-T.sum(Z*Omega, 0)[None, :])
        FF_L = T.dot(X, L)+(Theta_L-T.sum(Z_L*L, 0)[None, :])
        Phi = sig_f*T.sqrt(2./self.M)*T.cos(T.concatenate((FF, FF_L), 1))
        PhiTPhi = T.dot(Phi.T, Phi)
        A = PhiTPhi+(sig2_n+epsilon)*T.identity_like(PhiTPhi)
        R = sT.cholesky(A)
        t_Ri = sT.matrix_inverse(R)
        PhiTy = Phi.T.dot(y)
        beta = T.dot(t_Ri, PhiTy)
        t_alpha = T.dot(t_Ri.T, beta)
        mu_f = T.dot(Phi, t_alpha)
        mu_w = (T.sum(Omega, axis=1)+T.sum(L, axis=1))/(self.M+self.R)
        var_w = (T.var(Omega, axis=1)*self.M+\
            T.var(L, axis=1)*self.R)/(self.M+self.R)
        cost = T.log(2*sig2_n*np.pi)+(1./sig2_n*((y**2).sum()-\
            (beta**2).sum())+2*T.log(T.diagonal(R)).sum()-\
                (var_w+mu_w**2-T.log(var_w)-1).sum())/N
        dhyper = T.grad(cost, hyper)
        train_input = [X, y, hyper]
        train_input_name = ['X', 'y', 'hyper']
        train_output = [t_Ri, t_alpha, Omega, mu_f, cost, dhyper]
        train_output_name = ['Ri', 'alpha', 'Omega', 'mu_f', 'cost', 'dhyper']
        self.train_func = theano.function(
            train_input, train_output, on_unused_input='ignore')
        FFs = T.dot(Xs, Omega)+(Theta-T.sum(Z*Omega, 0)[None, :])
        FFs_L = T.dot(Xs, L)+(Theta_L-T.sum(Z_L*L, 0)[None, :])
        Phis = sig_f*T.sqrt(2./self.M)*T.cos(T.concatenate((FFs, FFs_L), 1))
        mu_pred = T.dot(Phis, alpha)
        std_pred = sig_n*(1+(Ri.dot(Phis.T)**2).sum(0).T)**0.5
        pred_input = [Xs, hyper, Ri, alpha]
        pred_input_name = ['Xs', 'hyper', 'Ri', 'alpha']
        pred_output = [mu_pred, std_pred]
        pred_output_name = ['mu_pred', 'std_pred']
        self.pred_func = theano.function(
            pred_input, pred_output, on_unused_input='ignore')
        self.message("done.")

    def init_model(self):
        best_hyper, min_cost = None, np.inf
        for _ in range(20):
            a_and_b = np.random.randn(2)
            kp = np.random.rand(2*self.R)
            l = np.random.rand(2*self.D*self.R)
            f = np.random.rand(2*self.M*self.R)
            theta = 2*np.pi*np.random.rand(self.M+self.R)
            hyper = np.concatenate((a_and_b, kp, l, f, theta))
            Ri, alpha, Omega, mu_f, cost, _ =\
                self.train_func(self.X, self.y, hyper)
            self.message("Random parameters yield cost:", cost)
            if(cost < min_cost):
                min_cost = cost
                best_hyper = hyper.copy()
        self.hyper = best_hyper.copy()
        self.Ri, self.alpha, self.Omega, self.mu_f, self.cost, _ =\
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
        train_start_time = time.time()
        self.init_model()
        if(opt is None):
            opt = Optimizer("smorms3", [1e-1/(self.R**2)], self.R*100, 24, 1e-3)
        if(plot_matrices):
            plot_mat_fig = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
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
                if(Xs is None or ys is None):
                    return
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
                    Xplot, self.hyper, self.Ri, self.alpha)
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
            self.Ri, self.alpha, self.Omega, mu_f, self.TrCost, dhyper =\
                self.train_func(self.X, self.y, hyper)
            self.mu_f = self.y_nml.backward_transform(mu_f)
            self.TrMSE = np.mean((self.mu_f-y)**2.)
            self.TrNMSE = self.TrMSE/np.var(y)
            self.message("="*20, "TRAINING ITERATION", iter, "="*20)
            self.message(self.NAME, " TrCost = %.4f"%(self.TrCost))
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
                cost_list.append(self.TrCost)
                train_mse_list.append(self.TrMSE)
                test_mse_list.append(self.TsMSE)
                plt.pause(0.01)
            if(plot_1d_function):
                plt.pause(0.05)
            return self.TrCost, self.TrNMSE, dhyper
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
            self.Xs, self.hyper, self.Ri, self.alpha)
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
        train_data = (self.X, self.y)
        normalizers = (self.X_nml, self.y_nml)
        computed_matrices = (self.hyper, self.Ri, self.alpha, self.Phi)
        performances = (self.TrCost, self.TrMSE, self.TrNMSE, self.TrTime, 
            self.TsMAE, self.TsMSE, self.TsRMSE, self.TsNMSE,
                self.TsMNLP, self.SCORE)
        save_pack = [prior_setting, train_data, normalizers,
            computed_matrices, performances]
        with open(path, "wb") as save_f:
            pickle.dump(save_pack, save_f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle
        with open(path, "rb") as load_f:
            load_pack = pickle.load(load_f)
        self.seed, self.R, self.M = load_pack[0][:3]
        self.generate_ID()
        npr.seed(self.seed)
        self.X, self.y = load_pack[1]
        self.N, self.D = self.X.shape
        self.build_theano_model()
        self.X_nml, self.y_nml = load_pack[2]
        self.hyper, self.Ri, self.alpha, self.Phi = load_pack[3]
        self.TrCost, self.TrMSE, self.TrNMSE, self.TrTime = load_pack[4][:4]
        self.TsMAE, self.TsMSE, self.TsRMSE = load_pack[4][4:7]
        self.TsNMSE, self.TsMNLP, self.SCORE = load_pack[4][7:]






