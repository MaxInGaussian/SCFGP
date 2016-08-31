################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
#  Author: Max W. y. Lam (maxingaussian@gmail.com)
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

class GeneralizedPredictor(object):
    
    " SCFGP as GeneralizedPredictor "

    ID, NAME, seed, opt, msg, SCORE = "", "", None, None, True, 0
    use_inducing_inputs = True
    use_optimized_phases = True
    add_low_rank_freq = True
    precompute_c_method = None
    R, M, N, D, P = -1, -1, -1, -1, -1
    pre_c, X, y, Xs, ys, hyper, Li, alpha, train_func, pred_func = [None]*10
    TrCost, TrMSE, TrNMSE, TsMAE, TsMSE, TsRMSE, TsNMSE, TsMNLP = [np.inf]*8
    
    def __init__(self,
                 likelihood=None,
                 rank=-1,
                 feature_size=-1,
                 use_inducing_inputs=True,
                 use_optimized_phases=True,
                 add_low_rank_freq=True,
                 precompute_c_method=None,
                 fftype=None,
                 msg=True):
        self.likelihood = likelihood
        self.R = rank
        self.M = feature_size
        if(fftype is None):
            self.use_inducing_inputs = use_inducing_inputs
            self.use_optimized_phases = use_optimized_phases
            self.add_low_rank_freq = add_low_rank_freq
        else:
            self.use_inducing_inputs = False
            self.use_optimized_phases = False
            self.add_low_rank_freq = False
            if("z" in fftype.lower()):
                self.use_inducing_inputs = True
            if("ph" in fftype.lower()):
                self.use_optimized_phases = True
            if(rank != "full"):
                self.add_low_rank_freq = True
        if(self.R != "full"):
            self.precompute_c_method = precompute_c_method
        else:
            self.precompute_c_method = None
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
        self.NAME = "SCFGP {Rank=%s, Feature Size=%d, Feature Type=%s}"%(
            str(self.R), self.M, self.fftype)
        self.seed = np.prod([ord(c) for c in self.ID])%4294967291
        npr.seed(self.seed)
    
    def build_theano_model(self):
        self.message("Compiling SCFGP theano model...")
        X, y, Xs, alpha, beta = T.dmatrices('X', 'y', 'Xs', 'alpha', 'beta')
        N, P, S = X.shape[0], y.shape[1], Xs.shape[0]
        hyper = T.dvector('hyper')
        t_ind = 0
        a = hyper[t_ind:t_ind+1];t_ind+=1
        sig_f = T.exp(a)
        sig2_f = sig_f**2
        m = T.reshape(hyper[t_ind:t_ind+self.N*self.P],
            (self.N, self.P));t_ind+=self.N
        w = T.reshape(hyper[t_ind:t_ind+self.N*self.P],
            (self.N, self.P));t_ind+=self.N
        if(self.R == "full"):
            Omega = T.reshape(hyper[t_ind:t_ind+self.D*self.M],
                (self.D, self.M));t_ind+=self.D*self.M
        else:
            if(self.precompute_c_method is None):
                c = T.reshape(hyper[t_ind:t_ind+self.D*self.R],
                    (self.D, self.R));t_ind+=self.D*self.R
            else:
                c = theano.shared(self.pre_c)+\
                    T.reshape(hyper[t_ind:t_ind+self.D*self.R],
                    (self.D, self.R));t_ind+=self.D*self.R
            d = T.reshape(hyper[t_ind:t_ind+self.M*self.R],
                (self.M, self.R));t_ind+=self.M*self.R
            Omega = T.dot(c, d.T)
        if(self.use_inducing_inputs):
            Z = T.reshape(hyper[t_ind:t_ind+self.D*self.M],
                (self.D, self.M))[None, :, :];t_ind+=self.D*self.M
            XOmega = T.sum((X[:, :, None]-Z)*Omega[None, :, :], 1)
            XsOmega = T.sum((Xs[:, :, None]-Z)*Omega[None, :, :], 1)
        else:
            XOmega, XsOmega = T.dot(X, Omega), T.dot(Xs, Omega)
        if(self.use_optimized_phases):
            ph = hyper[t_ind:t_ind+self.M*2];t_ind+=self.M*2
            cosXOmega = T.cos(XOmega+T.tile(ph[None, :self.M], (N, 1)))
            cosXsOmega = T.cos(XsOmega+T.tile(ph[None, :self.M], (S, 1)))
            sinXOmega = T.sin(XOmega+T.tile(ph[None, self.M:], (N, 1)))
            sinXsOmega = T.sin(XsOmega+T.tile(ph[None, self.M:], (S, 1)))
        else:
            sinXOmega, sinXsOmega = T.sin(XOmega), T.sin(XsOmega)
            cosXOmega, cosXsOmega = T.cos(XOmega), T.cos(XsOmega)
        const = sig_f*T.sqrt(2./self.M)
        Fourier_features_list = [[sinXOmega, cosXOmega], [sinXsOmega, cosXsOmega]]
        if(self.add_low_rank_freq):
            Xc, Xsc = X.dot(c), Xs.dot(c)
            sinXc, sinXsc = T.sin(Xc), T.sin(Xsc)
            cosXc, cosXsc = T.cos(Xc), T.cos(Xsc)
            Fourier_features_list[0].extend([sinXc, cosXc])
            Fourier_features_list[1].extend([sinXsc, cosXsc])
        Phi = const*T.concatenate(Fourier_features_list[0], 1)
        Phis = const*T.concatenate(Fourier_features_list[1], 1)
        EPhi = w*Phi
        Gamma = T.dot(Phi.T, EPhi)
        A = Gamma+T.identity_like(Gamma)
        L = sT.cholesky(A)
        Li = sT.matrix_inverse(L)
        B = T.dot(Li, EPhi.T)
        I_BTBW = T.eye(self.N)-T.dot(B.T, (B/w.T))
        alpha_ = T.dot(Phi.T, m)
        beta_ = T.dot(Phi.T, T.dot(I_BTBW, Phi))
        mu_f = T.dot(Phi, alpha_)
        var_f = T.dot(Phi, T.dot(Phi.T, I_BTBW))
        EIAtheataI = self.likelihood.EIAtheataI(mu_f, var_f)
        EItheataI = self.likelihood.EItheataI(mu_f, var_f)
        TIyI = self.likelihood.TIyI(y)
        cost = T.sum(EIAtheataI-TIyI*EItheataI)+T.sum(T.log(T.diagonal(L)))+\
            T.trace(I_BTBW)-1./2*T.sum(T.dot(alpha_.T, alpha_))
        dhyper = T.grad(cost, hyper)
        train_input = [X, y, hyper]
        train_input_name = ['X', 'y', 'hyper']
        train_output = [alpha_, beta_, y_pred, cost, dhyper]
        train_output_name = ['alpha', 'beta', 'y_pred', 'cost', 'dhyper']
        self.train_func = theano.function(
            train_input, train_output, on_unused_input='ignore')
        ys_pred = T.dot(Phis, alpha)
        ys_pred_std = T.dot(Phis, T.dot(T.identity_like(beta)-beta, Phis.T))
        pred_input = [Xs, hyper, alpha, beta]
        pred_input_name = ['Xs', 'hyper', 'alpha', 'beta']
        pred_output = [ys_pred, ys_pred_std]
        pred_output_name = ['ys_pred', 'ys_pred_std']
        self.pred_func = theano.function(
            pred_input, pred_output, on_unused_input='ignore')
        self.message("done.")

    def init_model(self):
        a = -2*np.log(4.)
        best_hyper, min_cost = None, np.inf
        for _ in range(50):
            m = np.random.randn(self.N, self.P)
            g = np.random.randn(self.N, self.P)**2.
            hyper_list = [np.array(a), m, g]
            if(self.R == "full"):
                Omega = np.random.randn(self.D, self.M)
                hyper_list.append(Omega.ravel())
            else:
                c = np.random.rand(self.D, self.R)
                d = np.random.randn(self.M, self.R)
                hyper_list.extend([c.ravel(), d.ravel()])
            cost = 0
            if(self.use_inducing_inputs):
                Z = np.random.randn(self.D, self.M)
                hyper_list.append(Z.ravel())
            if(self.use_optimized_phases):
                ph = np.random.rand(self.M*2)*2*np.pi
                hyper_list.append(ph.ravel())
            hyper = np.concatenate(hyper_list)
            Li, alpha, y_pred, cost, _ = self.train_func(self.X, self.y, hyper)
            self.message("Random parameters yield cost:", cost)
            if(cost < min_cost):
                min_cost = cost
                best_hyper = hyper.copy()
        self.hyper = best_hyper.copy()
        self.Li, self.alpha, self.y_pred, self.cost, _ = self.train_func(
            self.X, self.y, self.hyper)

    def fit(self, X, y, Xs=None, ys=None, funcs=None, opt=None, callback=None,
        plot_training=False, plot_1d_function=False):
        self.X_nml.fit(X)
        self.y_nml.fit(y)
        self.X = self.X_nml.forward_transform(X)
        self.y = self.y_nml.forward_transform(y)
        self.N, self.D = self.X.shape
        _, self.P = self.y.shape
        if(self.R != "full"):
            if(self.precompute_c_method is not None):
                if(self.R > self.D):
                    self.R = self.D
            if(self.precompute_c_method == "rpca"):
                from sklearn.decomposition import RandomizedPCA
                self.pre = RandomizedPCA(n_components=self.R)
            elif(self.precompute_c_method == "rbm"):
                from sklearn.neural_network import BernoulliRBM
                self.pre = BernoulliRBM(n_components=self.R, n_iter=200)
            elif(self.precompute_c_method == "spc"):
                from sklearn.decomposition import DictionaryLearning
                self.pre = DictionaryLearning(n_components=self.R, n_iter=200)
        if(self.precompute_c_method is not None):
            self.pre.fit(self.X)
            self.pre_c = self.pre.components_.T
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
            opt = Optimizer("smorms3", [0.05], 500, 8, 1e-4, [0.05])
            if(self.R != "full"):
                opt.max_cvrg_iter /= 1+self.R/self.D
                opt.cvrg_tol *= 1+self.R/self.D
        plt.close()
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
            plot_1d_fig = plt.figure(figsize=(8, 6), facecolor='white', dpi=120)
            plot_1d_fig.suptitle(self.NAME, fontsize=15)
            plot_1d_ax = plot_1d_fig.add_subplot(111)
        def animate(i):
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
                    Xplot, self.hyper, self.Li, self.alpha)
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
            self.Li, self.alpha, y_pred, self.TrCost, dhyper =\
                self.train_func(self.X, self.y, hyper)
            self.y_pred = self.y_nml.backward_transform(y_pred)
            self.TrMSE = np.mean((self.y_pred-y)**2.)
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
            if(plot_training):
                iter_list.append(iter)
                cost_list.append(self.TrCost)
                train_mse_list.append(self.TrMSE)
                test_mse_list.append(self.TsMSE)
                plt.pause(0.01)
            if(plot_1d_function):
                plt.pause(0.05)
            return self.TrCost, self.TrNMSE, dhyper
        if(plot_training):
            ani = anm.FuncAnimation(plot_train_fig, animate, interval=500)
        if(plot_1d_function):
            ani = anm.FuncAnimation(plot_1d_fig, animate, interval=500)
        opt.run(train, self.hyper)
        train_finish_time = time.time()
        self.TrTime = train_finish_time-train_start_time

    def predict(self, Xs, ys=None):
        self.Xs = self.X_nml.forward_transform(Xs)
        ys_pred, ys_pred_std = self.pred_func(
            self.Xs, self.hyper, self.Li, self.alpha)
        self.ys_pred = self.y_nml.backward_transform(ys_pred)
        self.ys_pred_std = np.tile(ys_pred_std[:, None],
            (self.P,))*self.y_nml.data["std"][None, :]
        if(ys is not None):
            self.ys = self.y_nml.forward_transform(ys)
            self.TsMAE = np.mean(np.abs(self.ys_pred-ys))
            self.TsMSE = np.mean((self.ys_pred-ys)**2.)
            self.TsRMSE = np.sqrt(np.mean((self.ys_pred-ys)**2.))
            self.TsNMSE = self.TsMSE/np.var(ys)
            self.TsMNLP = 0.5*np.log(2*np.pi)+0.5*np.mean(((ys-self.ys_pred)/\
                self.ys_pred_std)**2+np.log(self.ys_pred_std**2))
            self.SCORE = np.exp(-self.TsMNLP)/self.TsNMSE
            self.message(self.NAME, "  TsMAE = %.4f"%(self.TsMAE))
            self.message(self.NAME, "  TsMSE = %.4f"%(self.TsMSE))
            self.message(self.NAME, " TsRMSE = %.4f"%(self.TsRMSE))
            self.message(self.NAME, " TsNMSE = %.4f"%(self.TsNMSE))
            self.message(self.NAME, " TsMNLP = %.4f"%(self.TsMNLP))
            self.message(self.NAME, "  SCORE = %.4f"%(self.SCORE))
        return self.ys_pred, self.ys_pred_std

    def save(self, path):
        import pickle
        prior_setting = (self.seed, self.R, self.M,
            self.use_inducing_inputs, self.use_optimized_phases,
            self.add_low_rank_freq, self.precompute_c_method)
        train_data = (self.X, self.y)
        normalizers = (self.X_nml, self.y_nml)
        computed_matrices = (self.pre_c, self.hyper, self.Li, self.alpha)
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
        self.use_inducing_inputs, self.use_optimized_phases = load_pack[0][3:5]
        self.add_low_rank_freq, self.precompute_c_method = load_pack[0][5:7]
        npr.seed(self.seed)
        self.X, self.y = load_pack[1]
        self.N, self.D = self.X.shape
        _, self.P = self.y.shape
        self.build_theano_model()
        self.X_nml, self.y_nml = load_pack[2]
        self.pre_c, self.hyper, self.Li, self.alpha = load_pack[3]
        self.TrCost, self.TrMSE, self.TrNMSE, self.TrTime = load_pack[4][:4]
        self.TsMAE, self.TsMSE, self.TsRMSE = load_pack[4][4:7]
        self.TsNMSE, self.TsMNLP, self.SCORE = load_pack[4][7:]






