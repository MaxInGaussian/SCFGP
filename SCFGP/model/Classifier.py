################################################################################
#  Regression Model: Sparsely Correlated Fourier Features Based Gaussian Process
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

class Classifier(object):
    
    " SCFGP as Classifier "

    ID, NAME, seed, opt, msg, SCORE = "", "", None, None, True, 0
    use_inducing_inputs = True
    use_optimized_phases = True
    add_low_rank_freq = True
    precompute_c_method = None
    R, M, N, D, P = -1, -1, -1, -1, -1
    pre_c, X, y, y_lbls, Xs, ys, ys_pred, ys_pred_std, ys_pred_lbls = [None]*9
    hyper, EtawKiPhi, alpha, train_func, pred_func = [None]*5
    TrCost, TrMSE, TrNMSE, TsMAE, TsMSE, TsRMSE, TsNMSE = [np.inf]*7
    
    def __init__(self,
                 rank=-1,
                 feature_size=-1,
                 use_inducing_inputs=True,
                 use_optimized_phases=True,
                 add_low_rank_freq=True,
                 precompute_c_method=None,
                 fftype=None,
                 msg=True):
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
        self.y_nml = Normalizer("categorize")
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
        _3Fx = (lambda F, x: T.concatenate([
            F(x[i])[None, :, :] for i in range(self.P)], axis=0))
        _3T = (lambda x: T.transpose(x, (0, 2, 1)))
        _3dot = (lambda x, y: T.sum(x[:, :, :, None]*y[:, None, :, :], 2))
        _3diag = (lambda x: _3Fx(T.diag, x))
        _3diagonal = (lambda x: T.concatenate([
            T.diag(x[i])[None, :] for i in range(self.P)], axis=0))
        _3eye = (lambda A:_3diag(T.ones_like(A[:, :, 0])))
        _3hstack = (lambda LIST: T.concatenate(LIST, axis=2))
        _3chol = (lambda x: _3Fx(sT.cholesky, x))
        _3inv = (lambda x: _3Fx(sT.matrix_inverse, x))
        self.message("Compiling SCFGP theano model...")
        hyper = T.dvector('hyper')
        X, Xs = T.dmatrices('X', 'Xs')
        Y, EtawKiPhi, alpha = T.dtensor3s('Y', 'EtawKiPhi', 'alpha')
        N, S, P = X.shape[0], Xs.shape[0], Y.shape[0]
        tX, tXs = T.ones((self.P, 1, 1))*X, T.ones((self.P, 1, 1))*Xs
        t_ind = 0
        b = T.reshape(hyper[t_ind:t_ind+self.P], (self.P, 1, 1))
        t_ind += self.P
        sig_f, sig2_f = T.exp(b), T.exp(2*b)
        theta = T.reshape(hyper[t_ind:t_ind+self.P], (self.P, 1, 1))
        t_ind += self.P
        C = T.sum(T.exp(theta))
        softmax = T.exp(theta)/C
        u = Y-softmax
        w = 1/(softmax*(1-softmax))
        wi = 1/w
        t = theta+w*u
        if(self.R == "full"):
            Omega = T.reshape(hyper[t_ind:t_ind+self.D*self.M*self.P],
                (self.P, self.D, self.M))
            t_ind += self.D*self.M*self.P
        else:
            if(self.precompute_c_method is None):
                c = T.reshape(hyper[t_ind:t_ind+self.D*self.R*self.P],
                    (self.P, self.D, self.R))
            else:
                c = theano.shared(self.pre_c)[None, :, :]+\
                    T.reshape(hyper[t_ind:t_ind+self.D*self.R*self.P],
                    (self.P, self.D, self.R))
            t_ind += self.D*self.R*self.P
            d = T.reshape(hyper[t_ind:t_ind+self.M*self.R*self.P],
                (self.P, self.M, self.R));t_ind+=self.M*self.R*self.P
            Omega = _3dot(c, _3T(d))
        if(self.use_inducing_inputs):
            Z = T.reshape(hyper[t_ind:t_ind+self.D*self.M*self.P],
                (self.P, self.D, self.M))[:, None, :, :]
            t_ind += self.D*self.M*self.P
            XOmega = T.sum((tX[:, :, :, None]-Z)*Omega[:, None, :, :], 2)
            XsOmega = T.sum((tXs[:, :, :, None]-Z)*Omega[:, None, :, :], 2)
        else:
            XOmega, XsOmega = _3dot(tX, Omega), _3dot(tXs, Omega)
        if(self.use_optimized_phases):
            ph = T.reshape(hyper[t_ind:t_ind+2*self.M*self.P],
                (self.P, 1, 2*self.M))
            t_ind += self.M*2
            cosXOmega = T.cos(XOmega+T.tile(ph[:, :, :self.M], (1, N, 1)))
            cosXsOmega = T.cos(XsOmega+T.tile(ph[:, :, :self.M], (1, S, 1)))
            sinXOmega = T.sin(XOmega+T.tile(ph[:, :, self.M:], (1, N, 1)))
            sinXsOmega = T.sin(XsOmega+T.tile(ph[:, :, self.M:], (1, S, 1)))
        else:
            sinXOmega, sinXsOmega = T.sin(XOmega), T.sin(XsOmega)
            cosXOmega, cosXsOmega = T.cos(XOmega), T.cos(XsOmega)
        const = sig_f*T.sqrt(2./self.M)
        Fourier_features_list = [[sinXOmega, cosXOmega], [sinXsOmega, cosXsOmega]]
        if(self.add_low_rank_freq):
            Xc, Xsc = _3dot(tX, c), _3dot(tXs, c)
            sinXc, sinXsc = T.sin(Xc), T.sin(Xsc)
            cosXc, cosXsc = T.cos(Xc), T.cos(Xsc)
            Fourier_features_list[0].extend([sinXc, cosXc])
            Fourier_features_list[1].extend([sinXsc, cosXsc])
        Phi = const*_3hstack(Fourier_features_list[0])
        Phis = const*_3hstack(Fourier_features_list[1])
        Eta = _3T(wi*Phi)
        Gamma = _3dot(Eta, Phi)
        A = Gamma+_3eye(Gamma)
        L = _3chol(A)
        Li = _3inv(L)
        LiEta = _3dot(Li, Eta)
        wKi = _3eye(tX)-_3dot(Phi, _3dot(_3T(Li), LiEta))
        EtawKi = _3dot(Eta, wKi)
        EtawKiPhi_ = _3dot(EtawKi, Phi)
        alpha_ = _3dot(EtawKi, t)
        y_pred = _3dot(Phi, alpha_)
        mu_omega = T.mean(Omega, axis=2)
        sigma_omega = T.std(Omega, axis=2)
        NLML = 1./2*_3dot(_3dot(_3T(t), wi*wKi), t).sum()+\
            T.log(_3diagonal(L)).sum()-\
            1./2*_3dot(_3T(u), w*u).sum()-(Y*theta-T.log(C)).sum()-\
            1./2*(sigma_omega+mu_omega**2-T.log(sigma_omega)-1).sum()
        cost = NLML/N
        dhyper = T.grad(cost, hyper)
        train_input = [X, Y, hyper]
        train_input_name = ['X', 'Y', 'hyper']
        train_output = [EtawKiPhi_, alpha_, y_pred, cost, dhyper]
        train_output_name = ['EtawKiPhi', 'alpha', 'y_pred', 'cost', 'dhyper']
        self.train_func = theano.function(
            train_input, train_output, on_unused_input='ignore')
        ys_pred = _3dot(Phis, alpha)
        prod = _3T(Phis)-_3dot(EtawKiPhi, _3T(Phis))
        ys_pred_std = T.sqrt(_3diagonal(_3dot(Phis, prod)))
        pred_input = [Xs, hyper, EtawKiPhi, alpha]
        pred_input_name = ['Xs', 'hyper', 'EtawKiPhi', 'alpha']
        pred_output = [ys_pred, ys_pred_std]
        pred_output_name = ['ys_pred', 'ys_pred_std']
        self.pred_func = theano.function(
            pred_input, pred_output, on_unused_input='ignore')
        self.message("done.")

    def init_model(self):
        b = np.array([-2*np.log(4.)]*self.P)
        best_hyper, min_cost = None, np.inf
        for _ in range(20):
            theta = np.random.randn(self.P)
            hyper_list = [b.ravel(), theta.ravel()]
            if(self.R == "full"):
                Omega = np.random.randn(self.P, self.D, self.M)
                hyper_list.append(Omega.ravel())
            else:
                c = np.random.rand(self.P, self.D, self.R)
                d = np.random.randn(self.P, self.M, self.R)
                hyper_list.extend([c.ravel(), d.ravel()])
            cost = 0
            if(self.use_inducing_inputs):
                Z = np.random.randn(self.P, self.D, self.M)
                hyper_list.append(Z.ravel())
            if(self.use_optimized_phases):
                ph = np.random.rand(self.P, self.M*2)*2*np.pi
                hyper_list.append(ph.ravel())
            hyper = np.concatenate(hyper_list)
            EtawKiPhi, alpha, y_pred, cost, _ = self.train_func(self.X, self.y, hyper)
            self.message("Random parameters yield cost:", cost)
            if(cost < min_cost):
                min_cost = cost
                best_hyper = hyper.copy()
        self.hyper = best_hyper.copy()
        self.EtawKiPhi, self.alpha, self.y_pred, self.cost, _ = self.train_func(
            self.X, self.y, self.hyper)

    def fit(self, X, y, Xs=None, ys=None, funcs=None, opt=None, callback=None,
        plot_training=False, plot_1d_function=False):
        self.y_lbls = y.copy()
        self.X_nml.fit(X)
        self.y_nml.fit(y)
        self.X = self.X_nml.forward_transform(X)
        self.y = self.y_nml.forward_transform(y)
        self.N, self.D = self.X.shape
        self.P = self.y.shape[0]
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
            opt = Optimizer("smorms3", [0.05], 500, 10, 1e-2, False)
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
                    Xplot, self.hyper, self.EtawKiPhi, self.alpha)
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
            self.EtawKiPhi, self.alpha, self.y_pred, self.TrCost, dhyper =\
                self.train_func(self.X, self.y, hyper)
            self.y_pred_lbls = self.y_nml.backward_transform(self.y_pred)
            self.TrACC = np.mean(self.y_pred_lbls==self.y_lbls)
            self.TrNMSE = np.mean((self.y_pred-self.y)**2.)/np.var(self.y)
            self.message("="*20, "TRAINING ITERATION", iter, "="*20)
            self.message(self.NAME, " TrCost = %.4f"%(self.TrCost))
            self.message(self.NAME, "  TrACC = %.4f"%(self.TrACC))
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
        self.ys_pred, self.ys_pred_std = self.pred_func(
            self.Xs, self.hyper, self.EtawKiPhi, self.alpha)
        self.ys_pred_lbls = self.y_nml.backward_transform(self.ys_pred)
        if(ys is not None):
            self.ys = self.y_nml.forward_transform(ys)
            self.ys_lbls = ys.copy()
            self.TsACC = np.mean(self.ys_pred_lbls==self.ys_lbls)
            self.TsNMSE = np.mean((self.ys_pred-self.ys)**2.)/np.var(self.ys)
            self.SCORE = self.TsACC
            self.message(self.NAME, "  TsACC = %.4f"%(self.TsACC))
            self.message(self.NAME, " TsNMSE = %.4f"%(self.TsNMSE))
            self.message(self.NAME, "  SCORE = %.4f"%(self.SCORE))
        return self.ys_pred_lbls, self.ys_pred, self.ys_pred_std

    def save(self, path):
        import pickle
        prior_setting = (self.seed, self.R, self.M,
            self.use_inducing_inputs, self.use_optimized_phases,
            self.add_low_rank_freq, self.precompute_c_method)
        train_data = (self.X, self.y)
        normalizers = (self.X_nml, self.y_nml)
        computed_matrices = (self.pre_c, self.hyper, self.EtawKiPhi, self.alpha)
        performances = (self.TrCost, self.TrACC, self.TrNMSE, self.TrTime, 
            self.TsACC, self.TsNMSE, self.SCORE)
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
        self.pre_c, self.hyper, self.EtawKiPhi, self.alpha = load_pack[3]
        self.TrCost, self.TrACC, self.TrNMSE, self.TrTime = load_pack[4][:4]
        self.TsACC, self.TsNMSE, self.SCORE = load_pack[4][4:7]






