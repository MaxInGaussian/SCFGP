################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import sys, os, string, time
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt 
import theano
import theano.tensor as T
import theano.sandbox.linalg as sT

from .Scaler import Scaler
from .Optimizer import Optimizer

theano.config.mode = 'FAST_RUN'
theano.config.optimizer = 'fast_run'
theano.config.reoptimize_unpickled_function = False

class SCFGP(object):
    
    """
    Sparsely Correlated Fourier Features Based Gaussian Process
    """

    ID, NAME, seed, verbose = "", "", None, True
    freq_kern, iduc_kern, X_scaler, y_scaler = [None]*4
    R, M, N, D, FKP, IKP = -1, -1, -1, -1, -1, -1
    X, y, hyper, Ri, alpha, Omega, train_func, pred_func = [None]*8
    
    
    def __init__(self, rank=-1, nfeats=-1, evals=None,
        X_scaling_method='min-max', y_scaling_method='normal', verbose=True):
        self.M = nfeats
        self.R = rank
        self.X_scaler = Scaler(X_scaling_method)
        self.y_scaler = Scaler(y_scaling_method)
        self.evals = {
            "SCORE": ["Model Selection Score", []],
            "COST": ["Hyperparameter Selection Cost", []],
            "MAE": ["Mean Absolute Error", []],
            "NMAE": ["Normalized Mean Absolute Error", []],
            "MSE": ["Mean Square Error", []],
            "NMSE": ["Normalized Mean Square Error", []],
            "MNLP": ["Mean Negative Log Probability", []],
            "TIME(s)": ["Training Time", []],
        } if evals is None else evals
        self.verbose = verbose
    
    def message(self, *arg):
        if(self.verbose):
            print(" ".join(map(str, arg)))
            sys.stdout.flush()
    
    def generate_ID(self):
        self.ID = ''.join(
            chr(npr.choice([ord(c) for c in (
                string.ascii_uppercase+string.digits)])) for _ in range(5))
        self.NAME = "SCFGP (Rank=%s, Feature Size=%d)"%(str(self.R), self.M)
    
    def build_theano_model(self):
        epsilon = 1e-6
        kl = lambda mu, sig: sig+mu**2-T.log(sig)
        snorm_cdf = lambda y: .5*(1+T.erf(y/T.sqrt(2+epsilon)+epsilon))
        X, y, Xs, alpha, Ri = T.dmatrices('X', 'Y', 'Xs', 'alpha', 'Ri')
        N, S = X.shape[0], Xs.shape[0]
        hyper = T.dvector('hyper')
        t_ind = 0
        a = hyper[0];t_ind+=1
        b = hyper[1];t_ind+=1
        c = hyper[2];t_ind+=1
        sig_n, sig_f = T.exp(a), T.exp(b)
        sig2_n, sig2_f = sig_n**2, sig_f**2
        noise = T.log(1+T.exp(c))
        omega = hyper[t_ind:t_ind+self.M*self.D];t_ind+=self.M*self.D
        Omega = T.reshape(omega, (self.D, self.M))
        z = hyper[t_ind:t_ind+self.M*self.D];t_ind+=self.M*self.D
        Z = T.reshape(z, (self.D, self.M))
        theta = hyper[t_ind:t_ind+self.M];t_ind+=self.M
        Theta = T.reshape(theta, (1, self.M))
        FF = T.dot(X, Omega)+(Theta-T.sum(Z*Omega, 0)[None, :])
        Phi = sig_f*T.sqrt(2./self.M)*T.concatenate((T.cos(FF), T.sin(FF)), 1)
        PhiTPhi = T.dot(Phi.T, Phi)
        A = PhiTPhi+(sig2_n+epsilon)*T.identity_like(PhiTPhi)
        R = sT.cholesky(A)
        t_Ri = sT.matrix_inverse(R)
        PhiTy = Phi.T.dot(y)
        beta = T.dot(t_Ri, PhiTy)
        t_alpha = T.dot(t_Ri.T, beta)
        mu_f = T.dot(Phi, t_alpha)
        var_f = (T.dot(Phi, t_Ri.T)**2).sum(1)[:, None]
        disper = noise*(var_f+1)
        mu_w = T.sum(T.mean(Omega, axis=1))
        sig_w = T.sum(T.std(Omega, axis=1))
        hermgauss = np.polynomial.hermite.hermgauss(30)
        x = theano.shared(hermgauss[0])[None, None, :]
        w = theano.shared(hermgauss[1]/np.sqrt(np.pi))[None, None, :]
        herm_f = T.sqrt(2*var_f[:, :, None])*x+mu_f[:, :, None]
        nlk = (0.5*herm_f**2.-y[:, :, None]*herm_f)/disper[:, :, None]+0.5*(
            T.log(2*np.pi*disper[:, :, None])+y[:, :, None]**2/disper[:, :, None])
        enll = w*nlk
        cost = 2*T.log(T.diagonal(R)).sum()+2*enll.sum()+1./sig2_n*(
                (y**2).sum()-(beta**2).sum())+2*(N-self.M)*a
        penelty = kl(mu_w, sig_w)
        cost = (cost+penelty)/N
        dhyper = T.grad(cost, hyper)
        train_input = [X, y, hyper]
        train_input_name = ['X', 'y', 'hyper']
        train_output = [t_alpha, t_Ri, mu_f, cost, dhyper]
        train_output_name = ['alpha', 'Ri', 'mu_f', 'cost', 'dhyper']
        self.train_func = theano.function(train_input, train_output)
        FFs = T.dot(Xs, Omega)+(Theta-T.sum(Z*Omega, 0)[None, :])
        Phis = sig_f*T.sqrt(2./self.M)*T.concatenate((T.cos(FFs), T.sin(FFs)), 1)
        mu_pred = T.dot(Phis, alpha)
        std_pred = (noise*(1+(T.dot(Phis, Ri.T)**2).sum(1)))**0.5
        pred_input = [Xs, hyper, alpha, Ri]
        pred_input_name = ['Xs', 'hyper', 'alpha', 'Ri']
        pred_output = [mu_pred, std_pred]
        pred_output_name = ['mu_pred', 'std_pred']
        self.pred_func = theano.function(pred_input, pred_output)

    def init_model(self):
        a_b_c = npr.randn(3)
        omega = npr.randn(self.D*self.M)
        z = npr.randn(self.D*self.M)
        theta = 2*np.pi*npr.rand(self.M)
        self.hyper = np.concatenate((a_b_c, omega, z, theta))
        self.alpha, self.Ri, self.mu_f, self.cost, _ =\
            self.train_func(self.X, self.y, self.hyper)

    def fit(self, X, y, Xv=None, yv=None, funcs=None, opt=None, vis=None):
        for metric in self.evals.keys():
            self.evals[metric][1] = []
        self.opt = opt
        self.message("-"*50, "\nNormalizing SCFGP training data...")
        self.X_scaler.fit(X)
        self.y_scaler.fit(y)
        self.X = self.X_scaler.forward_transform(X)
        self.y = self.y_scaler.forward_transform(y)
        self.message("done.")
        self.N, self.D = self.X.shape
        if(self.M == -1):
            self.M = int(min(self.N/10., self.N**0.6))
        if(self.R == -1):
            self.R = int(min(self.D/2., self.M**0.6))
        self.generate_ID()
        if(funcs is None):
            self.message("-"*50, "\nCompiling SCFGP theano model...")
            self.build_theano_model()
            self.message("done.")
        else:
            self.train_func, self.pred_func = funcs
        if(Xv is not None and yv is not None):
            self.Xv = self.X_scaler.forward_transform(Xv)
            self.yv = self.y_scaler.forward_transform(yv)
        else:
            plot = False
        train_start_time = time.time()
        self.message("-"*50, "\nInitializing SCFGP hyperparameters...")
        self.init_model()
        self.message("done.")
        if(self.opt is None):
            self.opt = Optimizer("adam", [0.01, 0.8, 0.88], 500, 18, 1e-5, True)
        if(vis is not None):
            vis.model = self
            animate = vis.train_with_plot()
        def train(iter, hyper):
            self.hyper = hyper.copy()
            self.alpha, self.Ri, mu_f, COST, dhyper =\
                self.train_func(self.X, self.y, hyper)
            self.mu_f = self.y_scaler.backward_transform(mu_f)
            self.evals['COST'][1].append(np.double(COST))
            self.evals['TIME(s)'][1].append(time.time()-train_start_time)
            if(Xv is not None and yv is not None):
                self.predict(Xv, yv)
            if(iter == -1):
                return
            if(iter%(self.opt.max_iter//10) == 1):
                self.message("-"*12, "VALIDATION ITERATION", iter, "-"*12)
                self._print_current_evals()
            if(vis is not None):
                animate(iter)
                plt.pause(0.05)
            if(Xv is not None and yv is not None):
                return COST, self.evals['NMSE'][1][-1], dhyper
            return COST, COST, dhyper
        self.opt.run(train, self.hyper)
        self.message("-"*16, "TRAINING RESULT", "-"*16)
        self._print_current_evals()
        self.message("="*60)

    def predict(self, Xs, ys=None):
        mu_f, std_f = self.pred_func(
            self.X_scaler.forward_transform(Xs), self.hyper, self.alpha, self.Ri)
        mu_y = self.y_scaler.backward_transform(mu_f)
        up_bnd_f = self.y_scaler.backward_transform(mu_f+std_f[:, None])
        lw_bnd_f = self.y_scaler.backward_transform(mu_f-std_f[:, None])
        std_y = (up_bnd_f-lw_bnd_f)*0.5
        if(ys is not None):
            self.evals['MAE'][1].append(np.mean(np.abs(mu_y-ys)))
            self.evals['NMAE'][1].append(self.evals['MAE'][1][-1]/np.std(ys))
            self.evals['MSE'][1].append(np.mean((mu_y-ys)**2.))
            self.evals['NMSE'][1].append(self.evals['MSE'][1][-1]/np.var(ys))
            self.evals['MNLP'][1].append(0.5*np.mean(((
                ys-mu_y)/std_y)**2+np.log(2*np.pi*std_y**2)))
            self.evals['SCORE'][1].append(np.exp(
                -self.evals['MNLP'][1][-1])/self.evals['NMSE'][1][-1])
        return mu_y, std_y

    def save(self, path):
        import pickle
        save_vars = ['ID', 'R', 'M', 'X_scaler', 'y_scaler',
            'pred_func', 'hyper', 'alpha', 'Ri', 'evals']
        save_dict = {varn: self.__dict__[varn] for varn in save_vars}
        with open(path, "wb") as save_f:
            pickle.dump(save_dict, save_f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        import pickle
        with open(path, "rb") as load_f:
            load_dict = pickle.load(load_f)
        for varn, var in load_dict.items():
            self.__dict__[varn] = var
        self.NAME = "SCFGP (Rank=%s, Feature Size=%d)"%(str(self.R), self.M)

    def _print_current_evals(self):
        self.message(self.NAME, "   TIME = %.4f"%(self.evals['TIME(s)'][1][-1]))
        self.message(self.NAME, "  SCORE = %.4f"%(self.evals['SCORE'][1][-1]))
        self.message(self.NAME, "   COST = %.4f"%(self.evals['COST'][1][-1]))
        self.message(self.NAME, "    MAE = %.4f"%(self.evals['MAE'][1][-1]))
        self.message(self.NAME, "   NMAE = %.4f"%(self.evals['NMAE'][1][-1]))
        self.message(self.NAME, "    MSE = %.4f"%(self.evals['MSE'][1][-1]))
        self.message(self.NAME, "   NMSE = %.4f"%(self.evals['NMSE'][1][-1]))
        self.message(self.NAME, "   MNLP = %.4f"%(self.evals['MNLP'][1][-1]))




