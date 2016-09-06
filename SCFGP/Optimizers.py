################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import numpy as np

class Optimizer(object):
    
    " Uncontrainted Gradient-Based Optimizer (Objective Function Minimization) "

    available_optimizers = ["sgd", "rmsprop", "adam", "smorms3"]
    
    max_iter, max_cvrg_iter, cvrg_tol, ln_params = 0, 0, 0, []
    stop_by_train_error = False
    
    def __init__(self, opt, ln_params, max_iter, max_cvrg_iter, cvrg_tol,
        stop_by_train_error=False):
        assert opt.lower() in self.available_optimizers, "Invalid Optimizer!"
        self.opt = opt.lower()
        self.max_iter = max_iter
        self.max_cvrg_iter = max_cvrg_iter
        self.cvrg_tol = cvrg_tol
        self.ln_params = ln_params
        self.stop_by_train_error = stop_by_train_error
    
    def run(self, train, x0):
        if(self.opt == "sgd"):
            self.sgd(train, x0)
        elif(self.opt == "rmsprop"):
            self.rmsprop(train, x0)
        elif(self.opt == "adam"):
            self.adam(train, x0)
        elif(self.opt == "smorms3"):
            self.smorms3(train, x0)
    
    ### Stochastic Gradient Descent with Momentum
    def sgd(self, train, x0):
        if(self.ln_params is None):
            self.ln_params = [0.05, 0.9]
        x = x0.copy()
        step_size, m = self.ln_params
        last_e, f, min_e, e = None, None, np.Infinity, None
        min_f, argmin_x, cvrg_iter = np.Infinity, None, 0
        v = np.zeros_like(x)
        adjust_step = True
        for i in range(self.max_iter):
            f, e, g = train(i+1, x)
            if(not self.stop_by_train_error):
                e = f
            if(adjust_step and last_e is not None and e > 0 and last_e > 0):
                step_size *= max(0.8, min(1.2, (
                    1+last_e-e-10*self.cvrg_tol)**(50/(i+1))))
            last_e = e
            if(e < min_e):
                if(min_e-e < self.cvrg_tol):
                    cvrg_iter += 1
                else:
                    cvrg_iter = 0
                min_e = e
                argmin_x = x.copy()
            else:
                cvrg_iter += 1
            if(cvrg_iter > self.max_cvrg_iter*0.5):
                adjust_step = False
            if(f < min_f):
                min_f = f
            if(cvrg_iter > self.max_cvrg_iter):
                break
            elif(cvrg_iter > self.max_cvrg_iter*0.5):
                randp = np.random.rand()
                x = randp*x+(1-randp)*argmin_x
            v = m*v-(1.0-m)*g
            x += step_size*v
        train(-1, argmin_x)
    
    ### RMSprop (http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)
    def rmsprop(self, train, x0):
        if(self.ln_params is None):
            self.ln_params = [0.05, 0.9]
        x = x0.copy()
        step_size, gamma = self.ln_params
        last_e, f, min_e, e = None, None, np.Infinity, None
        min_f, argmin_x, cvrg_iter = np.Infinity, None, 0
        avg_sq_grad = np.zeros_like(x)
        adjust_step = True
        for i in range(self.max_iter):
            f, e, g = train(i+1, x)
            if(not self.stop_by_train_error):
                e = f
            if(adjust_step and last_e is not None and e > 0 and last_e > 0):
                step_size *= max(0.8, min(1.2, (
                    1+last_e-e-10*self.cvrg_tol)**(50/(i+1))))
            last_e = e
            if(e < min_e):
                if(min_e-e < self.cvrg_tol):
                    cvrg_iter += 1
                else:
                    cvrg_iter = 0
                min_e = e
                argmin_x = x.copy()
            else:
                cvrg_iter += 1
            if(cvrg_iter > self.max_cvrg_iter*0.5):
                adjust_step = False
            if(f < min_f):
                min_f = f
            if(cvrg_iter > self.max_cvrg_iter):
                break
            elif(cvrg_iter > self.max_cvrg_iter*0.5):
                randp = np.random.rand()
                x = randp*x+(1-randp)*argmin_x
            avg_sq_grad = avg_sq_grad*gamma+g**2*(1-gamma)
            x -= step_size*g/(np.sqrt(avg_sq_grad)+1e-10)
        train(-1, argmin_x)
    
    ### Adam (http://arxiv.org/pdf/1412.6980.pdf)
    def adam(self, train, x0):
        if(self.ln_params is None):
            self.ln_params = [0.05, 0.9, 0.999]
        x = x0.copy()
        step_size, b1, b2 = self.ln_params
        last_e, f, min_e, e = None, None, np.Infinity, None
        min_f, argmin_x, cvrg_iter = np.Infinity, None, 0
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        adjust_step = True
        for i in range(self.max_iter):
            f, e, g = train(i+1, x)
            if(not self.stop_by_train_error):
                e = f
            if(adjust_step and last_e is not None and e > 0 and last_e > 0):
                step_size *= max(0.8, min(1.2, (
                    1+last_e-e-10*self.cvrg_tol)**(50/(i+1))))
            last_e = e
            if(e < min_e):
                if(min_e-e < self.cvrg_tol):
                    cvrg_iter += 1
                else:
                    cvrg_iter = 0
                min_e = e
                argmin_x = x.copy()
            else:
                cvrg_iter += 1
            if(cvrg_iter > self.max_cvrg_iter*0.5):
                adjust_step = False
            if(f < min_f):
                min_f = f
            if(cvrg_iter > self.max_cvrg_iter):
                break
            elif(cvrg_iter > self.max_cvrg_iter*0.5):
                randp = np.random.rand()
                x = randp*x+(1-randp)*argmin_x
            m = (1-b1)*g+b1*m
            v = (1-b2)*(g**2)+b2*v
            mhat = m/(1-b1**(i+1))
            vhat = v/(1-b2**(i+1))
            x -= step_size*mhat/(np.sqrt(vhat)+1e-10)
        train(-1, argmin_x)
    
    ### SMORMS3 (http://sifter.org/~simon/journal/20150420.html)
    def smorms3(self, train, x0):
        if(self.ln_params is None):
            self.ln_params = [0.05]
        x = x0.copy()
        step_size = self.ln_params[0]
        f, last_e, min_e, e = None, None, np.Infinity, None
        min_f, argmin_x, cvrg_iter = np.Infinity, None, 0
        x = x0.copy()
        m = np.ones_like(x0)
        g1 = np.ones_like(x0)
        g2 = np.ones_like(x0)
        adjust_step = True
        for i in range(self.max_iter):
            f, e, g = train(i+1, x)
            if(not self.stop_by_train_error):
                e = f
            if(adjust_step and last_e is not None and e > 0 and last_e > 0):
                step_size *= max(0.8, min(1.2, (
                    1+last_e-e-10*self.cvrg_tol)**(50/(i+1))))
            last_e = e
            if(e < min_e):
                if(min_e-e < self.cvrg_tol):
                    cvrg_iter += 1
                else:
                    cvrg_iter = 0
                min_e = e
                argmin_x = x.copy()
            else:
                cvrg_iter += 1
            if(cvrg_iter > self.max_cvrg_iter*0.5):
                adjust_step = False
            if(f < min_f):
                min_f = f
            if(cvrg_iter > self.max_cvrg_iter):
                break
            elif(cvrg_iter > self.max_cvrg_iter*0.5):
                randp = np.random.rand()
                x = randp*x+(1-randp)*argmin_x
            r = 1/(m+1)
            g1 = (1-r)*g1+r*g
            g2 = (1-r)*g2+r*g**2
            rate = g1*g1/(g2+1e-16)
            m = m*(1-rate) + 1
            alpha = np.minimum(step_size, rate)/(np.sqrt(g2)+1e-10)
            x -= g*alpha
        train(-1, argmin_x)
