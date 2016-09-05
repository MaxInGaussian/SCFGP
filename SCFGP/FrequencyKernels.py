################################################################################
#  SCFGP: Sparsely Correlated Fourier Features Based Gaussian Process
#  Github: https://github.com/MaxInGaussian/SCFGP
#  Author: Max W. Y. Lam (maxingaussian@gmail.com)
################################################################################

import theano.tensor as T

class FrequencyKernel(object):
    
    " Frequency Kernel For Contruction of Sparsely Correlated Fourier Features "

    available_kerns = ["dot", "wht", "lin", "rbf", "per", "exp"]
    
    kern, freq_rank = None, -1
    params_size, params_vec = None, None
    
    def __init__(self, kern, freq_rank):
        assert kern.lower() in self.available_kerns, "Invalid Frequency Kernel!"
        self.kern = kern.lower()
        self.freq_rank = freq_rank
        if(self.kern == "dot"):
            self.params_size = 1
        elif(self.kern == "wht"):
            self.params_size = self.freq_rank
        elif(self.kern == "lin"):
            self.params_size = 2*self.freq_rank
        elif(self.kern == "rbf"):
            self.params_size = self.freq_rank
        elif(self.kern == "per"):
            self.params_size = self.freq_rank
        elif(self.kern == "exp"):
            self.params_size = self.freq_rank
    
    def set_params(self, params_vec):
        self.params_vec = params_vec
    
    def fit(self, X, Y):
        if(self.kern == "dot"):
            return self.simple_dot_kernel(X, Y)
        elif(self.kern == "wht"):
            return self.white_noise_kernel(X, Y)
        elif(self.kern == "lin"):
            return self.linear_kernel(X, Y)
        elif(self.kern == "rbf"):
            return self.radial_basis_function_kernel(X, Y)
        elif(self.kern == "per"):
            return self.periodic_kernel(X, Y)
        elif(self.kern == "exp"):
            return self.exponential_kernel(X, Y)
    
    ### Simple Dot Kernel
    def simple_dot_kernel(self, X, Y):
        sigma = self.params_vec[0]
        return sigma*T.dot(X, Y.T)
    
    ### White Noise Kernel
    def white_noise_kernel(self, X, Y):
        rank_scale = self.params_vec[None, None, :]
        X_3d, Y_3d = X[:, None, :], Y[None, :, :]
        return T.sum(rank_scale*X_3d*Y_3d, 2)

    ### Linear Kernel
    def linear_kernel(self, X, Y):
        coeff_X = self.params_vec[:self.freq_rank][None, None, :]
        coeff_Y = self.params_vec[self.freq_rank:][None, None, :]
        X_3d, Y_3d = X[:, None, :], Y[None, :, :]
        kern_prod = (X_3d-coeff_X)*(Y_3d-coeff_Y)
        return T.sum(kern_prod*X_3d*Y_3d, 2)

    ### Radial Basis Function Kernel
    def radial_basis_function_kernel(self, X, Y):
        length_scale = self.params_vec[None, None, :]
        X_3d, Y_3d = X[:, None, :], Y[None, :, :]
        kern_prod = T.exp(-1./2*((X_3d-Y_3d)/length_scale)**2.)
        return T.sum(kern_prod*X_3d*Y_3d, 2)

    ### Periodic Kernel
    def periodic_kernel(self, X, Y):
        length_scale = self.params_vec[None, None, :]
        X_3d, Y_3d = X[:, None, :], Y[None, :, :]
        kern_prod = T.exp(-2*(T.sin(T.sqrt((X_3d-Y_3d)**2.))/length_scale)**2.)
        return T.sum(kern_prod*X_3d*Y_3d, 2)

    ### Exponential Kernel
    def exponential_kernel(self, X, Y):
        length_scale = self.params_vec[None, None, :]
        X_3d, Y_3d = X[:, None, :], Y[None, :, :]
        kern_prod = T.exp(-1./2*T.sqrt(((X_3d-Y_3d)/length_scale)**2.))
        return T.sum(kern_prod*X_3d*Y_3d, 2)
    
    













