#SCFGP

SCFGP is a proposed variant of [Gaussian process regression](http://www.gaussianprocess.org/gpml/) that bases on sparsely correlated Fourier features. 
It is implemented in python using Theano. It was originally
designed and is now managed by Max W. Y. Lam (maxingaussian@gmail.com).

###Highlights of SCFGP

- SCFGP optimizes
the Fourier features so as to "learn" a tailmade covariance matrix from the data. 
This removes the necessity of deciding which kernel function to use in different problems.

- SCFGP implements a variety of formulation to transform the optimized Fourier features to covariance matrix, including the typical sin-cos concatenation introduced by [Miguel](http://www.jmlr.org/papers/v11/lazaro-gredilla10a.html), and the generalized approach described by [Yarin](http://jmlr.org/proceedings/papers/v37/galb15.html).

- SCFGP uses low-rank frequency matrix for sparse approximation of Fourier features. It is 
intended to show that low-rank frequency matrix is able to lower the computational 
burden in each step of optimization, and also render faster convergence and a stabler result.

- Compared with other 
state-of-the-art regressors, SCFGP usually gives the most accurate prediction on the benchmark datasets of regression.

# Installation
   
### SCFGP

To install SCFGP, use pip:

    $ pip install SCFGP

Or clone this repo:

    $ git clone https://github.com/MaxInGaussian/SCFGP.git
    $ python setup.py install

## Dependencies
### Theano
    Theano is used due to its nice and simple coding style to represent tedious formulas of SCFGP, and
    the capability of computing automatic differentiation efficiently.
    
To install Theano, see this page:

   http://deeplearning.net/software/theano/install.html

### scikit-learn (only used in the experiments)
    
To install scikit-learn, see this page:

   https://github.com/scikit-learn/scikit-learn

# Use SCFGP for Regression
```python
from SCFGP import *
model = SCFGP(<rank of frequency matrix>, <size of Fourier features>, fftype=<feature type>)
model.fit(X_train, y_train, X_test, y_test)
```
## Predict Boston Housing Prices
![BostonHousingMSE](experiments/boston_housing_mse.png?raw=true "Boston Housing MSE")
![BostonHousingMNLP](experiments/boston_housing_mnlp.png?raw=true "Boston Housing MNLP")
![BostonHousingTIME](experiments/boston_housing_time.png?raw=true "Boston Housing Time")
#Use SCFGP for Supervised Dimensionality Reduction
```python
from SCFGP import SCFGP
model = SCFGP(<rank of frequency matrix>, <size of Fourier features>, fftype=<feature type>)
model.fit(X_train, y_train, X_test, y_test)
```
## Visualize MNIST
### 1) Feature Type = Fourier (sine & cosine)
![MNIST-F-10](experiments/mnist/visualize_mnist_f_10.png?raw=true "MNIST F 10")
![MNIST-F-30](experiments/mnist/visualize_mnist_f_30.png?raw=true "MNIST F 10")
![MNIST-F-100](experiments/mnist/visualize_mnist_f_100.png?raw=true "MNIST F 10")
### 2) Feature Type = Fourier (sine & cosine) + Inducing Frequencies
![MNIST-ZF-10](experiments/mnist/visualize_mnist_zf_10.png?raw=true "MNIST ZF 10")
![MNIST-ZF-30](experiments/mnist/visualize_mnist_zf_30.png?raw=true "MNIST ZF 30")
![MNIST-ZF-100](experiments/mnist/visualize_mnist_zf_100.png?raw=true "MNIST ZF 100")
### 3) Feature Type = Cosine with Adjustable Phases (only cosine)
![MNIST-PH-10](experiments/mnist/visualize_mnist_ph_10.png?raw=true "MNIST PH 10")
![MNIST-PH-30](experiments/mnist/visualize_mnist_ph_30.png?raw=true "MNIST PH 30")
![MNIST-PH-100](experiments/mnist/visualize_mnist_ph_100.png?raw=true "MNIST PH 100")
### 4) Feature Type = Cosine with Adjustable Phases (only cosine) + Inducing Frequencies
![MNIST-ZPH-10](experiments/mnist/visualize_mnist_zph_10.png?raw=true "MNIST ZPH 10")
![MNIST-ZPH-30](experiments/mnist/visualize_mnist_zph_30.png?raw=true "MNIST ZPH 30")
![MNIST-ZPH-100](experiments/mnist/visualize_mnist_zph_100.png?raw=true "MNIST ZPH 100")
<h3 align="center">
"The similariy of high-dimensional features over different classes are preserved!"
</h3>

#License
Copyright (c) 2016, Max W. Y. Lam
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.