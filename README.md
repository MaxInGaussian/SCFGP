#SCFGP

SCFGP is a proposed improvement of "Gaussian Processes for Machine Learning" -- a state-of-the-art machine learning technique originated from and popularized by [Carl Edward Rasmussen and Christopher K. I. Williams](http://www.gaussianprocess.org/gpml/). The idea of SCFGP is based on optimization of a small number of sparsely correlated Fourier features, so that the training complexity can be greatly reduced. 

It is implemented in python using Theano and originally designed by Max W. Y. Lam (maxingaussian@gmail.com).

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
model = SCFGP(<rank of frequency matrix>, <size of Fourier features>, fftype={feature type}, msg={print message or not})
model.fit(X_train, y_train, {X_test}, {y_test})
# <>: necessary inputs, {}: optional inputs
```
## Predict Boston Housing Prices
```python
In [1]: X_train.shape, y_train.shape, X_test.shape, y_test.shape
Out[1]: ((400, 13), (400, 1), (106, 13), (106, 1))
```
![BostonHousingMAE](experiments/boston_housing/full_rank_plots/mae.png?raw=true "Boston Housing MAE")
![BostonHousingMSE](experiments/boston_housing/full_rank_plots/mse.png?raw=true "Boston Housing MSE")
![BostonHousingRMSE](experiments/boston_housing/full_rank_plots/rmse.png?raw=true "Boston Housing RMAE")
![BostonHousingNMSE](experiments/boston_housing/full_rank_plots/nmse.png?raw=true "Boston Housing NMSE")
![BostonHousingMNLP](experiments/boston_housing/full_rank_plots/mnlp.png?raw=true "Boston Housing MNLP")
![BostonHousingTime](experiments/boston_housing/full_rank_plots/time.png?raw=true "Boston Housing Time")
## Predict Age of Abalone
```python
In [2]: X_train.shape, y_train.shape, X_test.shape, y_test.shape
Out[2]: ((3133, 10), (3133, 1), (1044, 10), (1044, 1))
```
![AbaloneMAE](experiments/abalone/full_rank_plots/mae.png?raw=true "Abalone MAE")
![AbaloneMSE](experiments/abalone/full_rank_plots/mse.png?raw=true "Abalone MSE")
![AbaloneRMSE](experiments/abalone/full_rank_plots/rmse.png?raw=true "Abalone RMAE")
![AbaloneNMSE](experiments/abalone/full_rank_plots/nmse.png?raw=true "Abalone NMSE")
![AbaloneMNLP](experiments/abalone/full_rank_plots/mnlp.png?raw=true "Abalone MNLP")
![AbaloneTime](experiments/abalone/full_rank_plots/time.png?raw=true "Abalone Time")
## Predict Kinematics of 8-link Robot Arm
```python
In [3]: X_train.shape, y_train.shape, X_test.shape, y_test.shape
Out[3]: ((5000, 8), (5000, 1), (3192, 8), (3192, 1))
```
![Kin8nmMAE](experiments/kin8nm/low_rank_plots/mae.png?raw=true "Kin8nm MAE")
![Kin8nmMSE](experiments/kin8nm/low_rank_plots/mse.png?raw=true "Kin8nm MSE")
![Kin8nmRMSE](experiments/kin8nm/low_rank_plots/rmse.png?raw=true "Kin8nm RMAE")
![Kin8nmNMSE](experiments/kin8nm/low_rank_plots/nmse.png?raw=true "Kin8nm NMSE")
![Kin8nmMNLP](experiments/kin8nm/low_rank_plots/mnlp.png?raw=true "Kin8nm MNLP")
![Kin8nmTime](experiments/kin8nm/low_rank_plots/time.png?raw=true "Kin8nm Time")
<h3 align="center">
"Training time becomes less sensitive to the sample size, sensitive to the number of Fourier feature"
</h3>
## Examine the Efficacy of Training Process on Real-Time
```python
model.fit(X_train, y_train, {X_test}, {y_test}, plot_training=True)
# <>: necessary inputs, {}: optional inputs
```
# Training on High-dimensional Data (Boston Housing Prices)
![BostonHousingPlotTraining](experiments/boston_housing/plot_training.gif?raw=true "Boston Housing Plot Training")
#License
Copyright (c) 2016, Max W. Y. Lam
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.