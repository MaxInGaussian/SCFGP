# SCFGP

SCFGP is a proposed improvement of [Sparse Spectrum Gaussian Process](http://quinonero.net/Publications/lazaro-gredilla10a.pdf) (SPGP), which is a new branch of method to speed up Gaussian process model taking advantage of [Fourier features](https://papers.nips.cc/paper/3182-random-features-for-large-scale-kernel-machines.pdf). Recall that using Gaussian processes for machine learning is a state-of-the-art technique that is originated from and popularized by [Carl Edward Rasmussen and Christopher K. I. Williams](http://www.gaussianprocess.org/gpml/).

Based on minimization of the marginal likelihood, SCFGP selects a set of vectors to obtain a [Gramian matrix](https://en.wikipedia.org/wiki/Gramian_matrix), which is treated as the frequency matrix for later computation of Fourier features. This procedure indeed can be viewed as constructing sparsely correlated Fourier features. 

Note that the Fourier features are identically and independently distributed in SPGP, therefore the size of optimization parameters is proportional to the number of Fourier features times the number of dimension. This is undoubtedly an unfavorable property, since the model is likely to stick in local minima and becomes very unstable when dealing with very high dimensional data, such as images, speech signals, text, etc.

The formulation of SCFGP is briefly described in this sheet: (Derivation will be included in the future)

![SCFGP Formulas](SCFGP_formulas.png?raw=true "SCFGP Formulas")

SCFGP is implemented in python using Theano by Max W. Y. Lam (maxingaussian@gmail.com).


# Installation
   
### SCFGP

To install SCFGP, use pip:

    $ pip install SCFGP

Or clone this repo:

    $ git clone https://github.com/MaxInGaussian/SCFGP.git
    $ python setup.py install

## Dependencies
### Theano
    Theano is used due to its nice and simple syntax to set up the tedious formulas in SCFGP, and
    its capability of computing automatic differentiation.
    
To install Theano, see this page:

   http://deeplearning.net/software/theano/install.html

### scikit-learn (only used in the experiments)
    
To install scikit-learn, see this page:

   https://github.com/scikit-learn/scikit-learn
# Try SCFGP with Only 3 Lines of Code
```python
from SCFGP import *
# <>: necessary inputs, {}: optional inputs
model = SCFGP(rank=<rank_of_frequency_matrix>,
              feature_size=<number_of_Fourier_features>,
              fftype={feature_type},
              msg={print_message_or_not})
model.fit(X_train, y_train, {X_test}, {y_test})
predict_mean, predict_std = model.predict(X_test, {y_test})
```

# License
Copyright (c) 2016, Max W. Y. Lam
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
