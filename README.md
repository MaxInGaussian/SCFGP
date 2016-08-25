#OffGPR

    OffGPR is a proposed variant of Gaussian process regression (GPR) that based on
    optimization of Fourier features. 
    
The lightlight of this model is the use of 
low-rank frequency matrix for sparse approximation of Fourier features. It is 
intended to show that low-rank frequency matrix is able to lower the computational 
burden in each step of optimization, and also render faster convergence and a stabler result.

###Key Features of OffGPR

- It optimizes
the Fourier features so as to "learn" a tailmade covariance matrix from the data. 
This removes the necessity of deciding which kernel function to use in different problems.

- It gives the 

- It relax the 
and also speeds up the original regression ([Rasmussen & Williams]
(http://www.gaussianprocess.org/gpml/)).

This library, OffGPR was implemented in python using Theano. It was originally
created and is now managed by Max W. Y. Lam. Experiments show that it often
gives the most accurate prediction on benchmark datasets of regression over 
state-of-the-art regressors.

Current Version: 0.0.1

Author: Max W. Y. Lam (maxingaussian@gmail.com)

#Dependencies

### Theano:
    Theano is used due to its nice and simple coding style to represent tedious formulas of OffGPR, and
    the capability of computing automatic differentiation efficiently.
    
To install Theano, see this page:

   http://deeplearning.net/software/theano/install.html

For the documentation, see the project website of Theano:

   http://deeplearning.net/software/theano/

# Installation

To install OffGPR, use pip:

    $ pip install OffGPR

Or clone this repo:

    $ git clone https://github.com/MaxInGaussian/OffGPR.git
    $ python setup.py install

#License
Copyright (c) 2016, Max W. Y. Lam
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.