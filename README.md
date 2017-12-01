# computeUI

This repository provides code to compute the information decomposition defined in [*Quantifying Unique Information*](http://dx.doi.org/10.3390/e16042161).
The code implements the admUI algorithm presented in [*Computing the Unique Information*](https://arxiv.org/abs/1709.07487).

The repository contains two implementation:
- an implementation in Python that works together with the Python package [dit](https://github.com/dit/)
- an implementation in Matlab

## Matlab

The Matlab version was tested with Matlab 2017a, but should also work with older versions of Matlab.  However, Matlab 2017a is needed to make use of the MEX-feature (see below).

### Easy examples

### Files

- main_test.m: this is the wrapping function that compares the different methods.
- copytest.m: wrapping function for comparing the admUI algorithm and fmincon for the Copy example

These functions perform the main computations:
- admUIg.m: the alternating divergence minimization algorithm for computing the unique information (admUI) proposed in Banerjee, et al., 2017. Supports two different stopping criterias.
- fn_UI_fmincon - fmincon implementation with three options: blackbox, including only the gradient, including gradient and Hessian

### MEX

The Matlab *MEX* (Matlab executable) feature allows to precompile code, which greatly increases the speed.  To generate the mex-file, do the following:

...

## Python

The Python implementation was tested with Python 3.4.2.  It can be used standalone, plus there are wrapper functions that allow to work with probability distributions generated using [dit](https://github.com/dit/)

### Easy examples

Python with dit

Python without dit

### Files

