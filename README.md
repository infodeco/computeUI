# computeUI

This repository provides code to compute the information decomposition defined in [*Quantifying Unique Information*](http://dx.doi.org/10.3390/e16042161).
The code implements the admUI algorithm presented in [*Computing the Unique Information*](https://arxiv.org/abs/1709.07487).

The repository contains two implementation:
- an implementation in Python that works together with the Python package [dit](https://github.com/dit/).
- an implementation in Matlab. 

## Python

The Python implementation was tested with Python 3.4.2.  It can be used standalone, plus there are wrapper functions that allow to work with probability distributions generated using [dit](https://github.com/dit/)

### An easy example: The AND distribution S = AND(X,Y)

d = dit.Distribution(['000', '001', '010', '111'], [1. / 4] * 4) 

d.set_rv_names('SXY')

#### admUI algorithm 
Q = computeQUI(distSXY = d, DEBUG = True)

UIX = dit.shannon.conditional_entropy(Q, 'S', 'Y') + dit.shannon.conditional_entropy(Q, 'X', 'Y') - dit.shannon.conditional_entropy(Q, 'SX', 'Y')

#### Frank-Wolfe algorithm in the dit package
pid = algorithms.pid_broja(d, ['X', 'Y'], 'S') 

### Files

- test_admUI.py: wrapping function comparing the admUI algorithm with the Frank-Wolfe implementation in the [dit](https://github.com/dit/) package for some easy examples.
- test_admUI_cvxUI_dataPs.py, test_dit_dataPs.py: wrapping functions for generating datapoints to compare the admUI with an implementation [cvxopt_solve](https://github.com/Abzinger/BROJA-Bivariate-Partial_Information_Decomposition/blob/master/Python/cvxopt_solve.py) using the python interior-point solver [CVXOPT](http://cvxopt.org/) and the Frank-Wolfe implementation in the [dit](https://github.com/dit/) package.

This function performs the main computation:
- admUI.py: the alternating divergence minimization algorithm for computing the unique information (admUI) proposed in Banerjee, et al., 2017. 

## Matlab

The Matlab version was tested with Matlab 2017a, but should also work with older versions of Matlab.  However, Matlab 2017a is needed to make use of the MEX-feature (see below).

### An easy example: The AND distribution

ns = 2; ny = 2; nz = 2; 

P  = [1 1 1 0 0 0 0 1]; P = P/sum(P);

Pzys = reshape(P,nz,ny,ns); Psy = squeeze(sum(Pzys,1))'; Psz = squeeze(sum(Pzys,2))';

[UI,Q] = admUIg(Psy, Psz)

### Files

- test_dataPs.m, test_dataPy.m, test_dataPz.m: wrapping functions that compares the admUI algorithm with fmincon from the Matlab Optimization Toolbox (algorithm: interior-point) when including the gradient and Hessian, only the gradient, and when including none.
- copytest.m: wrapping function for comparing the admUI algorithm and fmincon for the Copy example, vis-a-vis two different stopping criteria: a heuristic and a rigorous one.
- test_admUI_accn_eps.m: wrapping function comparing the convergence of an accelerated version of the admUI algorithm with the original, for a given accuracy (currently tested only for binary-valued S, Y, Z).

These functions perform the main computations:
- admUIg.m: the alternating divergence minimization algorithm for computing the unique information (admUI) proposed in Banerjee, et al., 2017. Supports two different stopping criterias: a heuristic and a rigorous one.
- fn_UI_fmincon - fmincon implementation of the optimization problem with options for including the gradient and Hessian.

### MEX

The Matlab *MEX* (Matlab executable) feature allows to precompile code, which greatly increases the speed. The MEX files are generated and tested in Matlab 2017a. 

To generate the MEX function admUI_mex from the Matlab function admUI, do the following:
- Open the included MATLAB Coder project file admUI.prj in computeUI/matlab/MEX/ to specify the various code generation parameters like the sizes and data types of inputs.
- Select the script testAdmUI.m that exercises the target function admUI.
- Select Build Type: MEX to generate the MEX function admUI_mex which is used in the testcases in test_dataPs, test_dataPy, test_dataPz.

Repeat the same process to generate the MEX function admUIg_mex from the Matlab function admUIg which supports a more rigorous stopping criteria and is used in the copytest wrapping function.



