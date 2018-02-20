# computeUI

This repository provides code to compute the information decomposition defined in [*Quantifying Unique Information*](http://dx.doi.org/10.3390/e16042161).
The code implements the admUI algorithm proposed by Banerjee, et al. in [*Computing the Unique Information*](https://arxiv.org/abs/1709.07487).

The repository contains two implementation:
- an implementation in Python that works together with the Python package [dit](https://github.com/dit/).
- an implementation in Matlab. 

## Python

The Python implementation requires to have [`numpy`](http://www.numpy.org) installed.  It can be used standalone, plus there are wrapper functions that allow to work with probability distributions generated using [`dit`](https://github.com/dit/) (version 1.0.0.dev6). The standalone version was tested with Python versions 3.4.2 and 2.7.9.

### Installation and Files

To install, make sure sure that Python finds the following files (e.g. by copying the file into the python search path or
by amending `sys.path`):
- `admUI_numpy.py`: This file contains the function `computeQUI_numpy` that implements the alternating divergence minimization algorithm for computing the unique information (admUI).
- `admUI.py`: This file contains the wrapper function `computeQUI` that allows to work with probability distributions generated using [`dit`](https://github.com/dit/).

The following files contain tests and examples:
- `test_admUI.py`: testcase comparing the admUI algorithm with the Frank-Wolfe implementation in the [`dit`](https://github.com/dit/) package for some small examples.  On small examples, both algorithms perform well, and the `dit` implementation often beats `admUI`.  The comparison demonstrates that `admUI` achieves the specified error rate (unless one of the loops reaches the maximum number of iterations).
- `test_admUI_cvxUI_dataPs.py`, `test_dit_dataPs.py`: testcases for generating datapoints to compare the admUI with an implementation [cvxopt_solve](https://github.com/Abzinger/BROJA-Bivariate-Partial_Information_Decomposition/blob/master/Python/cvxopt_solve.py) using the python interior-point solver [CVXOPT](http://cvxopt.org/) and the Frank-Wolfe implementation in the [`dit`](https://github.com/dit/) package.

### Example: The AND distribution S = AND(X,Y) using `dit`

The following code prepares the joint distribution of the AND example, using `dit`:

```python
import admUI
import dit
d = dit.Distribution(['000', '001', '010', '111'], [1. / 4] * 4) 
d.set_rv_names('SXY')
```

The following code computes the unique information of X using the admUI algorithm:

```python
Q = admUI.computeQUI(distSXY = d)
print(Q)
dit.shannon.conditional_entropy(Q, 'S', 'Y') + dit.shannon.conditional_entropy(Q, 'X', 'Y') - dit.shannon.conditional_entropy(Q, 'SX', 'Y')
```

Alternatively, the `dit` package can be used to compute the same quantity:

```python
dit.algorithms.pid_broja(d, ['X', 'Y'], 'S') 
```

### Example: The AND distribution with `numpy`

The function `computeQUI_numpy` in `admUI_numpy.py` expects three inputs: the conditional distributions of X given S and Y given S and the marginal distribution of S.  The inputs have to be `numpy` arrays.

```python
import admUI_numpy
import numpy
PXgS = numpy.array([[ 2./3,  0.],
                    [ 1./3,  1.]])
PYgS = PXgS
PS = numpy.array([[ 0.75], [ 0.25]])
Q = admUI_numpy.computeQUI_numpy(PXgS, PYgS, PS)
```

The output is a threedimensional `numpy` array of the joint distribution.

## Matlab

The Matlab version was tested with Matlab 2017a, but should also work with older versions of Matlab.  However, Matlab 2017a is needed to make use of the MEX-feature (see below).

### An easy example: The AND distribution

```matlab
ns = 2; ny = 2; nz = 2; 
P  = [1 1 1 0 0 0 0 1]; P = P/sum(P);
Pzys = reshape(P,nz,ny,ns); Psy = squeeze(sum(Pzys,1))'; Psz = squeeze(sum(Pzys,2))';
[UI, Q] = admUIg(Psy, Psz)
```

### Files

These functions perform the main computations:
- `admUIg.m`: the alternating divergence minimization algorithm for computing the unique information (admUI).  Supports two different stopping criterias: a heuristic and a rigorous one.
- `fn_UI_fmincon`: `fmincon` implementation of the optimization problem with options for including the gradient and Hessian.

- `test_dataPs.m`, `test_dataPy.m`, `test_dataPz.m`: testcases comparing the admUI algorithm with `fmincon` from the Matlab Optimization Toolbox (algorithm: interior-point) with options for including the gradient and Hessian.
- `copytest.m`: testcase comparing the admUI algorithm and `fmincon` for the Copy example, vis-a-vis two different stopping criteria: a heuristic and a rigorous one.
- `test_admUI_accn_eps.m`: testcase comparing the convergence of an accelerated version of the `admUI` algorithm with the original, for a given accuracy (currently tested only for binary-valued S, Y, Z).

### MEX

The Matlab *MEX* (Matlab executable) feature allows to precompile code, which greatly increases the speed. The MEX files are generated and tested in Matlab 2017a.

To generate the MEX function `admUI_mex` from the Matlab function `admUI`, do the following:
- Open the included MATLAB Coder project file `admUI.prj` in `computeUI/matlab/MEX/` to specify the various code generation parameters like the sizes and data types of inputs.
- Select the script `testAdmUI.m` that exercises the target function admUI.
- Select Build Type: MEX to generate the MEX function `admUI_mex` which is used in the testcases in `test_dataPs`, `test_dataPy`, `test_dataPz`.

Repeat the same process to generate the MEX function `admUIg_mex` from the Matlab function `admUIg` which supports a more rigorous stopping criteria and is used in the copytest wrapping function.
