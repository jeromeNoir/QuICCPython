import numpy as np
from numpy.polynomial import chebyshev as cheb
import sys
sys.path.append('/home/nicolol/workspace/QuICC/Scripts/Python/pybind11/')

import QuICC as quicc_pybind


def cheb_eval(nr, a, b, r):
    # INPUT:
    # nr: int, number of radial functions
    # a: double, stretch factor for the mapping to shells
    # b: double, shift factor for the mapping to shells
    # r: np.array or list, list of collocation points
    
    # evaluates the projection matrix for the chebyshev basis
    xx = (np.array(r)-b)/a
    
    coeffs = np.eye(nr)*2
    coeffs[0,0]=1. # set 1 on the first diag entry because of DCT
    # proptierties
    
    # return the dense matrix, because it is needed for integration
    # (evaluation of antiderivative) purposes

    mat = np.mat(cheb.chebval(xx, coeffs).transpose())
    return mat

def fourier_eval(nr, r):
    
    # evaluates the projection matrix for the chebyshev basis
    xx = np.array(r);
    v = np.zeros((nr,1),dtype=complex)
    nr2 = int(nr/2)

    for k in range(1,nr2+1):
        v[k] = np.exp(k*xx*1j)
    for k in range(1,nr2+1):
        v[nr-k] = np.exp(-(k)*xx*1j)
    v[0] = 0.5;


    return np.mat(v.transpose())

def fourier_eval_shift(nr, r):

    # evaluates the projection matrix for the chebyshev basis
    xx = np.array(r);
    v = np.zeros((nr,1),dtype=complex)
    nr2 = int(nr/2)

    for k in range(-nr2,nr2+1):
        v[k] = np.exp(k*xx*1j)
    
    v[0] = 0.5;

    return np.mat(v.transpose())

def kron(a, b):
    # INPUT:
    # a: column vector
    # b: row vector
    # OUTPUT: np.matrix, rank 1 modification matrix from a and b
    
    a = np.reshape(a, (-1, 1))
    b = np.reshape(b, (-1, 1))

    return np.kron(a, b.T)


def plm(maxl, m, x):

    if maxl < m or m < 0 or maxl < 0:
        raise RuntimeError('Problems between l and m')
    
    else:

        # Compute the normalized associated legendre polynomial projection matrix
        mat = np.zeros((maxl-m+1,len(x))).T

        # call the c++ function
        quicc_pybind.plm(mat, m, x)

        #SingletonPlm.get_dict(SingletonPlm, x)[m] = mat

        #return SingletonPlm.get_dict(SingletonPlm, x)[m][:, l-m]
        return mat
