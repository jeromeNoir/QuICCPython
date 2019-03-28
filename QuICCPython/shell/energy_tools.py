import numpy as np
from scipy.fftpack import dct, idct
from numpy.polynomial.chebyshev import chebder

def ortho_pol_q(nr, a, b, x1,  x2):
    # precondition: assume that the weight of the spectra is already in DCT format

    # make copies
    x_1 = np.array(x1)
    x_2 = np.array(x2)
    
    # pad the data
    x1 = np.zeros(int(3/2*nr))
    x2 = np.zeros(int(3/2*nr))
    x1[:len(x_1)] = x_1
    x2[:len(x_2)] = x_2
  
    # bring the functions to physical space
    y1 = idct(x1, type = 2)
    y2 = idct(x2, type = 2)
    
    # multiply in physical
    y3 = y1*y2
    
    # bring back to spectral space
    c = dct(y3, type=2)/(2*len(y3))
    c = c[:nr]
    
    # represent the cheb indices in the mathematically correct way
    c[1:] *= 2.
    
    # integrate, prepare index
    idx = np.arange(len(c))
    # and compute integration weight for -1 to +1
    w = (1+(-1)**idx)/(1-idx**2)
    w[1]=0

    
    return np.sum(w*c)/4.

def ortho_pol_s(nr, a, b, x1,  x2):
    # precondition: assume that the weight of the spectra is already in DCT format

    # take derivative
    # first bring back to correct form
    x_1a = np.array(x1)
    x_1a[1:] *= 2
    x_2a = np.array(x2)
    x_2a[1:] *= 2
    dx_1 = chebder(x_1a)
    dx_2 = chebder(x_2a)
       
    # make copies
    x_1b = np.array(x1)
    x_2b = np.array(x2)
    
    # pad the data
    x1a = np.zeros(int(3/2*nr))
    x1b = np.zeros(int(3/2*nr))
    x2a = np.zeros(int(3/2*nr))
    x2b = np.zeros(int(3/2*nr))
    
    x1a[:len(dx_1)] = dx_1
    x1b[:len(x_1b)] = x_1b
    x2a[:len(dx_2)] = dx_2
    x2b[:len(x_2b)] = x_2b
    
    # create the spectral function that describes  T_1=x
    r1 = np.zeros(int(3/2*nr))
    r1[1] = 1.
    
    # prepare the chebs for DCT
    x1a[1:] *= .5
    #x1b[1:] *= .5
    x2a[1:] *= .5
    #x2b[1:] *= .5
    r1[1:]*=.5
    
    # bring the functions to physical space
    y1a = idct(x1a, type = 2)
    y1b = idct(x1b, type = 2)
    y2a = idct(x2a, type = 2)
    y2b = idct(x2b, type = 2)
    x = idct(r1, type = 2)
    
    # multiply in physical
    y3 = (y1a*(x+b/a)+y1b)*(y2a*(x+b/a)+y2b)
    
    # bring back to spectral space
    c = dct(y3, type=2)/(2*len(y3))
    c = c[:nr]
    
    # represent the cheb indices in the mathematically correct way
    c[1:] *= 2.
    
    # integrate, prepare index
    idx = np.arange(len(c))

    # and compute integration weight for -1 to +1
    w = (1+(-1)**idx)/(1-idx**2)
    w[1]=0

    return np.sum(w*c)/4

def ortho_tor(nr, a, b, x1,  x2):
    # precondition: assume that the weight of the spectra is already in DCT format

    # make copies
    x_1 = np.array(x1)
    x_2 = np.array(x2)
    
    # pad the data
    x1 = np.zeros(int(3/2*nr))
    x2 = np.zeros(int(3/2*nr))
    x1[:len(x_1)] = x_1
    x2[:len(x_2)] = x_2
  
    # create the spectral function that describes  T_1=x
    r1 = np.zeros(int(3/2*nr))
    r1[1] = 1.
    r1[1:]*=.5
    
    # bring the functions to physical space
    y1 = idct(x1, type = 2)
    y2 = idct(x2, type = 2)
    r  = idct(r1, type = 2)
    
    # multiply in physical
    y3 = y1*y2*(a*r + b)**2
    # done this way to approximate QuICC at most
    
    # bring back to spectral space
    c = dct(y3, type=2)/(2*len(y3))
    c = c[:nr]
        
    # represent the cheb indices in the mathematically correct way
    c[1:] *= 2.

    # integrate, prepare index
    idx = np.arange(len(c))
    # and compute integration weight for integration from -1 to +1
    w = (1+(-1)**idx)/(1-idx**2)
    w[1]=0

    return np.sum(w*c)/4
