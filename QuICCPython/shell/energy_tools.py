import numpy as np
from scipy.fftpack import dct, idct
from numpy.polynomial.chebyshev import chebder, chebval, chebgauss

def ortho_pol_q(nr, a, b, x1,  x2, xmin = -1, xmax = 1, I1 = None):
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

    # This was the old implementation, using the standard Chebyshev
    # representation, with analytic integral over \left[-1,1\right]
    # represent the cheb indices in the mathematically correct way
    """    
    c[1:] *= 2.
    
    # integrate, prepare index
    idx = np.arange(len(c))
    # and compute integration weight for -1 to +1
    w = (1+(-1)**idx)/(1-idx**2)
    w[1]=0
    #return np.sum(w*c)/4.
    x,w = chebgauss(len(y3))
    return np.sum(w*y3) * a

    """
    # Assume now that c is in the same "wight standard" of x1 and x2
    
    Ic = I1*c
    temp = chebval([xmax, xmin], Ic)
    

    return temp[0] - temp[1]



def ortho_pol_s(nr, a, b, x1,  x2, xmin = -1, xmax = 1, I1 = None):
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
    """
    c[1:] *= 2.
    
    # integrate, prepare index
    idx = np.arange(len(c))

    # and compute integration weight for -1 to +1
    w = (1+(-1)**idx)/(1-idx**2)
    w[1]=0

    #return np.sum(w*c)/4
    
    x,w = chebgauss(len(y3))
    return np.sum(w*y3) * a

    """
    Ic = I1*c
    temp = chebval([xmax, xmin], Ic)
    
    return temp[0] - temp[1]



def ortho_tor(nr, a, b, x1,  x2, xmin = -1, xmax = 1, I1 = None,
              operation = 'simple', l = None):
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

    if operation == 'simple':

        # bring the functions to physical space
        y1 = idct(x1, type = 2)
        y2 = idct(x2, type = 2)
        r  = idct(r1, type = 2)
        
        # multiply in physical
        y3 = y1*y2*(a*r + b)**2
        # done this way to approximate QuICC at most
    elif operation == 'curl':
        
        dx1 = np.append(chebder(x1), 0.)
        d2x1 = np.append(chebder(dx1), 0.)
        dx2 = np.append(chebder(x2), 0.)
        d2x2 = np.append(chebder(dx2), 0.)

        # transform everything
        r  = idct(r1, type = 2)
        r = (a*r + b)
        y1 = idct(x1, type = 2)
        y2 = idct(x2, type = 2)
        dy1 = idct(dx1, type = 2)
        dy2 = idct(dx2, type = 2)
        d2y1 = idct(d2x1, type = 2)
        d2y2 = idct(d2x2, type = 2)
        # compute the 2 pieces of the bilinear operator
        Diss1 = r*d2y1 + 2*dy1 - l*(l+1)/r*y1
        Diss2 = r*d2y2 + 2*dy2 - l*(l+1)/r*y2
        y3 = Diss1*Diss2
        
    # bring back to spectral space
    c = dct(y3, type=2)/(2*len(y3))
    c = c[:nr]
        
    # represent the cheb indices in the mathematically correct way
    """
    c[1:] *= 2.

    # integrate, prepare index
    idx = np.arange(len(c))
    # and compute integration weight for integration from -1 to +1
    w = (1+(-1)**idx)/(1-idx**2)
    w[1]=0

    #return np.sum(w*c)/4
    x,w = chebgauss(len(y3))
    return np.sum(w*y3)*a

    """
    Ic = I1*c
    temp = chebval([xmax, xmin], Ic)
    
    return temp[0] - temp[1]


def solid_angle_average_pol_q(nr, a, b, x1):
    # precondition: assume that the weight of the spectra is already in DCT format

    # make copies
    x_1 = np.array(x1)
        
    # pad the data
    x1 = np.zeros(int(3/2*nr))
    x1[:len(x_1)] = x_1
    xx = np.zeros_like(x1)
    xx[1] = .5
      
    # bring the functions to physical space
    y1 = idct(x1, type = 2)
    r = idct(xx, type = 2)*a + b
    y1 = y1/r
        
    # multiply in physical
    y3 = y1**2

    return y3*r**2


def solid_angle_average_pol_s(nr, a, b, x1):
    # precondition: assume that the weight of the spectra is already in DCT format

    # take derivative
    # first bring back to correct form
    x_1a = np.array(x1)
    x_1a[1:] *= 2
    dx_1 = chebder(x_1a)
           
    # make copies
    x_1b = np.array(x1)
        
    # pad the data
    x1a = np.zeros(int(3/2*nr))
    x1b = np.zeros(int(3/2*nr))
    
    
    x1a[:len(dx_1)] = dx_1
    x1b[:len(x_1b)] = x_1b
        
    # create the spectral function that describes  T_1=x
    r1 = np.zeros(int(3/2*nr))
    r1[1] = .5
    
    # prepare the chebs for DCT
    x1a[1:] *= .5
        
    # bring the functions to physical space
    y1a = idct(x1a, type = 2)
    y1b = idct(x1b, type = 2)
    x = idct(r1, type = 2)
    r = x*a + b

    # multiply in physical
    y3 = (y1a/a + y1b/r)**2

    return y3*r**2

def solid_angle_average_tor(nr, a, b, x1, operation = 'simple', l = None):
    # precondition: assume that the weight of the spectra is already in DCT format

    # make copies
    x_1 = np.array(x1)
        
    # pad the data
    x1 = np.zeros(int(3/2*nr))
    x1[:len(x_1)] = x_1
    
  
    # create the spectral function that describes  T_1=x
    r1 = np.zeros(int(3/2*nr))
    r1[1] = .5

    # r matches radius
    r  = idct(r1, type = 2) * a + b
    
    if operation == 'simple':

        # bring the functions to physical space
        y1 = idct(x1, type = 2)
                
        # multiply in physical
        y3 = y1**2
        # done this way to approximate QuICC at most
        
    elif operation == 'curl':
        
        dx1 = np.append(chebder(x1), 0.)
        d2x1 = np.append(chebder(dx1), 0.)
        
        # transform everything
        y1 = idct(x1, type = 2)
        dy1 = idct(dx1, type = 2)
        d2y1 = idct(d2x1, type = 2)

        # compute the 2 pieces of the bilinear operator
        Diss1 = d2y1 + 2*dy1/r - l*(l+1)*y1/r**2
        y3 = Diss1**2

    return y3*r**2
