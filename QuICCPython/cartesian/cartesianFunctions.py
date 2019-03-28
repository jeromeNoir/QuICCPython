import h5py
import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.fftpack import dct, idct


def PointValue(data, field, Xvalue, Yvalue, Zvalue):
    """
        INPUT:
        data
        field
        Xvalue
        Yvalue
        Zvalue
        OUTPUT:
        the value of the field at the point (Xvalue,Yvalue,Zvalue)
        """

    spectral_coeff = getattr(data.fields,field)
    [nx , ny , nz ] = spectral_coeff.shape

    spectral_coeff[:,0,:] = 2*spectral_coeff[:,0,:]
    spectral_coeff_cc = spectral_coeff[:, :, :].real - 1j*spectral_coeff[:, :, :].imag
    
    total = np.zeros((nx, int(ny*2) -1, nz), dtype=complex)
    
    for i in range(0,ny-1):
        total[:,ny-1-i,:] = spectral_coeff_cc[:,i+1,:]
    #total[:,(ny):,:] = np.fliplr(spectral_coeff_cc[:,(ny-1):,:])
    
    total[:,:(ny),:] = spectral_coeff[:,:,:]
    
    [nx2 , ny2 , nz2 ] =total.shape

    Proj_cheb = computeChebEval(nz2, 1.0, 0, Zvalue)
    Proj_fourier_x = computeFourierEval(nx2, Xvalue)
    Proj_fourier_y = computeFourierEval(ny2, Yvalue)

    value1 = np.dot(total, Proj_cheb.T)
    value2 = np.dot(value1, Proj_fourier_y.T)
    value3 = np.dot(value2.T,Proj_fourier_x.T )

    return float(2*value3.real)



def getVerticalSlice(data, field, direction, level):
    """
        INPUT:
        data
        field
        direction = either 'x' or 'y'
        level = should be a value (0, 2pi) you want in that direction
        OUTPUT:
        the 2D array of the slice
        """

    if direction == 'x':
        real_field = getVerticalSliceInX(data, field, level)

    elif direction == 'y':
        real_field = getVerticalSliceInY(data, field, level)

    else:
        raise RuntimeError('Direction for vertical slice given incorrectly')

    return real_field


def getVerticalSliceInX(data, field, level):
    
    spectral_coeff = getattr(data.fields,field)
    [nx , ny , nz ] = spectral_coeff.shape
    PI = computeFourierEval(nx, level);

    test = np.zeros((ny,nz), dtype=complex)

    # This is needed to get the right scalings - probably an artifact of the c++ fftw3 versus np.fft.irfft2
    spectral_coeff[:,0,:] = 2*spectral_coeff[:,0,:]
    
    for i in range(0,nz):
        test[:,i] = np.ndarray.flatten(np.dot(PI,spectral_coeff[:,:,i]))
        
        padfield = np.zeros( (int((ny+1)*3/2), int(nz*3/2)  ), dtype=complex)
        padfield[ :ny, :nz] = test[:,:]
        
        
        real_field = np.zeros((int((ny+1)*3/2), int(nz*3/2)),  dtype=complex)
        real_field2 = np.zeros((int(ny*3), int(nz*3/2)),  dtype=complex)
        
        for i in range(0, int((ny)*3/2)):
            real_field[i,:] = idct(padfield[i,:])
        
        for i in range(0,int(nz*3/2)):
            real_field2[:,i] = np.fft.irfft(real_field[:,i])
        
        [ny2 , nz2] = real_field2.shape
        
        real_field2 = real_field2*ny2

    return real_field2


def getVerticalSliceInY(data, field, level):

    spectral_coeff = getattr(data.fields,field)
    [nx , ny , nz ] = spectral_coeff.shape
    PI = computeFourierEval(2*ny-1, level);
    
    spectral_coeff[:,0,:] = 2* spectral_coeff[:,0,:]
    spectral_coeff_cc = spectral_coeff[:, :, :].real - 1j*spectral_coeff[:, :, :].imag
    
    total = np.zeros((nx, int(ny*2) -1, nz), dtype=complex)
    
    for i in range(0,ny-1):
        total[:,ny-1-i,:] = spectral_coeff_cc[:,i+1,:]
    
    total[:,:(ny),:] = spectral_coeff[:,:,:]
    
    test = np.zeros((nx,nz), dtype=complex)
    
    for i in range(0,nz):
        test[:,i] = np.ndarray.flatten(np.dot(total[:,:,i], PI.T))

    padfield = np.zeros((int((nx+1)*3/2), int(nz*3/2)  ), dtype=complex)
    padfield[:(int((nx+1)/2)+1),  :nz] = test[:(int((nx+1)/2)+1),:]
    padfield[-(int((nx)/2)):,  :nz] = test[-(int((nx)/2)):,  :]
    

    real_field = np.zeros((int((nx+1)*3/2), int(nz*3/2)),  dtype=complex)
    real_field2 = np.zeros((int((nx+1)*3/2), int(nz*3/2)),  dtype=complex)
    
    for i in range(0, int((nx+1)*3/2)):
        real_field[i,:] = idct(padfield[i,:])

    for i in range(0,int(nz*3/2)):
        real_field2[:,i] = np.fft.ifft(real_field[:,i])
        
        [nx2 , nz2] = real_field2.shape
        real_field2 = real_field2*nx2*2

    return real_field2

def getHorizontalSlice(data,field, level):
    """
        INPUT:
        data (structure from reading file in)
        field to get the slice from
        level = should be a value -1 to 1 to denote the level you want in Z
        OUTPUT:
        horizontal slice
    """

    spectral_coeff = getattr(data.fields,field)
    [nx , ny , nz ] = spectral_coeff.shape
    x = np.array([level])
    PI = computeChebEval(nz, 1.0, 0, x);
    
    # This is needed to get the right scalings -probably an artifact of the c++ fftw3 versus np.fft.irfft2
    spectral_coeff[:,0,:] = 2* spectral_coeff[:,0,:]
    
    padfield = np.zeros((int((nx+1)*3/2), int((ny+1)*3/2), nz  ), dtype=complex)
    padfield[:(int((nx+1)/2)+1), :ny, :] = spectral_coeff[:(int((nx+1)/2)+1),:,:]
    padfield[-(int((nx)/2)):, :ny, :] = spectral_coeff[-(int((nx)/2)):, :, :]
    
    real_field = np.fft.irfft2(np.dot(padfield, PI.T))
    [nx2 , ny2  ] = real_field.shape
    real_field = real_field*nx2*ny2

    
    return real_field

def computeFourierEval(nr, r):
    # evaluates the projection matrix for the fourier basis
    
    xx = np.array(r);
    v = np.zeros((nr,1),dtype=complex)
    nr2 = int(nr/2)
    for k in range(1,nr2+1):
        v[k] = np.exp(k*xx*1j)
    for k in range(1,nr2+1):
        v[nr-k] = np.exp(-(k)*xx*1j)
    v[0] = 0.5;

    return np.mat(v.transpose())

def computeChebEval(nr, a, b, r):
    # evaluates the projection matrix for the chebyshev basis

    xx = (np.array(r)-b)/a
    coeffs = np.eye(nr)*2.0
    coeffs[0,0]=1.0

    return np.mat(cheb.chebval(xx, coeffs).transpose())
