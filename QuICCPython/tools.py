import numpy as np
from numpy.polynomial import chebyshev as cheb
import quicc_bind
from scipy.special import j_roots
import scipy.special as special

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
        quicc_bind.plm(mat, m, x)

        #SingletonPlm.get_dict(SingletonPlm, x)[m] = mat

        #return SingletonPlm.get_dict(SingletonPlm, x)[m][:, l-m]
        return mat


def dplm(maxl, m, x):

    if maxl < m or m < 0 or maxl < 0:
        raise RuntimeError('Problems between l and m')
    
    else:

        # Compute the normalized associated legendre polynomial projection matrix
        mat = np.zeros((maxl-m+1,len(x))).T

        # call the c++ function
        quicc_bind.dplm(mat, m, x)

        #SingletonPlm.get_dict(SingletonPlm, x)[m] = mat

        #return SingletonPlm.get_dict(SingletonPlm, x)[m][:, l-m]
        return mat


def plm_sin(maxl, m, x):

    if maxl < m or m < 0 or maxl < 0:
        raise RuntimeError('Problems between l and m')
    
    else:

        # Compute the normalized associated legendre polynomial projection matrix
        mat = np.zeros((maxl-m+1,len(x))).T

        # call the c++ function
        quicc_bind.plm_sin(mat, m, x)

        #SingletonPlm.get_dict(SingletonPlm, x)[m] = mat

        #return SingletonPlm.get_dict(SingletonPlm, x)[m][:, l-m]
        return mat

#Jacobi Polynomial from recursions
def JacobiPoly(NN, a, b, x): 
    y = np.zeros((NN+1,len(x)))
    y[0,:] = 1.0+0.0*x #ones 
    if NN>0:
        y[1,:] = 0.5*(a-b+(a+b+2.0)*x)
    
    for n in range(1,NN):
        a1 = 2.*(n+1.)*(n+a+b+1.)*(2.*n+a+b)
        a2 = (2.*n+a+b+1.)*(a*a-b*b)
        a3 = (2.*n+a+b)*(2.*n+a+b+1.)*(2.*n+a+b+2.)
        a4 = 2.*(n+a)*(n+b)*(2.*n+a+b+2.)
        if a1 == 0:
            print(a1)
        y[n+1,:] = ( (a2+a3*x)*y[n,:] - a4*y[n-1,:])/a1
    return y

def worland_norm(n , l):
    """Normalization factor = 1/Norm
    """

    if l == 0:
        if n == 0:
            return np.sqrt(2.0/np.pi)
        else:
            return 2.0*np.exp(special.gammaln(n+1.0) - special.gammaln(n+0.5))
    else:
        return np.sqrt(2.0*(2.0*n+l)*np.exp(special.gammaln(n+l) + special.gammaln(n+1.0) - special.gammaln(n+0.5) - special.gammaln(n+l+0.5)))

def W(n, l, r):
    """
    returns the Worland polynomial divided by the norm
    P_n=(r**l)*eval_jacobi(n, -1/2, l-1/2, 2*r*r-1)
    """
    #Compute Worland norm (the function returns 1/norm) =
    norm=float(worland_norm(n,l))

    p_2r2m1= 2.0*r**2 - 1.0
    p_rl= r**l
    
    p=JacobiPoly(n, -0.5, l-0.5, p_2r2m1)
    p=p[n,:]

    #return p(p_r2)*p_rl
    return norm*p*p_rl


def laplacianW(n, l, r):
    """
    returns: diff(r^2*diff(W(l,n,r^2-1), r), r)/r^2 - l*(l+1) * r^l * W(l, n, 2*r^2-1) =
    
    2 (l + n) r^l (2 (1 + l + n) r^2 JacobiP[-2 + n, 3/2, 3/2 + l, -1 + 2 r^2] 
    + (3 + 2 l) JacobiP[-1 + n, 1/2, 1/2 + l, -1 + 2 r^2])
     
    n: order of Jacobi polynomial 
    l: order of spherical harmonic
    """

    norm=float(worland_norm(n,l))
    amp=2.0*l+2.0*n
    
    p_r2l = r**l
    p_r2 = r**2
    p_2r2m1 = 2*p_r2-1
    
    if (n > 1):
        p1=2.0 * (1.0 + l + n) * JacobiPoly(n-2, 3.0/2, 3.0/2 + l, p_2r2m1) #ok 
        p2= (3.0 + 2.0 * l) * JacobiPoly(n-1, 0.5, 0.5 + l, p_2r2m1) #ok 
        p1=p1[n-2,:]
        p2=p2[n-1,:]
    elif (n==1):
        p1=r*0.0
        p2=(3.0+2.0*l)*JacobiPoly(n-1, 0.5, 0.5 + l, p_2r2m1)
        p2=p2[n-1,:]
        #print p1
    else:
        p1=r*0.0
        p2=r*0.0
    
    #print p1

    return norm*amp*p_r2l*( p_r2*p1 + p2) #normalized
    #return amp*p_r2l*(p_r2*p1(p_2r2m1)+p2(p_2r2m1)) #not normalised

def diffW(n, l, r):
    """
    returns diff(r*W(l,n,r^2-1),r) = r^l (2 (l + n) r^2 JacobiP[-1 + n, 1/2, 1/2 + l, -1 + 2 r^2] 
    + (1 + l) JacobiP[n, -(1/2), -(1/2) + l, -1 + 2 r^2])
    n: order of Jacobi polynomial 
    l: order of spherical harmonic
    """
    
    norm=float(worland_norm(n,l))
    
    p_r2l = r**l
    p_r2 = r**2
    p_2r2m1 = 2.*p_r2 - 1.
    
    
    p2 = (1.0+l) * JacobiPoly(n, -0.5, -0.5 + l, p_2r2m1)
    p2 = p2[n,:]
    
    if (n>0):
        p1 = 2.0*(l+n) * JacobiPoly(n-1, 0.5, 0.5+l, p_2r2m1) 
        p1 = p1[n-1,:]
    else:
        p1 = r*0.0 

    return norm * p_r2l * (p_r2 * p1 + p2)

def legSchmidtM(l, m, x):
    #p_mat=qsh.plm(10, m)
    p_mat=plm(l+1, m, x)
    
    if m==0:
        norm=np.sqrt(2./(2.*l+1.))
    else:
        norm=np.sqrt(4./(2.*l+1.))
    
    #return SchmidtNorm*lpmv(m,l,x)
    return norm*p_mat[:,l-m]

        
def diffLegM(l, m, x):
    #p_mat=qsh.plm(10, m)
    p_mat=plm(l+3, m, x)

    if m==0:
        norm=np.sqrt(2./(2.*l+1.))
        normLp1=np.sqrt(2./(2.*l+3.))
    else:
        norm=np.sqrt(4./(2.*l+1.))
        #normLp1=sqrt(4./(2.*l+3.))
        normLp1=np.sqrt(4./(2.*l+3.))*np.sqrt((l+m+1.0)/(l-m+1.0))
        
    #return SchmidtNorm*((1.0+l-m)*lpmv(m, l+1, x) - (l+1.0)*x*lpmv(m, l, x)) #Fix it !!! 
    #return norm*((1.0+l-m)*p_mat[:,l+1-m] - (l+1.0)*x*p_mat[:,l-m])
    return (normLp1*(1.0+l-m)*p_mat[:,l+1-m] - norm*(l+1.0)*x*p_mat[:,l-m])
    #return (1.0+l-m)*legSchmidtM(l, m, x) - (l+1.0)*x*legSchmidtM(l, m, x)
