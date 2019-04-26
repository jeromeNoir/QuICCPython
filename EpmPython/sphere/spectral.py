import numpy as np
#from numpy.polynomial import chebyshev as cheb
import sys
#sys.path.append('/home/nicolol/workspace/QuICC/Scripts/Python/pybind11/')
import quicc_bind as quicc_pybind
#sys.path.append('/Users/leo/quicc-github/QuICC/Scripts/Python/z-integral/')
#import rotationModules as rot
from scipy.special import j_roots
import scipy.special as special
import time
import EpmPython.read as read 


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

def idxlm(nL, nM):
    idxlm = {}
    ind = 0
    for l in range(0, nL):
        for m in range(0, min(l, nM-1)+1):
            idxlm[(l,m)] = ind
            ind = ind + 1

    return idxlm

def writeStateFile(state, filename):
    """
    Store state file into file
    """

def computeUniformVorticity(state, rmax):
    """
    function to compute uniform vorticity from a state file from 0 to rmax
    Input: 
    - state: input state to compute uniform vorticity
    - rmax : maximum radius to compute vorticity
    """
    #rot.projectWf(dataT, E, nN, nL, nM, "rotation")
    #Projection of spectral coefficients onto the T10, T11 modes
    #Computing Wf with n=0,1,2,... nN-1
    #dataT: Toroidal data 
    #E: Ekman number
    #nN: number of radial modes
    #nL: number of spherical harmonic order 
    #nM: number of azimuthal wave number 
    #scale: 'rotation' or 'viscous' timescale
    dataT = state.fields.velocity_tor
    nN = state.specRes.N
    nL = state.specRes.L
    nM = state.specRes.M
    E = state.parameters.E

    #2*nX + 1 = 2*NN+l + 3
    l = 1
    nX = (nN+l//2 + 1)

    #Jacobi polynomial roots and weights 
    #Legendre alpha=0, beta=0
    alpha = 0
    beta = 0
    roots, w = j_roots(nX, alpha, beta)

    #ekmanR=1.0
    #ekmanR=1.-10.*np.sqrt(E)
    ekmanR = rmax

    #x = [-a,b] 
    #z = [0,1]
    grid_r=(roots+1.)/(2.)*ekmanR
    poly = np.empty((nX, nN), dtype = 'float64', order = 'F')
    quicc_pybind.wnl(poly, l, grid_r)
    
    #index table
    idx=idxlm(nL,nM)

    Y11R=0
    Y11I=0
    Y10=0

    #compute integral 
    for n in range(0,nN):
        #print('nN', nN, 'n', n, 'dataT.shape:', dataT.shape)
        #breakpoint()
        #print("idx:",idx[1,1])
        fi_wi=-2*grid_r**3*poly[:, n]*w
        #print(grid_r**3*poly[:, n]*w)
        #print(fi_wi.sum()*ekmanR/2)
        Y11R = Y11R + dataT[idx[1,1], n].real*fi_wi.sum()*(ekmanR/2.)*5./(ekmanR**5)
        #print(dataT[idx[1,1], n,0])
        fi_wi= 2*grid_r**3*poly[:, n]*w
        Y11I = Y11I + dataT[idx[1,1], n].imag*fi_wi.sum()*(ekmanR/2.)*5./(ekmanR**5)
        #print(dataT[idx[1,1], n, 1])
        #print(grid_r**3*poly[:, n]*w)
        fi_wi=grid_r**3*poly[:, n]*w
        #print(fi_wi.sum()*ekmanR/2)
        #print(dataT[idx[1,0], n, 0])
        Y10 = Y10 + dataT[idx[1,0], n].real*fi_wi.sum()*(ekmanR/2.)*5./(ekmanR**5)
        #print("projectWf(n=",n,")", Y11R, Y11I, Y10)
    
    omegaF = np.zeros(3)
    omegaF[0]=Y11R
    omegaF[1]=Y11I
    omegaF[2]=Y10

    return omegaF

def schmidtN(LL, MM):
    """
    Schmidt normalization factors
    LL: max spherical harmonic order 
    MM: max spherical harmonic degree
    """
    NF=np.array([])
    ind =  0
    #TODO: Leo add exception to verify size 
    for l in range(0, int(LL)+1):
            for m in range(0,min(l,MM)+1):
                if (m==0):
                    #print(l,m)
                    #NF[ind]=1.0
                    #NF=1.0
                    NF=np.append(NF,1.0) #Schmidt normalization for m = 0 
                else:
                    #print(l,m)
                    #NF[ind]=np.sqrt(2)/2;
                    #NF=np.sqrt(2)/2
                    NF=np.append(NF,np.sqrt(2)/2) #Schmidt normalization for m != 0
                
                ind=ind+1;
    return NF

def rotateStateSH(state, scale, phi0, theta0):
    """
    optimized rotations using pybind11 and pseudospectral rotations
    state: state.hdf5 to rotate
    scale: 'rotation' or 'viscous' timescale
    phi0: euler angle azimuthal rotation
    theta0: euler angle tilt
    """ 
    #f = h5py.File(state, 'r') #read state 
    #Spectral resolution 
    LL=state.specRes.L-1 #int(np.array(f['/Truncation/L'], dtype='double'))
    MM=state.specRes.M-1 #int(np.array(f['/Truncation/M'], dtype='double'))
    NN=state.specRes.N-1 #int(np.array(f['/Truncation/N'], dtype='double'))

    #Toroidal Velocity
    dataT=state.fields.velocity_tor #np.array(f['/Velocity/VelocityTor'][:])
    #Poloidal Velocity
    dataP=state.fields.velocity_pol #np.array(f['/Velocity/VelocityPol'][:])
    #Codensity
    dataC=state.fields.codens_ity #np.array(f['/Codensity/Codensity'][:])

    E = state.parameters.E
    #E=f['PhysicalParameters/E'][()]
    #f.close()

    #Y11R, Y11I, Y10, phi0, theta0 = projectWf(dataT, E, NN, LL, MM, scale)
    
    #### determine axis of fluid from uniform vorticity
    # using thetat0 and phi0 determined from mean vorticity
    # theta0=0.837615978648410;
    # phi0=5.569620149823718;
    #Euler Angles
    #Alpha: 1st rotation about Z
    #R = R(alpha-pi/2, -pi/2, beta)* R(0,pi/2, gamma+pi/2)
    alp=-phi0-np.pi/2.
    #Beta: 2nd rotation about X
    bta=-theta0
    #Gamma: 3rd rotation about Z
    #gam=0
            
    #Rotate about Z by alpha
    Qlm = np.zeros(dataT[:,0].shape[0], dtype='complex128', order='F')
    Slm = np.zeros(dataT[:,0].shape[0], dtype='complex128', order='F')

    NF = schmidtN(LL, MM)
   
    #print(Qlm.shape)
    #print(dataT[:,0][0])
 
    #Rotating EPM results
    for n in range(0, NN+1 ): #N worland polynomials 
    #for n in range(1,2): #First polynomial 
        ind=0 
        #Dividing by normalisation 
        #print(NF.shape)
        #print(dataT[:,n,0].shape)
        Qlm = dataT[:, n] / NF #dataT[:, n, 0] / NF + 1j*dataT[:, n, 1] / NF #index, n, imag
        ### Azimuthal rotation (Z) by (-ALPHA - PI/2)
        quicc_pybind.Zrotate(Qlm, Slm, alp, LL, MM)
        ###### tilt by bta X(-bta)
        quicc_pybind.Xrotate(Slm, Qlm, bta, LL, MM)
       
        #multiplying by normalization
        dataT[:, n] = Slm * NF

        #Dividing by normalisation 
        Qlm = dataP[:, n] / NF #dataP[:, n, 0] / NF + 1j*dataP[:, n, 1] / NF #index, n, imag
        ### Azimuthal rotation (Z) by (-ALPHA - PI/2)
        quicc_pybind.Zrotate(Qlm, Slm, alp, LL, MM)
        ###### tilt by bta X(-bta)
        quicc_pybind.Xrotate(Slm, Qlm, bta, LL, MM)
        #multiplying by normalization
        dataP[:, n] = Slm * NF
        
        #Dividing by normalisation 
        #Qlm = dataC[:, n, 0] / NF + 1j*dataC[:, n, 1] / NF #index, n, imag
        Qlm = dataC[:, n] / NF
        ### Azimuthal rotation (Z) by (-ALPHA - PI/2)
        quicc_pybind.Zrotate(Qlm, Slm, alp, LL, MM)
        ###### tilt by bta X(-bta)
        quicc_pybind.Xrotate(Slm, Qlm, bta, LL, MM)
  
        #multiplying by normalization
        dataC[:, n] = Slm * NF
   
    #print('dataT:',dataT.shape[1], 'n', n)
    
    #if (n!= dataC):
    #    raise RuntimeError("Requested row normalization is inconsistent")

    del Qlm, Slm

    #rotatedState = (dataT, dataP, dataC)
    #return rotatedState
    #create a copy of the statefile to rewrite and return  
    rotState = read.SpectralState(state.filename)
    rotState.fields.velocity_tor = dataT
    rotState.fields.velocity_pol = dataP
    rotState.fields.codens_ity = dataC

    return rotState

def alignAlongFluidAxis(state, omegaF):
    """
    function to align state along fluid axis
    Input:
    - state: input state to align 
    - omega_f : axis of rotation of the flow
    """
    #rotate(state, alpha, beta, 0)

    Y11R, Y11I, Y10 = omegaF 
    scale = "rotation"   

    #+Omega_0 
    #Leo: Adding Omega_0 to the flow because we are in the mantle frame?
    if scale == 'viscous':
        Y10=Y10+1.0/E #For viscous time-scale
    else:
        Y10=Y10+1.0 #For rotation time-scale
        
    if (Y11I>=0 and Y11R>=0):
        phi0=np.arctan(Y11I/Y11R)
    elif (Y11R<0):
        phi0=np.arctan(Y11I/Y11R)+np.pi
    else:
        phi0=np.arctan(Y11I/Y11R)+2*np.pi

    theta0=np.arctan(np.sqrt(Y11R**2+Y11I**2)/Y10)
        
    rotState = rotateStateSH(state, scale, phi0, theta0)

    return rotState

def getGammaF(filename):
    """
    computes and stores angle of rotation of the fluid
    Input: 
    - filename : file to store gamma
    """

def goToFluidFrameOfReference(state, gammaF):
    """
    rotate state with the fluid
    Input:
    - state: state to rotate
    - gammaF: angle of rotation 
    """

    """
    optimized rotations using pybind11 and pseudospectral rotations
    scale: 'rotation' or 'viscous' timescale
    phi0: euler angle azimuthal rotation
    theta0: euler angle tilt
    gam: azimuthal rotation given by \int \omega_f(t) dt  
    """ 
    #f = h5py.File(state, 'r') #read state 
    #Spectral resolution 
    LL=state.specRes.L-1 #int(np.array(f['/Truncation/L'], dtype='double'))
    MM=state.specRes.M-1 #int(np.array(f['/Truncation/M'], dtype='double'))
    NN=state.specRes.N-1 #int(np.array(f['/Truncation/N'], dtype='double'))

    #Toroidal Velocity
    dataT=state.fields.velocity_tor #np.array(f['/Velocity/VelocityTor'][:])
    #Poloidal Velocity
    dataP=state.fields.velocity_pol #np.array(f['/Velocity/VelocityPol'][:])
    #Codensity
    dataC=state.fields.codens_ity #np.array(f['/Codensity/Codensity'][:])

    E = state.parameters.E
    #E=f['PhysicalParameters/E'][()]
    #f.close()

    #Y11R, Y11I, Y10, phi0, theta0 = projectWf(dataT, E, NN, LL, MM, scale)
            
    #Rotate about Z by alpha
    Qlm = np.zeros(dataT[:,0].shape[0], dtype='complex128', order='F')
    Slm = np.zeros(dataT[:,0].shape[0], dtype='complex128', order='F')

    NF = schmidtN(LL, MM)
   
    #print(Qlm.shape)
    #print(dataT[:,0][0])
 
    #Rotating EPM results
    for n in range(0, NN+1 ): #N worland polynomials 
    #for n in range(1,2): #First polynomial 
        ind=0 
        #Dividing by normalisation 
        Qlm = dataT[:, n] / NF #dataT[:, n, 0] / NF + 1j*dataT[:, n, 1] / NF #index, n, imag
        #Toroidal
        quicc_pybind.Zrotate(Qlm, Slm, -gammaF, LL, MM)
       
        #multiplying by normalization
        dataT[:, n] = Slm * NF

        #Dividing by normalisation 
        Qlm = dataP[:, n] / NF #dataP[:, n, 0] / NF + 1j*dataP[:, n, 1] / NF #index, n, imag
        #Poloidal    
        quicc_pybind.Zrotate(Qlm, Slm, -gammaF, LL, MM)
        #multiplying by normalization
        dataP[:, n] = Slm * NF
        
        #Dividing by normalisation 
        #Qlm = dataC[:, n, 0] / NF + 1j*dataC[:, n, 1] / NF #index, n, imag
        Qlm = dataC[:, n] / NF
        #Codensity    
        quicc_pybind.Zrotate(Qlm, Slm, -gammaF, LL, MM)
  
        #multiplying by normalization
        dataC[:, n] = Slm * NF
   
    #print('dataT:',dataT.shape[1], 'n', n)    
    #if (n!= dataC):
    #    raise RuntimeError("Requested row normalization is inconsistent")
    del Qlm, Slm

    rotState = read.SpectralState(state.filename)
    #(dataT, dataP, dataC)
    #return rotatedState
    rotState.fields.velocity_tor = dataT
    rotState.fields.velocity_pol = dataP
    rotState.fields.codens_ity = dataC
    return rotState

def sz2ct(s,z):
    return z/np.sqrt(z*z+s*s)

def sz2r(s,z):
    return np.sqrt(z*z+s*s)

def sz2st(s,z):
    return s/np.sqrt(z*z+s*s)

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

class Integrator():
    def __init__(self, _grid_s, _tor, _pol, _res, _var):
        self.grid_s = _grid_s
        self.tor = _tor
        self.pol = _pol
        self.res = _res
        self.type = _var

def getZIntegrator(state, var="vortZ", nNs=40):
    """
    compute integrator 
    Input: 
    - var: variable to integrate vortZ, uS, or uPhi
    """

    nL=state.specRes.L
    nM=state.specRes.M
    nN=state.specRes.N
    E=state.parameters.E

    nNmax, nLmax, nMmax, nNs = nN, nL, nM, nNs
    
    N_z = nNmax + nLmax // 2  # function of Nmax and Lmax

    # Jacobi polynomial roots and weights
    # Legendre alpha=0, beta=0
    alpha = 0
    beta = 0
    roots, w = j_roots(N_z, alpha, beta)

    d = 10. * np.sqrt(E)
    ekmanR = 1. - d

    grid_s = np.linspace(0, ekmanR, nNs + 1)

    grid_s = 0.5 * (grid_s[0:-1] + grid_s[1:])
    grid_z = np.zeros((nNs, N_z))
    grid_cost = np.zeros_like(grid_z)
    grid_sint = np.zeros_like(grid_z)
    grid_r = np.zeros_like(grid_z)

    # Generating (s,z) grids
    for id_s, s in enumerate(grid_s):
        # Quadrature points
        #z_local = roots * sqrt(1.0 - s * s) * (1.0 - d)  # from -1 to 1
        z_local = roots * np.sqrt((1.0-d)*(1.0-d) - s*s)  # from -1 to 1
        grid_cost[id_s, :] = sz2ct(s, z_local)  # from -1 to 1
        grid_sint[id_s, :] = sz2st(s, z_local)
        grid_r[id_s, :] = sz2r(s, z_local)
        grid_z[id_s, :] = z_local
    
    #vortz_tor = np.zeros((nNs, nNmax, nLmax * (nLmax + 1) // 2), dtype=complex)
    vortz_tor = np.zeros((nNs, nNmax, nMmax*(nMmax+1)//2 + nMmax*(nLmax-nMmax) ), dtype=complex)
    vortz_pol = np.zeros_like(vortz_tor)
    #ur_pol = np.zeros_like(vortz_tor)
    uphi_tor = np.zeros_like(vortz_tor)
    uphi_pol = np.zeros_like(vortz_tor)
    #uth_tor = np.zeros_like(vortz_tor)
    #uth_pol = np.zeros_like(vortz_tor)
    us_tor = np.zeros_like(vortz_tor)
    us_pol = np.zeros_like(vortz_tor)

    tr0=time.time()

    for id_s in range(0, nNs):
        #print('id_s=', id_s)
        print('id_s: ', id_s, end=" ")  # ' timestep: ', timestep)
        t0 = time.time()    
        for n in range(0, nNmax):  # debug
            # print('n=', n)
            ind = 0
            for l in range(0, nLmax):
                #for m in range(0, min(l, Mmax) + 1):
                for m in range(0, min(l, nMmax-1) + 1):
                    # pdb.set_trace()
                    # Leo: compute on the fly instead of using memory 
                    r_local = grid_r[id_s, :]
                    cos_local = grid_cost[id_s, :]
                    sin_local = grid_sint[id_s, :]

                    # polynomials for reconstruction
                    plm_w1 = W(n, l, r_local)
                    plm_lapw1 = laplacianW(n, l, r_local)
                    plm_diffw1 = diffW(n, l, r_local)

                    plm_leg = legSchmidtM(l, m, cos_local)
                    plm_diffLeg = diffLegM(l, m, cos_local)

                    if var == "vortZ":
                        # computing contributions from toroidal and poloidal vorticity onto Z
                        vort_z = 1j * m * plm_lapw1 * plm_leg  # times P
                        vortz_tor[id_s, n, ind] = sum(w*vort_z)*np.sqrt(1-grid_s[id_s]**2)*(1.0-d)

                        vort_z = l * (l + 1.0) * plm_w1 * plm_leg * cos_local / r_local \
                                                 - plm_diffw1 * plm_diffLeg / r_local  # times P
                        vortz_pol[id_s, n, ind] = sum(w*vort_z)*np.sqrt(1-grid_s[id_s]**2)*(1.0-d)

                        #sum(w*vort_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d)

                    elif var == "uPhi":
                        #uth_tor[id_s, n, ind, :] = 1j * m * plm_w1 * plm_leg / sin_local
                        u_phi = -plm_w1 * plm_diffLeg / sin_local # -\partial_\theta T
                        uphi_tor[id_s, n, ind] = sum(w*u_phi)*np.sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    
                        #ur_pol[id_s, n, ind, :] = l * (l + 1.0) * plm_w1 * plm_leg / r_local  # 1/r * L2(P)
                        #uth_pol[id_s, n, ind, :] = plm_diffw1 * plm_diffLeg / (r_local * sin_local)  # ???

                        # 1/(r sin(theta)) \partial_r * r * \partial_\phi P
                        u_phi = 1j * m * plm_diffw1 * plm_leg / (r_local * sin_local)
                        uphi_pol[id_s, n, ind] = sum(w*u_phi)*np.sqrt(1-grid_s[id_s]**2)*(1.0-d)

                    elif var == "uS":
                        # u_s = u_r * sin(theta) + u_theta * cos(theta)
                        u_s = l * (l + 1.0) * plm_w1 * plm_leg * sin_local / r_local + plm_diffw1 * plm_diffLeg * cos_local / (
                                                                                    r_local * sin_local)
                        us_pol[id_s, n, ind] = sum(w*u_s)*np.sqrt(1-grid_s[id_s]**2)*(1.0-d)

                        u_s = 1j * m * plm_w1 * plm_leg * cos_local / sin_local
                        us_tor[id_s, n, ind] = sum(w*u_s)*np.sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    
                        # u_z = u_r * cos(theta) - u_theta * sin(theta)
                        #uz_pol[id_s, n, ind, :] = l * (
                        #            l + 1.0) * plm_w1 * plm_leg * cos_local / r_local - plm_diffw1 * plm_diffLeg / r_local
                        #uz_tor[id_s, n, ind, :] = 1j * m * plm_w1 * plm_leg
                        # h = u_z * vort_z

                    ind = ind + 1

            #Make sure dimensions idx[l,m] agree 
            #assert(ind == vortz_tor.shape[2]), "size of integrator and ind=idx[l,m] don't agree"
            #assert(ind == vortz_pol.shape[2]), "size of integrator and ind=idx[l,m] don't agree"
            #assert(ind == us_tor.shape[2]), "size of integrator and ind=idx[l,m] don't agree"
            #assert(ind == us_pol.shape[2]), "size of integrator and ind=idx[l,m] don't agree"
            #assert(ind == uphi_tor.shape[2]), "size of integrator and ind=idx[l,m] don't agree"
            #assert(ind == uphi_pol.shape[2]), "size of integrator and ind=idx[l,m] don't agree"

        t1 = time.time()
        print("time: ", t1-t0)

    tr1=time.time()
    print("Total time:", tr1-tr0)

    #resolution 
    res = (nNmax, nLmax, nMmax, nNs)

    if var == "vortZ":
        return Integrator(grid_s, vortz_tor, vortz_pol, res, var)
    elif var == "uPhi":
        return Integrator(grid_s, uphi_tor, uphi_pol, res, var)
    elif var == "uS":
        return Integrator(grid_s, us_tor, us_pol, res, var)
    else:
        print("output variable must be defined omegaZ, uPhi, or uS")

def readIntegrator(filename):
    """
    read integrator from a filename
    """

def computeZIntegral(state, zInt):
    """
    compute z-integral for a given state 
    Input:
    - state: state to integrate
    - integrator: pre-computed integration matrix
    """

    """
    returns the z-integral of vorticity and the flow up to 
    a max spherical harmonic order Mmax and Worland Polynomial (Nmax)
    vort_int=computeZintegral(state, vortz_tor, vortz_pol, ur_pol, uphi_tor, uphi_pol, (Nmax, Lmax, Mmax, N_s), w, grid_s)
    """ 
    nL=state.specRes.L
    nM=state.specRes.M
    nN=state.specRes.N
    dataP=state.fields.velocity_pol
    dataT=state.fields.velocity_tor
    E=state.parameters.E
    
    nNmax, nLmax, nMmax, nNs = zInt.res
    #w = zInt.w
    #grid_s = zInt.grid_s
    
    #ekmanRemover=1.-10.*sqrt(E)
    d = 10.*np.sqrt(E)
    
    idx = idxlm(nL, nM)
    idxM = idxlm(nLmax, nMmax)
    
    #Allocating memory for output
    zIntegral = np.zeros((nMmax, nNs), dtype=complex)

    #Make sure dimensions agree 
    assert(dataT.shape[0]==len(idx))
    assert(dataP.shape[0]==len(idx))
    assert(zInt.tor.shape[2] == len(idxM))
    assert(zInt.pol.shape[2] == len(idxM))

    #main idea compute for each m the integral 
    #Int(f(s,z)dz) = sum(weight*vort_z(s,z)) = f(s)
    #vort_z function that returns the component of the z-vorticity
    for id_s in range(0, nNs):
        for n in range(0, nNmax): 
            for l in range(0, nLmax):
                #TODO: Leo: add exception to verify size 
                #for m in range(0, min(l, Mmax)+1):
                for m in range(0, min(l, nMmax-1)+1):
                    #For real data:
                    _zIntegral = zInt.tor[id_s, n, idxM[l,m]] * dataP[idx[l,m], n] + zInt.pol[id_s, n, idxM[l,m]] * dataT[idx[l,m], n]
                    #Integrate
                    zIntegral[m, id_s] = zIntegral[m, id_s] + _zIntegral #sum(w*vort_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d) #Leo: check this

    return zIntegral

# the function takes care of the looping over modes
def getEquatorialSlice(state, p=0, modeRes = (120,120) ):
    #if not (self.geometry == 'shell' or self.geometry == 'sphere'):
    #    raise NotImplementedError('makeEquatorialSlice is not implemented for the geometry: '+self.geometry)
    # generate indexer
    # this generate the index lenght also
    state.idx = state.idxlm()
    ridx = {v: k for k, v in state.idx.items()}

    #if modeRes == None:
    #    modeRes=(self.specRes.L, self.specRes.M)
    # generate grid
    X, Y, r, phi = state.makeEquatorialGrid()
    grid_r = r
    grid_phi = phi
    # pad the fields
    dataT = np.zeros((state.nModes, state.physRes.nR), dtype='complex')
    dataT[:,:state.specRes.N] = state.fields.velocity_tor
    dataP = np.zeros((state.nModes, state.physRes.nR), dtype='complex')
    dataP[:,:state.specRes.N] = state.fields.velocity_pol
    
    # prepare the output fields
    FR = np.zeros((len(r), int(state.physRes.nPhi/2)+1), dtype = 'complex')
    FTheta = np.zeros_like(FR)
    FPhi = np.zeros_like(FR)
    FieldOut = [FR, FTheta, FPhi]
            
    # initialize the spherical harmonics
    # only for the equatorial values
    #state.makeSphericalHarmonics(np.array([np.pi/2]))
    makeSphericalHarmonics(state, np.array([np.pi/2]))

    for i in range(state.nModes):
        # get the l and m of the index
        l, m = ridx[i]
        # statement to reduce the number of modes considered
        #if l> modeRes[0] or m> modeRes[1]:
        #    continue
        #Core function to evaluate (l,m) modes summing onto FieldOut             
        #state.evaluate_mode(l, m, FieldOut, dataT[i, :], dataP[i,:], r, None, phi, kron='equatorial', phi0=p)
        evaluate_mode(state, l, m, FieldOut, dataT[i, :], dataP[i,:], r, None, phi, kron='equatorial', phi0=p)

    field2 = []

    #loop over the fields
    for i, f in enumerate(FieldOut):
        temp = f
        #m=0 factor
        if i > 0:
            temp[0,:] *= 2.
        #apply irfft
        f = np.fft.irfft(temp, axis=1)
        #normalize
        f = f * len(f[0,:])
        f = np.hstack([f,np.column_stack(f[:,0]).T])
        field2.append(f)
    FieldOut = field2
    #return X, Y, field2
    return {'x': X, 'y': Y, 'uR': FieldOut[0].T, 'uTheta': FieldOut[1].T, 'uPhi': FieldOut[2].T}

def getMeridionalSlice(state, p=0, modeRes = (120,120) ):
    """
    p: phi angle of meridional splice 
    modeRes: resolution of output
    the function takes care of the looping over modes
    TODO: Nico add comments and change syntax   
    """

    #if not (self.geometry == 'shell' or self.geometry == 'sphere'):
    #    raise NotImplementedError('makeMeridionalSlice is not implemented for the geometry: '+self.geometry)

    # generate indexer
    # this generate the index lenght also
    state.idx = state.idxlm()
    # returns (l,m) from index 
    ridx = {v: k for k, v in state.idx.items()}

    """
    if modeRes == None:
        modeRes=(self.specRes.L, self.specRes.M)
    """
    
    # generate grid
    # X, Y: meshgrid for plotting in cartesian coordinates
    # r, theta: grid used for evaluation 
    X, Y, r, theta = state.makeMeridionalGrid()
    
    # pad the fields to apply 3/2 rule 
    dataT = np.zeros((state.nModes, state.physRes.nR), dtype='complex')
    dataT[:,:state.specRes.N] = state.fields.velocity_tor
    dataP = np.zeros((state.nModes, state.physRes.nR), dtype='complex')
    dataP[:,:state.specRes.N] = state.fields.velocity_pol
    
    # prepare the output fields in radial, theta, and phi 
    FR = np.zeros((len(r), len(theta)))
    FTheta = np.zeros_like(FR)
    FPhi = np.zeros_like(FR)
    FieldOut = [FR, FTheta, FPhi]
            
    # initialize the spherical harmonics
    makeSphericalHarmonics(state, theta)

    for i in range(state.nModes):
        
        # get the l and m of the index
        l, m = ridx[i]

        # statement to redute the number of modes considered
        #if l> modeRes[0] or m> modeRes[1]:
        #continue
        #state.evaluate_mode(l, m, FieldOut, dataT[i, :], dataP[i,:], r, theta, None, kron='meridional', phi0=p)
        evaluate_mode(state, l, m, FieldOut, dataT[i, :], dataP[i,:], r, theta, None, kron='meridional', phi0=p)

    return {'x': X, 'y': Y, 'uR': FieldOut[0].T, 'uTheta': FieldOut[1].T, 'uPhi': FieldOut[2].T}

def makeSphericalHarmonics(state, theta):
    # change the theta into x
    x = np.cos(theta)
    state.xth = x

    # initialize storage to 0
    state.Plm = np.zeros((state.nModes, len(x)))
    state.dPlm = np.zeros_like(state.Plm)
    state.Plm_sin = np.zeros_like(state.Plm)

    # generate the reverse indexer
    ridx = {v: k for k, v in state.idx.items()}
    for m in range(state.specRes.M):
        # compute the assoc legendre
        #temp = sphere.plm(state.specRes.L-1, m, x)
        temp = plm(state.specRes.L-1, m, x)
        #temp = wor.plm(self.specRes.L-1, m, x) #Leo: this implementation doesn't work 

        # assign the Plm to storage
        for l in range(m, state.specRes.L):
            state.Plm[state.idx[l, m], :] = temp[:, l-m]
            pass
        
        pass

    # compute dPlm and Plm_sin
    for i in range(state.nModes):
        l, m = ridx[i]
        state.dPlm[i, :] = -.5 * (((l+m)*(l-m+1))**0.5 * state.plm(l,m-1) -
                                 ((l-m)*(l+m+1))**.5 * state.plm(l, m+1, x) )

        if m!=0:
            state.Plm_sin[i, :] = -.5/m * (((l-m)*(l-m-1))**.5 *
                                          state.plm(l-1, m+1, x) + ((l+m)*(l+m-1))**.5 *
                                        state.plm(l-1, m-1)) * ((2*l+1)/(2*l-1))**.5
        else:
            state.Plm_sin[i, :] = state.plm(l, m)/(1-x**2)**.5

    pass

def evaluate_mode(state, l, m, *args, **kwargs):
    """
    evaluate (l,m) mode and return FieldOut
    input: 
    spectral coefficients: dataT, dataP, 
    physical grid: r, theta, phi0
    kron: 'meridional' or 'equatorial'
    outpu: FieldOut
    e.g:
    evaluate_mode(l, m, FieldOut, dataT[i, :], dataP[i, :], r, theta, kron='meridional', phi0=p)
    """
    #print(l, m)
    # raise exception if wrong geometry
    #if not (self.geometry == 'shell' or self.geometry == 'sphere'):
    #    raise NotImplementedError('makeMeridionalSlice is not implemented for the geometry: '+self.geometry)

    # prepare the input data
    #Evaluated fields 
    Field_r = args[0][0]
    Field_theta = args[0][1]
    Field_phi = args[0][2]
    #spectral coefficients
    modeT = args[1]
    modeP = args[2]
    #grid points 
    r = args[3]
    theta = args[4]
    phi = args[5]
    phi0 = kwargs['phi0']

    
    # define factor
    factor = 1. if m==0 else 2.

    if kwargs['kron'] == 'isogrid':
        x = kwargs['x']
        #TODO: Leo
        q_part = None
        s_part = None
        t_part = None
    
    else: # hence either meridional or equatorial
        # prepare the q_part
        # q = l*(l+1) * Poloidal / r            
        # idct: inverse discrete cosine transform
        # type = 2: type of transform 
        # q = Field_radial 
        # prepare the q_part
        # idct = \sum c_n * T_n(xj)
        modeP_r = np.zeros_like(r)

        # prepare the s_part
        # s_part = orthogonal to Field_radial and Field_Toroidal
        # s_part = (Poloidal)/r + d(Poloidal)/dr = 1/r d(Poloidal)/dr
        dP = np.zeros_like(r)

        # prepare the t_part
        # t_part = Field_Toroidal                
        t_part = np.zeros_like(r)

        for n in range(state.specRes.N):
            modeP_r=modeP_r+modeP[n]*W(n, l, r)/r
            dP = dP+modeP[n]*diffW(n, l, r)
            t_part = t_part + modeT[n]*W(n,l,r)

        q_part =  modeP_r*l*(l+1)#same stuff
        s_part = modeP_r + dP 
    
    # depending on the kron type it changes how 2d data are formed
    # Mapping from qst to r,theta, phi 
    # phi0: azimuthal angle of meridional plane. It's also the phase shift of the flow with respect to the mantle frame, 
    if kwargs['kron'] == 'meridional':
        eimp = np.exp(1j *  m * phi0)
        
        idx_ = state.idx[l, m]
        Field_r += np.real(kron(q_part, state.Plm[idx_, :]) *
                           eimp) * factor
        
        Field_theta += np.real(kron(s_part, state.dPlm[idx_, :]) * eimp) * 2
        Field_theta += np.real(kron(t_part, state.Plm_sin[idx_, :]) * eimp * 1j * m) * 2#factor
        Field_phi += np.real(kron(s_part, state.Plm_sin[idx_, :]) * eimp * 1j * m) * 2#factor 
        Field_phi -= np.real(kron(t_part, state.dPlm[idx_, :]) * eimp) * 2

    elif kwargs['kron'] == 'equatorial':
        """
        eimp = np.exp(1j *  m * (phi0 + phi))
        
        idx_ = self.idx[l, m]
        Field_r += np.real(tools.kron(q_part, eimp) *
                           self.Plm[idx_, :][0]) * factor
        
        Field_theta += np.real(tools.kron(s_part, eimp)) * self.dPlm[idx_, :][0] * 2
        Field_theta += np.real(tools.kron(t_part, eimp) * 1j * m) * self.Plm_sin[idx_, :][0] * 2#factor
        Field_phi += np.real(tools.kron(s_part, eimp) * 1j * m) * self.Plm_sin[idx_, :][0] * 2#factor
        Field_phi -= np.real(tools.kron(t_part, eimp)) * self.dPlm[idx_, :][0] * 2
        """
        idx_ = state.idx[l, m]
        
        #Field_r += np.real(tools.kron(self.Plm[idx_, :], eimp) * q_part) * factor
        Field_r[:, m] += state.Plm[idx_, :] * q_part

        #Field_theta += np.real(tools.kron(self.dPlm[idx_, :], eimp) * s_part) * 2
        Field_theta[:, m] += state.dPlm[idx_, :] * s_part
        #Field_theta += np.real(tools.kron(self.Plm_sin[idx_, :], eimp * 1j * m) * t_part) * 2#factor
        Field_theta[:, m] += state.Plm_sin[idx_, :] * 1j * m * t_part
        #Field_phi += np.real(tools.kron(self.Plm_sin[idx_, :], eimp * 1j * m) * s_part) * 2#factor
        Field_phi[:, m] += state.Plm_sin[idx_, :]* 1j * m * s_part
        #Field_phi -= np.real(tools.kron(self.dPlm[idx_, :], eimp) * t_part) * 2
        Field_phi[:, m] -= state.dPlm[idx_, :] * t_part
        
    elif kwargs['kron'] == 'isogrid':
        #eimp = np.exp(1j *  m * (phi0 + phi))
        
        idx_ = state.idx[l, m]
        
        #Field_r += np.real(tools.kron(self.Plm[idx_, :], eimp) * q_part) * factor
        Field_r[:, m] += self.Plm[idx_, :] * q_part

        #Field_theta += np.real(tools.kron(self.dPlm[idx_, :], eimp) * s_part) * 2
        Field_theta[:, m] += state.dPlm[idx_, :] * s_part
        #Field_theta += np.real(tools.kron(self.Plm_sin[idx_, :], eimp * 1j * m) * t_part) * 2#factor
        Field_theta[:, m] += state.Plm_sin[idx_, :] * 1j * m * t_part
        #Field_phi += np.real(tools.kron(self.Plm_sin[idx_, :], eimp * 1j * m) * s_part) * 2#factor
        Field_phi[:, m] += state.Plm_sin[idx_, :]* 1j * m * s_part
        #Field_phi -= np.real(tools.kron(self.dPlm[idx_, :], eimp) * t_part) * 2
        Field_phi[:, m] -= state.dPlm[idx_, :] * t_part
        
    else:
        raise ValueError('Kron type not understood: '+kwargs['kron'])

    pass
            

