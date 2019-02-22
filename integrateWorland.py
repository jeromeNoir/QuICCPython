"""
Functions used to perform a Z-integral using a Legendre Quadrature on a rotated plane 
Last modified echeverl@student.ethz.ch
"""
import sys
sys.path.append('/Users/leo/quicc-github/QuICC/Python')
from scipy.special import j_roots, eval_jacobi, sph_harm, jacobi, lpmv
from scipy.misc import factorial as sciFactorial #using the gamma function 
import quicc.geometry.worland.worland_basis as wb
import numpy as np
import h5py 
from math import *
import cppimport
sys.path.append('/Users/leo/quicc-github/QuICC/Scripts/Python/pybind11')
pybind = cppimport.imp("QuICC")
#Error importing this module 
#import rotationModules as rot

#from memory_profiler import profile
                   
class Integrator():
    def __init__(self,w, grid_s, vortz_tor, vortz_pol, us_tor, us_pol, uphi_tor, uphi_pol, res):
        self.w = w
        self.grid_s = grid_s
        self.vortz_tor = vortz_tor
        self.vortz_pol = vortz_pol
        self.us_tor = us_tor
        self.us_pol = us_pol
        self.uphi_tor = uphi_tor
        self.uphi_pol = uphi_pol
        self.res = res
    
#Useful functions
#Worland polynomial
def W1(n, l):
    #P_n=(r**l)*eval_jacobi(n, alpha, beta, 2*r*r-1)    
    
    #Compute Worland norm (the function returns 1/norm) 
    norm=float(wb.worland_norm(n,l))
    
    p=jacobi(n, -0.5, l-0.5)
    p_r2=np.poly1d([2,0,-1])
    p_rl=np.poly1d([1,0])**l
    #return p(p_r2)*p_rl
    return norm*p(p_r2)*p_rl

#Laplacian of Worland 
def laplacianW1(n, l):
    """
    returns: diff(r^2*diff(W(l,n,r^2-1), r), r)/r^2 - l*(l+1) * r^l * W(l, n, 2*r^2-1) =
    
    2 (l + n) r^l (2 (1 + l + n) r^2 JacobiP[-2 + n, 3/2, 3/2 + l, -1 + 2 r^2] 
    + (3 + 2 l) JacobiP[-1 + n, 1/2, 1/2 + l, -1 + 2 r^2])
     
    n: order of Jacobi polynomial 
    l: order of spherical harmonic
    """
    norm=float(wb.worland_norm(n,l))
    amp=2.0*l+2.0*n
    if (n > 1):
        p1=2.0 * (1.0 + l + n) * jacobi(n-2, 3.0/2, 3.0/2 + l) #ok 
        p2= (3.0 + 2.0 * l) * jacobi(n-1, 0.5, 0.5 + l) #ok 
    elif (n==1):
        p1=np.poly1d([0])
        p2=(3+2*l)*jacobi(n-1, 0.5, 0.5 + l)
        #print p1
    else:
        p1=np.poly1d([0])
        p2=np.poly1d([0])

    p_r2l = np.poly1d([1,0])**l
    p_r2 = np.poly1d([1,0,0])
    p_2r2m1 = np.poly1d([2,0,-1])
    
    #print p1

    return norm*amp*p_r2l*(p_r2*p1(p_2r2m1)+p2(p_2r2m1)) #normalized
    #return amp*p_r2l*(p_r2*p1(p_2r2m1)+p2(p_2r2m1)) #not normalised 

#differential of Worland 
def diffW1(n, l):
    """
    returns diff(r*W(l,n,r^2-1),r) = r^l (2 (l + n) r^2 JacobiP[-1 + n, 1/2, 1/2 + l, -1 + 2 r^2] 
    + (1 + l) JacobiP[n, -(1/2), -(1/2) + l, -1 + 2 r^2])
    
    n: order of Jacobi polynomial 
    l: order of spherical harmonic
    """

    norm=float(wb.worland_norm(n,l))
    
    p2=(1.0+l)*jacobi(n, -0.5, l - 0.5) #ok
    
    if (n>0):
        p1=2.0*(l+n)*jacobi(n-1, 0.5, 0.5 + l) #ok 
    else:
        p1=np.poly1d([0])
    
    p_r2l = np.poly1d([1,0])**l
    p_r2 = np.poly1d([1,0,0])
    p_2r2m1 = np.poly1d([2,0,-1])

    return norm*p_r2l*(p_r2*p1(p_2r2m1)+p2(p_2r2m1)) #normalized 
    #return p_r2l*(p_r2*p1(p_2r2m1)+p2(p_2r2m1)) #not normalized

def LegSchmidt(l, m, x):
    #TODO: use spherical plm:Compute the normalized associated legendre polynomial projection matrix"""
    
    #LegNorm=sqrt(factorial(l-m)/factorial(l+m))
    #compute manually with gamma function 
    if m==0:
        SchmidtNorm=1.0
    else:
        #SchmidtNorm=sqrt(2.0*factorial(l-m)/factorial(l+m)) #this is very slow force it to float
        SchmidtNorm=sqrt(2.0*sciFactorial(float(l-m))/sciFactorial(float(l+m)))
        
    return SchmidtNorm*lpmv(m,l,x)

#differential of Legendre Associated Polynomial 
def diffLeg(l, m, x):
    #returns diff(P(l,m,cos(theta)),theta)
    #m: degree
    #l: order
    #syntax of lpmv:
    #lpmv(m,l,x)
    #return (1.0+l-m)*lpmv(m,l,x)-(l+1.0)*x*lpmv(m,l,x)
    if m==0:
        SchmidtNorm=1.0
    else:
        #SchmidtNorm=sqrt(2.0*factorial(l-m)/factorial(l+m))
        SchmidtNorm=sqrt(2.0*sciFactorial(l-m)/sciFactorial(l+m))
    
    #Using Schimdt seminormalized
    return SchmidtNorm*((1.0+l-m)*lpmv(m, l+1, x) - (l+1.0)*x*lpmv(m, l, x))#Fix it !!! 


#Functions using recurrences
def pmm(maxl, m, x):
    """Compute the normalized associated legendre polynomial of order and degree m"""

    #x = txgrid(maxl, m) 
    mat = np.zeros((len(x), 1))
    mat[:,0] = 1.0/np.sqrt(2.0)
    sx = np.sqrt(1.0 - x**2)
    for i in range(1, m+1):
        mat[:,0] = -np.sqrt((2.0*i + 1.0)/(2.0*i))*sx*mat[:,0]

    return mat

def plm(maxl, m, x):
    """Compute the normalized associated legendre polynomial projection matrix"""
    #get's you the evaluation of the Plm from m to Lmax 
    #x = txgrid(maxl, m) #gauss-legendre grid 
    mat = np.zeros((len(x), maxl - m + 1)) #
    #mat[:,0] = pmm(maxl, m)[:,0]
    mat[:,0] = pmm(maxl, m, x)[:,0]
    mat[:,1] = np.sqrt(2.0*m + 3.0)*x*mat[:,0]
    for i, l in enumerate(range(m+2, maxl + 1)):
        mat[:,i+2] = np.sqrt((2.0*l + 1)/(l - m))*np.sqrt((2.0*l - 1.0)/(l + m))*x*mat[:,i+1] - np.sqrt((2.0*l + 1)/(2.0*l - 3.0))*np.sqrt((l + m - 1.0)/(l + m))*np.sqrt((l - m - 1.0)/(l - m))*mat[:,i]

    return mat

def legSchmidtM(l, m, x):
    #p_mat=qsh.plm(10, m)
    p_mat=plm(l+1, m, x)
    
    if m==0:
        norm=sqrt(2./(2.*l+1.))
    else:
        norm=sqrt(4./(2.*l+1.))
    
    #return SchmidtNorm*lpmv(m,l,x)
    return norm*p_mat[:,l-m]

        
def diffLegM(l, m, x):
    #p_mat=qsh.plm(10, m)
    p_mat=plm(l+3, m, x)

    if m==0:
        norm=sqrt(2./(2.*l+1.))
        normLp1=sqrt(2./(2.*l+3.))
    else:
        norm=sqrt(4./(2.*l+1.))
        #normLp1=sqrt(4./(2.*l+3.))
        normLp1=sqrt(4./(2.*l+3.))*sqrt((l+m+1.0)/(l-m+1.0))
        
    #return SchmidtNorm*((1.0+l-m)*lpmv(m, l+1, x) - (l+1.0)*x*lpmv(m, l, x)) #Fix it !!! 
    #return norm*((1.0+l-m)*p_mat[:,l+1-m] - (l+1.0)*x*p_mat[:,l-m])
    return (normLp1*(1.0+l-m)*p_mat[:,l+1-m] - norm*(l+1.0)*x*p_mat[:,l-m])
    #return (1.0+l-m)*legSchmidtM(l, m, x) - (l+1.0)*x*legSchmidtM(l, m, x)


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

def W(n, l, r):
    """
    returns the Worland polynomial divided by the norm
    P_n=(r**l)*eval_jacobi(n, alpha, beta, 2*r*r-1)
    """
    #Compute Worland norm (the function returns 1/norm) =
    norm=float(wb.worland_norm(n,l))

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

    norm=float(wb.worland_norm(n,l))
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
    
    norm=float(wb.worland_norm(n,l))
    
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

#Worland polynomial from recursive Jacobi
def WorPoly(l, NT, r):
    Nrr = len(r)
    #print Nrr
    #JaP = np.zeros((1+NT, Nrr))
    JaP = JacobiPoly(NT, -0.5, l-0.5, 2*r**2-1.0)
    #print(JaP.shape)
    
    JaP_1 = np.zeros_like(JaP)
    JaP_1[1:,:] = JacobiPoly(NT-1, 0.5, l+0.5, 2*r**2-1.0)
    
    JaP_2 = np.zeros_like(JaP)
    JaP_2[2:,:] = JacobiPoly(NT-2, 1.5, l+1.5, 2*r**2-1.0)

    nl = np.arange(0,NT+1) + l
    nl = nl[np.newaxis,:].T #transposing 1D array
    
    dJaP = 2*(nl*r)*JaP_1
    
    nlnl = nl*(nl+1)
    
    #d2JaP = #2*bsxfun(@times, JaP_1, nl) + 4*(nlnl*(r*r))*JaP_2
    d2JaP = 2*np.multiply(JaP_1, nl) + 4*np.multiply(nlnl*(r**2),JaP_2)
    #print d2Jap.shape
    
    #print Wor.shape
    Wor = np.multiply(JaP, r**l)
    
    #print hn
    #hn = np.zeros(Nrr)
    #import pdb; pdb.set_trace()
    hn = np.sum(Wor[:, 1:Nrr-1]**2,1) * np.pi/(2*(Nrr-2))
    hn = np.sqrt(hn)
    
    #Wor = np.zeros(Nrr)
    #Wor1 = np.zeros(Nrr)
    Wor1 = np.multiply(JaP, r**(l-1)) #W_n/r
    #dWor = np.zeros(Nrr)
    dWor = l*Wor1 + np.multiply(dJaP, r**l) #dW/dr
    #d2Wor_Vor = np.zeros(Nrr)
    d2Wor_Vor = 2.0*(l+1.0)*np.multiply(dJaP, r**(l-1.0)) + np.multiply(d2JaP, r**l) #wrong! times 1000 
    
    
    if l == 0:
        Wor1[:,0] =0.0
        dWor[:,0] = 0.0
        d2Wor_Vor[:,0] = 0.0
    
    return (Wor, Wor1, dWor, d2Wor_Vor, hn)

#@profile
def computeVorticityCoeffDep(state):
    """
    returns two matrices to evaluate the z-component of vorticity 
    from the toroidal and poloidal components of the velocity
    vortZTor, vortZPol = computeVorticityCoeff(state) 
    the parameters (L, M, N, E) are obtained form a hdf5 state file 
    """
    #Artifitial grid setup
    #Getting grid setup from real rotated state 
    #state = '/Volumes/BigData/Simulations/daint_new/Leo-Precession/cscs_files/state0233.hdf5'
    #state = '/Volumes/BigData/Simulations/daint_new/Leo-Precession/cscs_files/Rotstate0233.hdf5'
    
    #read data to get truncation and Ekman number
    #TODO: remove +1 
    f = h5py.File(state, 'r') 
    nL=int(np.array(f['/Truncation/L'], dtype='double')+1)
    nM=int(np.array(f['/Truncation/M'], dtype='double')+1)
    nN=int(np.array(f['/Truncation/N'], dtype='double')+1)
    E=f['PhysicalParameters/E'][()]
    f.close()

    #Setup grid: 
    #Gauss-Legendre cuadrature in Z 
    #2*n+1 = max_degree : find degree of polynomial in Z 
    #N_z = 2 #acccuracy O(10-15)
    N_z = 40

    #Grid flexible in S
    nNs = 40

    nMmax = 10
    grid_s=np.linspace(0,1,N_s)

    #Remove the ekman layer 
    #ekmanRemover=1.-10.*sqrt(E)
    ekmanR=1.-10.*sqrt(E)
    grid_s=np.linspace(0, ekmanR, N_s)

    #Jacobi polynomial roots and weights 
    #Legendre alpha=0, beta=0
    alpha = 0
    beta = 0
    roots, w = j_roots(N_z, alpha, beta)

    grid_z=np.zeros((N_s, N_z))
    grid_cost=np.zeros_like(grid_z)
    grid_r=np.zeros_like(grid_z)

    for id_s, s in enumerate(grid_s):
        z_local=roots*sqrt(1.0-s*s) #from -1 to 1 
        grid_cost[id_s,:]=sz2ct(s, z_local) #from -1 to 1 
        grid_r[id_s,:]=sz2r(s, z_local)
        grid_z[id_s,:]=z_local
        #print(id_s)
        #print(grid_z[id_s,:])
        #print(grid_cost[id_s,:])
        #print(grid_r[id_s,:])

    #Allocating memory 
    vortz_tor=np.zeros((nN, nL, nM, nNs, N_z), dtype=complex)
    vortz_pol=np.zeros_like(vortz_tor)
    
    #Computing physical grid to evaluate vertical vorticity from spectral coefficients
    #%debug
    #%%prun -s cumulative
    #plt.figure(figsize=(12,6)) #for debugging
    #s = np.ones(N_z)  #for debugging
    for n in range(0, nN): #n=0:nN-1
    #for n in xrange(0, 1): #debug
        print('n=', n)
        for l in range(0, nL): #l=0:nL-1
        #for l in xrange(3,4): #debug
            #TODO: Leo fix this limit
            for m in range(0, min(l,nMmax-1)+1): #m=0:min(l, nMmax-1)
            #for m in xrange(2, 3): #debug

                #print('m=', m)
                #for id_s in xrange(4, 5): #debug
                for id_s in range(0, N_s):
                    r_local=grid_r[id_s,:]
                    cos_local=grid_cost[id_s,:]

                    #polynomials for reconstruction
                    #plm_w1=W1(n, l)
                    #plm_lapw1=laplacianW1(n, l)
                    #plm_diffw1=diffW1(n, l)
                    
                    plm_w1=W(n, l, r_local)
                    plm_lapw1=laplacianW(n, l, r_local)
                    plm_diffw1 = diffW(n, l, r_local)


                    #plm_leg=LegSchmidt(l, m, cos_local)
                    #plm_diffLeg=diffLeg(l, m, cos_local)
                    plm_leg=legSchmidtM(l, m, cos_local)
                    plm_diffLeg=diffLegM(l, m, cos_local)


                    #Plots for debugging!!!!
                    #plt.subplot(141)
                    #plt.plot(s*grid_s[id_s], grid_z[id_s,:],'o')

                    #plt.plot(grid_z[id_s,:],'o')
                    #plt.plot(r_local,'o')
                    #plt.plot(r_local,'-')

                    #print(r_local)
                    #print(plm_lapw1(r_local))

                    #VortZ_tor from theta component
                    #plt.subplot(132)
                    #plt.plot(s*grid_s[id_s], m*plm_lapw1(r_local) * plm_leg,'o')

                    #vortZ_pol from theta component
                    #plt.subplot(142)
                    #print plm_w1
                    #plt.plot(r_local, plm_w1(r_local))
                    #plt.plot(l * (l+1.0) * plm_w1(r_local) * plm_leg * cos_local / r_local, grid_z[id_s,:], 'o')

                    #plt.subplot(133)
                    #vortZ_tor from theta component
                    #plt.plot(m*plm_lapw1(r_local) * plm_leg, grid_z[id_s,:],'o')

                    #vortZ_pol from r component  
                    #plt.subplot(143)
                    #plt.plot(- plm_diffw1(r_local) * plm_diffLeg / r_local, grid_z[id_s,:], 'o')

                    #VortZ_pol r and theta 
                    #plt.subplot(144)
                    #plt.plot(l * (l+1.0) * plm_w1(r_local) * plm_leg * cos_local / r_local
                    #                            - plm_diffw1(r_local) * plm_diffLeg / r_local, grid_z[id_s,:],'o')

                    #computing contributions from toroidal and poloidal vorticity onto Z 
                    #vortz_tor[n, l, m, id_s, :] = 1j * m * plm_lapw1(r_local) * plm_leg #times P
                    #vortz_pol[n, l, m, id_s, :] = l * (l+1.0) * plm_w1(r_local) * plm_leg * cos_local / r_local \
                    #                            - plm_diffw1(r_local) * plm_diffLeg / r_local #times
                    vortz_tor[n, l, m, id_s, :] = 1j * m * plm_lapw1 * plm_leg #times P 

                    vortz_pol[n, l, m, id_s, :] = l * (l+1.0) * plm_w1 * plm_leg * cos_local / r_local \
                                            - plm_diffw1 * plm_diffLeg / r_local #times T
                    
                    #ur_pol[n, l, m, id_s, :] = l * (l+1.0) * plm_w1 * plm_leg / r_local #1/r * L2(P) 
                    #uphi_tor[n, l, m, id_s, :] = - plm_diffLeg #-\partial_\theta T 
                    
                    #uphi_pol[n, l, m, id_s, :] = 1j * m * plm_diffw1/(r_local * sin_local)
                    
                    #if(dataT[n,l,m]>0):
                    #    print('vort_pol:',l * (l+1.0) * plm_w1(r_local) * plm_leg * cos_local / r_local - plm_diffw1(r_local) * plm_diffLeg / r_local)
                    
    return (w, grid_s, vortz_tor, vortz_pol)

#Leo: not used
def computeZintegralDep(dataT, dataP, vortz_tor, vortz_pol, nL, M, nNs, w, grid_s):
    """
    returns the integral of vertical vorticity up to 
    a max spherical harmonic order M 
    and Worland Polynomial (Ns)
    vort_int=computeZintegral(dataP, dataT, vortz_tor, vortz_pol, M, Ns)
    vort_int size is M x Ns 
    """ 
    #Allocating memory for output
    vort_int = np.zeros((M, N_s), dtype=complex)
    #Ur_int = np.zeros((M, N_s), dtype=complex)
    #Uphi_int = np.zeros((M, N_s), dtype=complex)

    #%%debug 
    #%%prun -s cumulative
    #M = 3 
    #main idea compute for each m the integral 
    #Int(f(s,z)dz) = sum(weight*vort_z(s,z)) = f(s)
    #vort_z function that returns the component of the z-vorticity
    #for m = 0:N
    # for s = s0:sN
    #  for n=0:N 
    #   for l=0:L
    #    vort_int[idm,id_s] = sum(f(s,z))
    #for n in xrange(0, N-1): #Leo: N-1 check boundaries ???
    for n in range(0, 10): #Leo: N-1 check boundaries ???    #print('n:', n)
        ind=0
        for l in range(0, nL): #l=0:L-1
            #print('l:',l)
            for m in range(0, l+1): #m=0:l
                #print('m:',m)
                for id_s in range(0, nNs): #id_s=0:N_s
                    #print('ids:', id_s)
                    #Store Int(vort_z d z) as a functio of S
                    #vort_int[m, id_s] = quad_vort_z(w, grid_s[id_s], grid_z[id_s,:], n, l, m, dataT[n,l,m])

                    #For artificial data:
                    #vort_z = vortz_tor[n, l, m, id_s, :] * dataP[n, l, m] + vortz_pol[n, l, m, id_s, :] * dataT[n, l, m]

                    #For real data:
                    vort_z = vortz_tor[n, l, m, id_s, :] * dataP[ind, n] + vortz_pol[n, l, m, id_s, :] * dataT[ind, n]
                    #u_r = ur_pol[n, l, m, id_s, :] * dataP[ind, n]
                    #u_phi = uphi_tor[n, l, m, id_s, :] * dataT[ind, n] + uphi_pol[n, l, m, id_s, :] * dataP[ind, n]
                    
                    #don't forget to include the factor sqrt(1-s^2) to compute the integral 

                    #For artificial data:
                    #vort_int[m, id_s] = vort_int[m, id_s] + sum(w*vort_z)*sqrt(1-grid_s[id_s]**2)

                    #For real data:
                    z_e = sqrt((1.0-d)*(1.0-d) - grid_s[id_s]*grid_s[id_s])
                    #vort_int[m, id_s] = vort_int[m, id_s] + sum(w*vort_z)*sqrt(1-grid_s[id_s]**2)
                    vort_int[m, id_s] = vort_int[m, id_s] + sum(w*vort_z)*z_e #sqrt(1-grid_s[id_s]**2)*(1.0-d) #Leo: check this
                    #Ur_int[m, id_s] = Ur_int[m, id_s] + sum(w*u_r)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #Uphi_int[m, id_s] = Uphi_int[m, id_s] + sum(w*u_phi)*sqrt(1-grid_s[id_s]**2)*(1.0-d)

                    #if dataT[n, l, m]>0:
                    #    print(n,l,m)
                    #    print('vortz_pol:',vortz_pol[m, id_s, n, l, :])
                        #print('vort_z:',vort_z)


                ind = ind + 1

    #Release matrix and repeat
    #Modify: generate grid-matrix with evaluations of W(r)*Ylm(theta,phi) returns W_Ylm[m,l,n,n_s,n_z] 
    #Compute Integral
    #for m = 0:M
    # for l = 0:L


    #  for n = 0:N
    #   for s = 0:n_s
    #    for z = 0:n_z
    #     w[z]*coeff[n,l,m]*W_Ylm[m, l, n,n_s,n_z]    
    return vort_int #, Ur_int, Uphi_int

def computeGeostrophicCoeff(state, res):
    """
    returns weights, grid and two arrays to evaluate the z-integral of vorticity and the flow 
    from the toroidal and poloidal components of the velocity
    vortZTor, vortZPol = computeVorticityCoeff(state) 
    the parameters (L, M, N, E) are obtained form a hdf5 state file 
    """
    nNmax, nLmax, nMmax, nNs = res    

    f = h5py.File(state, 'r') #read data
    #TODO: remove +1
    nL=int(np.array(f['/Truncation/L'], dtype='double')+1)
    nM=int(np.array(f['/Truncation/M'], dtype='double')+1)
    nN=int(np.array(f['/Truncation/N'], dtype='double')+1)
    E=f['PhysicalParameters/E'][()]
    f.close()

    #Setup grid: 
    #Gauss-Legendre cuadrature in Z 
    #2*Nz+1 = max_degree : find degree of polynomial in Z 
    #N_z = 2 #acccuracy O(10-15)
    #degZ = 2N+l-m = 2N+L
    #2*Nz + 1 = 2N + L
    #Nz = N + L/2
    #N_z = 40
    N_z = int(nNmax + nLmax/2) #function of Nmax and Lmax

    #Jacobi polynomial roots and weights 
    #Legendre alpha=0, beta=0
    alpha = 0
    beta = 0
    roots, w = j_roots(N_z, alpha, beta)

    #Remove the ekman layer 
    #ekmanRemover=1.-10.*sqrt(E)
    d = 10.*sqrt(E)
    ekmanR=1. - d

    #Grid flexible in S
    #grid_s=np.linspace(0,1,N_s)
    grid_s=np.linspace(0,ekmanR,N_s+1)
    #grid_s=grid_s[1:]
    #avoiding the origin taking cells' centers
    grid_s=0.5*(grid_s[0:-1]+grid_s[1:])

    grid_z=np.zeros((N_s, N_z))
    grid_cost=np.zeros_like(grid_z)
    grid_sint=np.zeros_like(grid_z)
    grid_r=np.zeros_like(grid_z)

    #Generating (s,z) grids
    for id_s, s in enumerate(grid_s):
        #Quadrature points 
        #z_local=roots*sqrt(1.0-s*s)*(1-d) #from -1 to 1
        z_local=roots*sqrt((1.0-d)*(1.0-d) - s*s) #from -1 to 1
        grid_cost[id_s,:]=sz2ct(s, z_local) #from -1 to 1
        grid_sint[id_s,:]=sz2st(s, z_local)
        grid_r[id_s,:]=sz2r(s, z_local)
        grid_z[id_s,:]=z_local
    
    #Leo:
    #nLmax*(nLmax+1)/2 : large space than needed 
    vortz_tor=np.zeros((nNs, nNmax, int(nLmax*(nLmax+1)/2), N_z), dtype=complex)
    vortz_pol=np.zeros_like(vortz_tor)
    ur_pol=np.zeros_like(vortz_tor)
    uphi_tor=np.zeros_like(vortz_tor)
    uphi_pol=np.zeros_like(vortz_tor)    
    uth_tor=np.zeros_like(vortz_tor)
    uth_pol=np.zeros_like(vortz_tor)
    us_tor=np.zeros_like(vortz_tor)
    us_pol=np.zeros_like(vortz_tor)
    uz_tor=np.zeros_like(vortz_tor)
    uz_pol=np.zeros_like(vortz_tor)
    
    for id_s in range(0, nNs): #id_s=0:nNs-1
        print('id_s=', id_s)
        for n in range(0, nNmax): #debug n=0:nNmax-1
            #print('n=', n)
            ind = 0
            for l in range(0, nLmax): #l=0:Lmax-1
                #TODO: Leo add exception to verify size is correct 
                for m in range(0, min(l, nMmax-1)+1): #m = 0:min(l, nMmax-1)
                #for m in range(0, min(l, Mmax)+1):
                    #pdb.set_trace()
                    r_local=grid_r[id_s,:]
                    cos_local=grid_cost[id_s,:]
                    sin_local=grid_sint[id_s,:]

                    #polynomials for reconstruction
                    plm_w1=W(n, l, r_local)                
                    plm_lapw1=laplacianW(n, l, r_local)                
                    plm_diffw1 = diffW(n, l, r_local)                
                    plm_leg=legSchmidtM(l, m, cos_local)
                    plm_diffLeg=diffLegM(l, m, cos_local)
                                
                    #computing contributions from toroidal and poloidal vorticity onto Z 
                    vortz_tor[id_s, n, ind, :] = 1j * m * plm_lapw1 * plm_leg #times P 
                
                    vortz_pol[id_s, n, ind, :] = l * (l+1.0) * plm_w1 * plm_leg * cos_local / r_local \
                                            - plm_diffw1 * plm_diffLeg / r_local #times P

                    uth_tor[id_s, n, ind, :] = 1j* m * plm_w1* plm_leg /sin_local
                    uphi_tor[id_s, n, ind, :] = -plm_w1*plm_diffLeg / sin_local #-\partial_\theta T 

                    ur_pol[id_s, n, ind, :] = l * (l+1.0) * plm_w1 * plm_leg / r_local #1/r * L2(P) 
                    uth_pol[id_s, n, ind, :] = plm_diffw1*plm_diffLeg/ (r_local*sin_local) #???
                    #1/(r sin(theta)) \partial_r * r * \partial_\phi P
                    uphi_pol[id_s, n, ind, :] = 1j * m * plm_diffw1 * plm_leg/(r_local * sin_local)

                    #u_s = u_r * sin(theta) + u_theta * cos(theta)
                    us_pol[id_s, n, ind, :] = l * (l+1.0) * plm_w1 * plm_leg * sin_local / r_local + plm_diffw1*plm_diffLeg * cos_local/ (r_local*sin_local)
                    us_tor[id_s, n, ind, :] = 1j* m * plm_w1 * plm_leg * cos_local/sin_local 

                    #u_z = u_r * cos(theta) - u_theta * sin(theta)
                    uz_pol[id_s, n, ind, :] = l * (l+1.0) * plm_w1 * plm_leg * cos_local / r_local - plm_diffw1*plm_diffLeg/ r_local
                    uz_tor[id_s, n, ind, :] = 1j* m * plm_w1 * plm_leg

                    ind = ind + 1
                    
    res = (nNmax, nLmax, nMmax, nNs)
    zIntegrator = Integrator(w, grid_s, vortz_tor, vortz_pol, us_tor, us_pol, uphi_tor, uphi_pol, res)
    #vortz_tor, vortz_pol, us_tor, us_pol, uphi_tor, uphi_pol)
    
    return zIntegrator


def computeGeostrophicCoeffpybind(state, res):
    """
    Accelerated version
    returns weights, grid and two arrays to evaluate the z-integral of vorticity and the flow
    from the toroidal and poloidal components of the velocity
    vortZTor, vortZPol = computeVorticityCoeffC(state)
    the parameters (L, M, N, E) are obtained form a hdf5 state file
    """
    nNmax, nLmax, nMmax, nNs = res

    f = h5py.File(state, 'r')  # read data
    #TODO: remove +1
    nL = int(np.array(f['/Truncation/L'], dtype='double') + 1)
    nM = int(np.array(f['/Truncation/M'], dtype='double') + 1)
    nN = int(np.array(f['/Truncation/N'], dtype='double') + 1)
    E = f['PhysicalParameters/E'][()]
    f.close()

    # Setup grid:
    # Gauss-Legendre cuadrature in Z
    # 2*Nz+1 = max_degree : find degree of polynomial in Z
    # N_z = 2 #acccuracy O(10-15)
    # degZ = 2N+l-m = 2N+L
    # 2*Nz + 1 = 2N + L
    # Nz = N + L/2
    # N_z = 40
    N_z = int(nNmax + nLmax / 2) # function of Nmax and Lmax

    # Jacobi polynomial roots and weights
    # Legendre alpha=0, beta=0
    alpha = 0
    beta = 0
    roots, w = j_roots(N_z, alpha, beta)

    # Remove the ekman layer
    # ekmanRemover=1.-10.*sqrt(E)
    d = 10. * sqrt(E)
    ekmanR = 1. - d

    # Grid flexible in S
    # grid_s=np.linspace(0,1,N_s)
    grid_s = np.linspace(0, ekmanR, nNs + 1)
    # grid_s=grid_s[1:]
    # avoiding the origin taking cells' centers
    grid_s = 0.5 * (grid_s[0:-1] + grid_s[1:])

    grid_z = np.zeros((nNs, N_z))
    grid_cost = np.zeros_like(grid_z)
    grid_sint = np.zeros_like(grid_z)
    grid_r = np.zeros_like(grid_z)

    # Generating (s,z) grids
    for id_s, s in enumerate(grid_s):
        # Quadrature points
        # z_local=roots*sqrt(1.0-s*s)*(1-d) #from -1 to 1
        z_local = roots * sqrt((1.0 - d) * (1.0 - d) - s * s)  # from -1 to 1
        grid_cost[id_s, :] = sz2ct(s, z_local)  # from -1 to 1
        grid_sint[id_s, :] = sz2st(s, z_local)
        grid_r[id_s, :] = sz2r(s, z_local)
        grid_z[id_s, :] = z_local
    
    #Leo:
    #nLmax*(nLmax+1)/2 : large space than needed 
    vortz_tor = np.zeros((nNs, nNmax, int(nLmax * (nLmax + 1) / 2), N_z), dtype=complex)
    vortz_pol = np.zeros_like(vortz_tor)
    ur_pol = np.zeros_like(vortz_tor)
    uphi_tor = np.zeros_like(vortz_tor)
    uphi_pol = np.zeros_like(vortz_tor)
    uth_tor = np.zeros_like(vortz_tor)
    uth_pol = np.zeros_like(vortz_tor)
    us_tor = np.zeros_like(vortz_tor)
    us_pol = np.zeros_like(vortz_tor)
    uz_tor = np.zeros_like(vortz_tor)
    uz_pol = np.zeros_like(vortz_tor)

    opW = np.empty((N_z, nNmax), dtype='float64', order='F')
    opLaplW = np.empty((N_z, nNmax), dtype='float64', order='F')
    opDiffW = np.empty((N_z, nNmax), dtype='float64', order='F')

    for id_s in range(0, nNs):
        print('id_s=', id_s)
        r_local = grid_r[id_s, :]
        cos_local = grid_cost[id_s, :]
        sin_local = grid_sint[id_s, :]

        ind = 0
        for l in range(0, nLmax): #l=0,nLmax-1
            pybind.wnl(opW, l, r_local)
            pybind.slaplwnl(opLaplW, l, r_local)
            pybind.drwnl(opDiffW, l, r_local)
            #TODO: Leo: add exception to verify size 
            for m in range(0, min(l, nMmax-1) + 1): #m=0,min(l,Mmax-1)
            #for m in range(0, min(l, Mmax) + 1):
                # pdb.set_trace()
                plm_leg = legSchmidtM(l, m, cos_local)
                plm_diffLeg = diffLegM(l, m, cos_local)

                for n in range(0, nNmax):  # debug #n=0,nNmax-1
                    # polynomials for reconstruction
                    # plm_w1 = W(n, l, r_local)
                    plm_w1 = opW[:,n]
                    # plm_lapw1 = laplacianW(n, l, r_local)
                    plm_lapw1 = opLaplW[:,n]
                    # plm_diffw1 = diffW(n, l, r_local)
                    plm_diffw1 = opDiffW[:,n]

                    # print('n=', n)
                    # computing contributions from toroidal and poloidal vorticity onto Z
                    vortz_tor[id_s, n, ind, :] = 1j * m * plm_lapw1 * plm_leg  # times P

                    vortz_pol[id_s, n, ind, :] = l * (l + 1.0) * plm_w1 * plm_leg * cos_local / r_local \
                                                 - plm_diffw1 * plm_diffLeg / r_local  # times P

                    uth_tor[id_s, n, ind, :] = 1j * m * plm_w1 * plm_leg / sin_local
                    uphi_tor[id_s, n, ind, :] = -plm_w1 * plm_diffLeg / sin_local  # -\partial_\theta T

                    ur_pol[id_s, n, ind, :] = l * (l + 1.0) * plm_w1 * plm_leg / r_local  # 1/r * L2(P)
                    uth_pol[id_s, n, ind, :] = plm_diffw1 * plm_diffLeg / (r_local * sin_local)  # ???
                    # 1/(r sin(theta)) \partial_r * r * \partial_\phi P
                    uphi_pol[id_s, n, ind, :] = 1j * m * plm_diffw1 * plm_leg / (r_local * sin_local)

                    # u_s = u_r * sin(theta) + u_theta * cos(theta)
                    us_pol[id_s, n, ind, :] = l * (
                                l + 1.0) * plm_w1 * plm_leg * sin_local / r_local + plm_diffw1 * plm_diffLeg * cos_local / (
                                                          r_local * sin_local)
                    us_tor[id_s, n, ind, :] = 1j * m * plm_w1 * plm_leg * cos_local / sin_local

                    # u_z = u_r * cos(theta) - u_theta * sin(theta)
                    uz_pol[id_s, n, ind, :] = l * (
                                l + 1.0) * plm_w1 * plm_leg * cos_local / r_local - plm_diffw1 * plm_diffLeg / r_local
                    uz_tor[id_s, n, ind, :] = 1j * m * plm_w1 * plm_leg

                ind = ind + 1

    res = (nNmax, nLmax, nMmax, nNs)
    zIntegrator = Integrator(w, grid_s, vortz_tor, vortz_pol, us_tor, us_pol, uphi_tor, uphi_pol, res)
    # vortz_tor, vortz_pol, us_tor, us_pol, uphi_tor, uphi_pol)

    return zIntegrator

#Function to index data
#def idxlmWrong(L, M):
#    #idxlm = np.zeros((L+1, M+1), dtype = int)
#    idxlm = {}
#    ind = 0
#    #Leo: l = 0:L-1
#    for l in range(0, L):
#        for m in range(0, min(l, M)+1): #Leo: m = 0:min(L-1, M) + 1
#        #Leo: check this could be  
#        #for m in range(0, min(l+1, M)):
#            idxlm[(l,m)] = ind
#            ind = ind + 1
#    
#    return idxlm

# TODO: Leo: add exception to check right indexing!!
# make sure to use the right 
# indexing Lmax and Mmax fixing indexing 
def idxlm(nL, nM):
    idxlm = {}
    ind = 0
    for l in range(0, nL):
        for m in range(0, min(l, nM-1)+1):
            idxlm[(l,m)] = ind
            ind = ind + 1
    
    #if ind != idxlm.size():
    #    raise RuntimeError("idxlm size differs from max content")

    return idxlm

def computeZintegral(state, scale, zInt, rotator):
    """
    returns the z-integral of vorticity and the flow up to 
    a max spherical harmonic order Mmax and Worland Polynomial (Nmax)
    vort_int=computeZintegral(state, vortz_tor, vortz_pol, ur_pol, uphi_tor, uphi_pol, (nNmax, nLmax, nMmax, nNs), w, grid_s)
    """ 
    f = h5py.File(state, 'r')
    #TODO: remove +1
    nL=int(np.array(f['/Truncation/L'], dtype='double')+1)
    nM=int(np.array(f['/Truncation/M'], dtype='double')+1)
    nN=int(np.array(f['/Truncation/N'], dtype='double')+1)
    dataP=np.array(f['Velocity/VelocityPol'][:])
    dataT=np.array(f['Velocity/VelocityTor'][:])
    E=f['PhysicalParameters/E'][()]
    f.close()
    
    nNmax, nLmax, nMmax, nNs = zInt.res
    w = zInt.w
    grid_s = zInt.grid_s
    
    #ekmanRemover=1.-10.*sqrt(E)
    d = 10.*sqrt(E)
    
    idx = idxlm(nL, nM)
    idxM = idxlm(nLmax, nMmax)
    
    dataT, dataP, dataC = rot.rotateState(state, scale, rotator)
    
    if scale == 'viscous':
        dataT = dataT*E
        dataP = dataP*E
        dataC = dataC*E 

    #Transform into complex data
    dataT=dataT[:,:,0]+dataT[:,:,1]*1j
    dataP=dataP[:,:,0]+dataP[:,:,1]*1j
    
    #Allocating memory for output
    vort_int = np.zeros((nMmax, mnNs), dtype=complex)
    Ur_int = np.zeros((nMmax, nNs), dtype=complex)
    Uth_int = np.zeros((nMmax, nNs), dtype=complex)
    Uphi_int = np.zeros((nMmax, nNs), dtype=complex)
    Us_int = np.zeros((nMmax, nNs), dtype=complex)
    Uz_int = np.zeros((nMmax, nNs), dtype=complex)
    H_int = np.zeros((nMmax, nNs), dtype=complex)

    #Make sure dimensions agree 
    assert(dataT.shape[0]==len(idx))
    assert(dataP.shape[0]==len(idx))
    assert(zInt.vortz_tor.shape[2] == len(idxM))
    assert(zInt.vortz_pol.shape[2] == len(idxM))
    assert(zInt.uphi_tor.shape[2] == len(idxM))
    assert(zInt.uphi_pol.shape[2] == len(idxM))
    assert(zInt.us_pol.shape[2] == len(idxM))
    assert(zInt.us_pol.shape[2] == len(idxM))

    #main idea compute for each m the integral 
    #Int(f(s,z)dz) = sum(weight*vort_z(s,z)) = f(s)
    #vort_z function that returns the component of the z-vorticity
    for id_s in range(0, nNs):
        for n in range(0, nNmax): 
            for l in range(0, nLmax):
                #TODO: add exception to verify size is correct
                #for m in range(0, min(l, Mmax)+1):
                for m in range(0, min(l, nMmax-1)+1):
                    #For real data:
                    vort_z = zInt.vortz_tor[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n] + zInt.vortz_pol[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n]
                    #u_r = zInt.ur_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    #u_th = zInt.uth_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.uth_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    u_phi = zInt.uphi_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.uphi_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    u_s = zInt.us_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.us_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    #u_z = zInt.uz_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.uz_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    
                    #don't forget to include the factor sqrt(1-s^2) to compute the integral 
                    #For real data:
                    #Integrate
                    z_e = sqrt((1.0-d)*(1.0-d) - grid_s[id_s]*grid_s[id_s])
                    vort_int[m, id_s] = vort_int[m, id_s] + sum(w*vort_z)*z_e #wrong: sqrt(1-grid_s[id_s]**2)*(1.0-d) #Leo: check this
                    #Ur_int[m, id_s] = Ur_int[m, id_s] + sum(w*u_r)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #Uth_int[m, id_s] = Uth_int[m, id_s] + sum(w*u_th)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    Uphi_int[m, id_s] = Uphi_int[m, id_s] + sum(w*u_phi)*z_e #wrong: sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    Us_int[m, id_s] = Us_int[m, id_s] + sum(w*u_s)*z_e #wrong: sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #Uz_int[m, id_s] = Uz_int[m, id_s] + sum(w*u_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #H_int[m, id_s] = H_int[m, id_s] + sum(w*u_z*vort_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
     
    
    return vort_int, Us_int, Uphi_int

def computeZintegral(state, scale, zInt):
    """
    returns the z-integral of vorticity and the flow up to 
    a max spherical harmonic order Mmax and Worland Polynomial (Nmax)
    vort_int=computeZintegral(state, scale = 'rotation', zInt=Integrator)
    """ 
    f = h5py.File(state, 'r')
    #TODO: remove +1
    nL=int(np.array(f['/Truncation/L'], dtype='double')+1)
    nM=int(np.array(f['/Truncation/M'], dtype='double')+1)
    nN=int(np.array(f['/Truncation/N'], dtype='double')+1)
    dataP=np.array(f['Velocity/VelocityPol'][:])
    dataT=np.array(f['Velocity/VelocityTor'][:])
    E=f['PhysicalParameters/E'][()]
    f.close()
    
    nNmax, nLmax, nMmax, nNs = zInt.res
    w = zInt.w
    grid_s = zInt.grid_s
    
    #ekmanRemover=1.-10.*sqrt(E)
    d = 10.*sqrt(E)
    
    idx = idxlm(nL, nM)
    idxM = idxlm(nLmax, nMmax)
    
    #dataT, dataP, dataC = rot.rotateState(state, scale, rotator)
    
    if scale == 'viscous':
        dataT = dataT*E
        dataP = dataP*E
        #dataC = dataC*E 

    #Transform into complex data
    dataT=dataT[:,:,0]+dataT[:,:,1]*1j
    dataP=dataP[:,:,0]+dataP[:,:,1]*1j
    
    #Allocating memory for output
    vort_int = np.zeros((nMmax, nNs), dtype=complex)
    Ur_int = np.zeros((nMmax, nNs), dtype=complex)
    Uth_int = np.zeros((nMmax, nNs), dtype=complex)
    Uphi_int = np.zeros((nMmax, nNs), dtype=complex)
    Us_int = np.zeros((nMmax, nNs), dtype=complex)
    Uz_int = np.zeros((nMmax, nNs), dtype=complex)
    H_int = np.zeros((nMmax, nNs), dtype=complex)

    #Make sure dimensions agree 
    assert(dataT.shape[0]==len(idx))
    assert(dataP.shape[0]==len(idx))
    assert(zInt.vortz_tor.shape[2] == len(idxM))
    assert(zInt.vortz_pol.shape[2] == len(idxM))
    assert(zInt.uphi_tor.shape[2] == len(idxM))
    assert(zInt.uphi_pol.shape[2] == len(idxM))
    assert(zInt.us_pol.shape[2] == len(idxM))
    assert(zInt.us_pol.shape[2] == len(idxM))

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
                    vort_z = zInt.vortz_tor[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n] + zInt.vortz_pol[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n]
                    #u_r = zInt.ur_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    #u_th = zInt.uth_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.uth_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    u_phi = zInt.uphi_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.uphi_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    u_s = zInt.us_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.us_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    #u_z = zInt.uz_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.uz_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    
                    #don't forget to include the factor sqrt(1-s^2) to compute the integral 
                    #For real data:
                    #Integrate
                    vort_int[m, id_s] = vort_int[m, id_s] + sum(w*vort_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d) #Leo: check this
                    #Ur_int[m, id_s] = Ur_int[m, id_s] + sum(w*u_r)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #Uth_int[m, id_s] = Uth_int[m, id_s] + sum(w*u_th)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    Uphi_int[m, id_s] = Uphi_int[m, id_s] + sum(w*u_phi)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    Us_int[m, id_s] = Us_int[m, id_s] + sum(w*u_s)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #Uz_int[m, id_s] = Uz_int[m, id_s] + sum(w*u_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #H_int[m, id_s] = H_int[m, id_s] + sum(w*u_z*vort_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
     
    
    return vort_int, Us_int, Uphi_int

def computeZintegralTest(state, scale, zInt):
    """
    returns the z-integral of vorticity and the flow up to 
    a max spherical harmonic order Mmax and Worland Polynomial (Nmax)
    vort_int=computeZintegral(state, vortz_tor, vortz_pol, ur_pol, uphi_tor, uphi_pol, (Nmax, Lmax, Mmax, N_s), w, grid_s)
    """ 
    f = h5py.File(state, 'r')
    #TODO: remove +1
    nL=int(np.array(f['/Truncation/L'], dtype='double')+1)
    nM=int(np.array(f['/Truncation/M'], dtype='double')+1)
    nN=int(np.array(f['/Truncation/N'], dtype='double')+1)
    dataP=np.array(f['Velocity/VelocityPol'][:])
    dataT=np.array(f['Velocity/VelocityTor'][:])
    E=f['PhysicalParameters/E'][()]
    f.close()
    
    nNmax, nLmax, nMmax, nNs = zInt.res
    w = zInt.w
    grid_s = zInt.grid_s
    
    #ekmanRemover=1.-10.*sqrt(E)
    d = 10.*sqrt(E)
    
    idx = idxlm(nL, nM)
    idxM = idxlm(nLmax, nMmax)
    
    #dataT, dataP, dataC = rot.rotateState(state, scale, rotator)
    
    if scale == 'viscous':
        dataT = dataT*E
        dataP = dataP*E
        #dataC = dataC*E 

    #Transform into complex data
    dataT=dataT[:,:,0]+dataT[:,:,1]*1j
    dataP=dataP[:,:,0]+dataP[:,:,1]*1j
    
    #Allocating memory for output
    vort_int = np.zeros((nMmax, nNs), dtype=complex)
    Ur_int = np.zeros((nMmax, nNs), dtype=complex)
    Uth_int = np.zeros((nMmax, nNs), dtype=complex)
    Uphi_int = np.zeros((nMmax, nNs), dtype=complex)
    Us_int = np.zeros((nMmax, nNs), dtype=complex)
    Uz_int = np.zeros((nMmax, nNs), dtype=complex)
    H_int = np.zeros((nMmax, nNs), dtype=complex)

    #Make sure dimensions agree 
    assert(dataT.shape[0]==len(idx))
    assert(dataP.shape[0]==len(idx))
    assert(zInt.vortz_tor.shape[2] == len(idxM))
    assert(zInt.vortz_pol.shape[2] == len(idxM))
    assert(zInt.uphi_tor.shape[2] == len(idxM))
    assert(zInt.uphi_pol.shape[2] == len(idxM))
    assert(zInt.us_pol.shape[2] == len(idxM))
    assert(zInt.us_pol.shape[2] == len(idxM))

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
                    vort_z = zInt.vortz_tor[id_s, n, idxM[l,m]] * dataP[idx[l,m], n] + zInt.vortz_pol[id_s, n, idxM[l,m]] * dataT[idx[l,m], n]
                    #u_r = zInt.ur_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    #u_th = zInt.uth_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.uth_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    u_phi = zInt.uphi_tor[id_s, n, idxM[l,m]] * dataT[idx[l,m], n] + zInt.uphi_pol[id_s, n, idxM[l,m]] * dataP[idx[l,m], n]
                    u_s = zInt.us_tor[id_s, n, idxM[l,m]] * dataT[idx[l,m], n] + zInt.us_pol[id_s, n, idxM[l,m]] * dataP[idx[l,m], n]
                    #u_z = zInt.uz_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] + zInt.uz_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                    
                    #don't forget to include the factor sqrt(1-s^2) to compute the integral 
                    #For real data:
                    #Integrate
                    vort_int[m, id_s] = vort_int[m, id_s] + vort_z #sum(w*vort_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d) #Leo: check this
                    Uphi_int[m, id_s] = Uphi_int[m, id_s] + u_phi #sum(w*u_phi)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    Us_int[m, id_s] = Us_int[m, id_s] + u_s #sum(w*u_s)*sqrt(1-grid_s[id_s]**2)*(1.0-d)

                    #Ur_int[m, id_s] = Ur_int[m, id_s] + sum(w*u_r)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #Uth_int[m, id_s] = Uth_int[m, id_s] + sum(w*u_th)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #Uz_int[m, id_s] = Uz_int[m, id_s] + sum(w*u_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
                    #H_int[m, id_s] = H_int[m, id_s] + sum(w*u_z*vort_z)*sqrt(1-grid_s[id_s]**2)*(1.0-d)
     
    
    return vort_int, Us_int, Uphi_int
