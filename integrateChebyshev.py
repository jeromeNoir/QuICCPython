"""
Functions used to perform a Z-integral using a Legendre Quadrature in a Spherical Shell 
Author: nicolo.lardelli@erdw.ethz.ch
"""
import sys
sys.path.append('/home/nicolol/workspace/QuICC/Python')
from quicc.projection import spherical, shell
import numpy as np
from numpy.polynomial.legendre import leggauss
import h5py 
from numpy import fft
                   
class Integrator():

    # constructor
    # res = (Ns, Nmax, Lmax, Mmax): tupla of integers
    def __init__(self, res=None):
        # usage:
        # if res is defined -> use for precomputation
        # if red in not defined -> use for loading .hdf5
        if res is not None:
            self.res = res
            
            pass
        pass

    # export the integrator to an hdf5 file
    def write(self, hdf5_filename):
        fout = h5py.File(hdf5_filename, 'w')

        # TODO: loop over all the attributes and store them in fout
        for attr in vars(self):
            if not attr.startswith('__'):
                fout.create_dataset(attr, data=getattr(self, attr))
                
        fout.close()
        pass

    # import the integrator from an hdf5 file
    def load(self, hdf5_filename):
        
        fin = h5py.File(hdf5_filename, 'r')

        for k in fin.keys():
            setattr(self, k, fin[k].value)
        fin.close()
        pass
        
    def generateGrid(self, state):
        # here state is needed to obtain the eta parameter    
    
        # import eta value and others
        # open hdf5 file
        fin = h5py.File(state, 'r')
        LL = fin['/truncation/spectral/dim2D'].value + 1
        MM = fin['/truncation/spectral/dim3D'].value + 1
        NN = fin['/truncation/spectral/dim1D'].value + 1
        self.file_res = (NN, LL, MM)
        E = fin['/physical/ekman'].value
        eta = fin['/physical/rratio'].value
        fin.close()
        
        # compute the diffeomorfism parameters between Tchebyshev and radius space
        self.a = a = .5
        self.b = b =.5*(1+eta)/(1-eta)
        
        # compute boundary layer
        d = 10.*E**.5
        riBoundary = eta/(1-eta)+d
        roBoundary = 1/(1-eta)-d
        
        # TODO: decide if the one needs to import the resolution from an argument
        # compute the outside  tangent cylinder part
        #NsPoints = 20 # NsPoints is the number of points int the radius of the gap
        
        NsPoints, Nmax, Lmax, Mmax = self.res
        # there are points also on top of the tangent cylinder
        
        # build the cylindrical radial grid
        s = np.linspace(0, roBoundary, 2 * NsPoints + 1)[1::2]
        s1 = s[s>=riBoundary]
        s2 = s[s<riBoundary]
        # compute the outside  tangent cylinder part
        ss1, no = np.meshgrid(s1, np.ones(2 * NsPoints))
        fs1 = (roBoundary ** 2 - s1 ** 2) ** .5
        x, w = leggauss(NsPoints * 2)
        no, zz1 = np.meshgrid(fs1, x)
        no, ww1 = np.meshgrid(fs1, w)
        zz1 *= fs1
        
        
        # compute the inside of the tangent cylinder
        ss2, no = np.meshgrid(s2, np.ones(2 * NsPoints))
        fs2 = ((roBoundary ** 2 - s2 ** 2) ** .5 - (riBoundary ** 2 - s2 ** 2) ** .5) / 2
        x, w = leggauss(NsPoints)
        w *= .5
        no, zz2 = np.meshgrid(fs2, x)
        no, ww2 = np.meshgrid(fs2, w)
        zz2 *= fs2
        means = ((roBoundary ** 2 - s2 ** 2) ** .5 + (riBoundary ** 2 - s2 ** 2) ** .5) / 2
        zz2 += means
        zz2 = np.vstack((zz2, -zz2))
        ww2 = np.vstack((ww2, ww2))
        
        # combine the 2 grids together
        ss = np.hstack((ss2, ss1))
        self.ss = ss
        zz = np.hstack((zz2, zz1))
        self.zz = zz
        self.ww = np.hstack((ww2, ww1))
        
        # prepare the grid for tchebyshev polynomials
        self.ttheta = np.arctan2(ss, zz)
        self.rr = (ss**2+zz**2)**.5
        self.xx = (self.rr - b)/a
        self.ccos_theta = np.cos(self.ttheta)
        self.ssin_theta = np.sin(self.ttheta)
        
        # prepare the division weight
        self.fs = np.hstack([fs2*2, fs1]) * 2

        return

    # generate an integrator
    def generateZIntegrator(self):

        # import the diffeomorphism
        a = self.a
        b = self.b
        
        # import resolution
        Ns, Nmax, Lmax, Mmax = self.res
        NN, LL, MM = self.file_res
        # generate the dictionary that maps the l,m to idx
        idx = idxlm(LL, MM)
        
        # allocate the memory for the 
        nz = self.xx.shape[0] # points in z directions

        # allocate the transform for the radial transform
        self.Tn = np.zeros((Ns, nz, Nmax))
        self.dTndr = np.zeros((Ns, nz, Nmax))
        self.Tn_r = np.zeros((Ns, nz, Nmax))
        self.Tn_r2 = np.zeros((Ns, nz, Nmax))
        self.d2Tndr2 = np.zeros((Ns, nz, Nmax))
        
        # allocate the plm parts for the integrals
        self.plm = np.zeros((Ns, Lmax, Mmax, nz))
        self.dplm = np.zeros((Ns, Lmax, Mmax, nz))
        self.plm_sin = np.zeros((Ns, Lmax, Mmax, nz))
        for id_s in range(Ns):

            x = self.xx[:, id_s]
            #w = self.ww[:, id_s]
            #theta = self.ttheta[:, id_s]
            #sin_theta = self.ssin_theta[:, id_s]
            xtheta = self.ccos_theta[:, id_s]
            
            # compute the radial matrices
            self.Tn[id_s, :, :] = shell.proj_radial(Nmax, a, b, x) # evaluate the Tn
            self.dTndr[id_s, :, :] = shell.proj_dradial_dr(Nmax, a, b, x)  # evaluate 1/r d/dr(r Tn)
            self.Tn_r[id_s, :, :] = shell.proj_radial_r(Nmax, a, b, x)  # evaluate 1/r Tn

            # produce the mapping for the tri-curl part
            self.Tn_r2[id_s, :, :] = shell.proj_radial_r2(Nmax, a, b, x) # evaluate 1/r**2 Tn
            self.d2Tndr2[id_s, :, :] = shell.proj_lapl(Nmax, a, b, x) # evaluate 1/r**2 dr r**2 dr


            for l in range(0, min(LL, Lmax) ):
                for m in range(0, min(l+1, MM, Mmax) ):

                    # the computation is carried out over a vertical grid
                    # assume that u_r, u_theta and u_phi are vectors
                    # compute the theta evaluations
                    #xtheta = xx_th[:, id_s]
                    self.plm[id_s, l, m, :] = spherical.lplm(Lmax, l, m, xtheta)
                    self.dplm[id_s, l, m, :] = spherical.dplm(Lmax, l, m, xtheta)
                    self.plm_sin[id_s, l, m, :] = spherical.lplm_sin(Lmax, l, m, xtheta)

        pass

# function creating a dictionary to index data for SLFl and WLFl geometries
def idxlm(L, M):
    #idxlm = np.zeros((L+1, M+1), dtype = int)
    idxlm = {}
    ind = 0 
    for l in range(0, L):
        for m in range(0, min(l, M)+1):
            idxlm[(l,m)] = ind
            ind = ind + 1
    
    return idxlm

# define the dot function where A is real and b is complex
# avoid the casting of A into complex ==> slow
def dot(A, b):
    # not faster for small matrices

    assert(A.dtype =='float')
    assert(b.dtype == 'complex')
    return np.dot(A,b.real) + np.dot(A,b.imag)*1j

def integrate(state,  zInt):
    """
    returns the z-integral of vorticity and the flow up to 
    a max spherical harmonic order Mmax and Tschebyshev Polynomial (Nmax)
    
    vort_int = integrate(state, Integrator)
    """
    
    fin = h5py.File(state, 'r')

    # import the maximal resolution of the file
    LL = fin['/truncation/spectral/dim2D'].value + 1
    MM = fin['/truncation/spectral/dim3D'].value + 1
    NN = fin['/truncation/spectral/dim1D'].value + 1
    E = fin['/physical/ekman'].value
    dataP = fin['velocity/velocity_pol'].value
    dataT = fin['velocity/velocity_tor'].value
    eta = fin['/physical/rratio'].value
    fin.close()

    # compute the diffeomorfism parameters between Tchebyshev and radius space
    a = zInt.a
    b = zInt.b
    fs = zInt.fs
    s = zInt.ss[0, :]

    # import the resolution of the integrator
    Ns, Nmax, Lmax, Mmax = zInt.res

    # generate the dictionary that maps the l,m to idx
    # it is based on the resolution of the imported data -> state
    idx = idxlm(LL, MM)

    # transform into complex data
    dataT=dataT[:,:,0]+dataT[:,:,1]*1j
    dataP=dataP[:,:,0]+dataP[:,:,1]*1j

    # truncate the data
    dataT = dataT[:, :Nmax]
    dataP = dataP[:, :Nmax]
    
    #Allocating memory for output
    vort_int = np.zeros((Mmax, Ns), dtype=complex)
    Ur_int = np.zeros((Mmax, Ns), dtype=complex)
    Uth_int = np.zeros((Mmax, Ns), dtype=complex)
    Uphi_int = np.zeros((Mmax, Ns), dtype=complex)
    Us_int = np.zeros((Mmax, Ns), dtype=complex)
    Uz_int = np.zeros((Mmax, Ns), dtype=complex)
    H_int = np.zeros((Mmax, Ns), dtype=complex)

    for id_s in range(Ns):

        #x = zInt.xx[:, id_s]
        w = zInt.ww[:, id_s]
        #theta = zInt.ttheta[:, id_s]
        sin_theta = zInt.ssin_theta[:, id_s]
        cos_theta = zInt.ccos_theta[:, id_s]
        #xtheta = cos_theta
                            
        # compute the radial matrices
        #Tn  = shell.proj_radial(Nmax, a, b, x) # evaluate the Tn
        Tn  = zInt.Tn[id_s, :, :]
        #dTndr = shell.proj_dradial_dr(Nmax, a, b, x)  # evaluate 1/r d/dr(r Tn)
        dTndr = zInt.dTndr[id_s, :, :]
        #Tn_r = shell.proj_radial_r(Nmax, a, b, x)  # evaluate 1/r Tn
        Tn_r = zInt.Tn_r[id_s, :, :]
                            
        # produce the mapping for the tri-curl part
        #Tn_r2 = shell.proj_radial_r2(Nmax, a, b, x) # evaluate 1/r**2 Tn
        Tn_r2 = zInt.Tn_r2[id_s, :, :]
        #d2Tndr2 = shell.proj_lapl(Nmax, a, b, x) # evaluate 1/r**2 dr r**2 dr
        d2Tndr2 = zInt.d2Tndr2[id_s, :, :]

        for l in range(0, min(LL, Lmax) ):
            for m in range(0, min(l+1, MM, Mmax) ):

                # the computation is carried out over a vertical grid
                # assume that u_r, u_theta and u_phi are vectors
                # import the theta parts from the integrator
                #plm = spherical.lplm(Lmax, l, m, xtheta)
                plm = zInt.plm[id_s, l, m, :]
                #dplm = spherical.dplm(Lmax, l, m, xtheta)
                dplm = zInt.dplm[id_s, l, m, :]
                #plm_sin = spherical.lplm_sin(Lmax, l, m, xtheta)
                plm_sin = zInt.plm_sin[id_s, l, m, :]

                # compute the transformed radial part
                ur_part = l*(l+1)*dot(Tn_r, dataP[idx[l, m], :])
                utor_part = dot(dTndr, dataP[idx[l, m], :])
                upol_part = dot(Tn, dataT[idx[l, m], :])
                omegar_part = l*(l+1)*dot(Tn_r, dataT[idx[l, m], :])
                omegator_part = dot(dTndr, dataT[idx[l, m], :])
                omegapol_part = -( dot(d2Tndr2, dataP[idx[l, m], :]) - l*(l+1)*dot(Tn_r2, dataP[idx[l, m], :]) )

                # compute the r-coordinate components
                u_r = ur_part * plm
                u_theta = dplm * upol_part + 1j * m * plm_sin * utor_part
                u_phi = 1j * m * plm_sin * upol_part - dplm * utor_part
                omega_r = omegar_part * plm
                omega_theta = dplm * omegator_part + 1j * m * plm_sin * omegapol_part
                #omega_phi = 1j * m * plm_sin * omegator_part - dplm * omegapol_part

                # convert in cylindrical coordinate components
                #u_z = u_r * cos_theta - u_theta * sin_theta
                u_s = u_r * sin_theta + u_theta * cos_theta
                omega_z = omega_r * cos_theta - omega_theta * sin_theta
                #omega_s = omega_r * sin_theta + omega_theta * cos_theta

                # For real data:
                #vort_z = zInt.vortz_tor[id_s, n, idxM[l, m], :] * dataP[idx[l, m], :] \
                #         + zInt.vortz_pol[id_s, n, idxM[l, m], :] * dataT[idx[l, m], :]
                # u_r = zInt.ur_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                #u_th = zInt.uth_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] \
                #        + zInt.uth_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                #u_phi = zInt.uphi_tor[id_s, n, idxM[l, m], :] * dataT[idx[l, m], n] \
                #        + zInt.uphi_pol[id_s, n, idxM[l, m], :] * dataP[idx[l, m], n]
                #u_s = zInt.us_tor[id_s, n, idxM[l, m], :] * dataT[idx[l, m], n] \
                #      + zInt.us_pol[id_s, n, idxM[l, m], :] * dataP[idx[l, m], n]
                #u_z = zInt.uz_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] \
                #      + zInt.uz_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]

                # don't forget to include the factor sqrt(1-s^2) to compute the integral
                # For real data:
                # Integrate
                vort_int[m, id_s] += sum(w * omega_z) #/ fs[id_s]
                # Nicolo: modified the weight to fs
                #Ur_int[m, id_s] += sum(w*u_r) / fs[id_s]
                #Uth_int[m, id_s] += sum(w*u_th) / fs[id_s]
                Uphi_int[m, id_s] += sum(w * u_phi) #/ fs[id_s]
                Us_int[m, id_s] += sum(w * u_s) #/ fs[id_s]
                #Uz_int[m, id_s] += sum(w*u_z) / fs[id_s]
                #H_int[m, id_s] += sum(w*u_z*vort_z) / fs[id_s]

    result = {'s': s, 'm': np.arange(Mmax), 'Omega_z': vort_int, 'U_phi': Uphi_int, 'U_s': Us_int}
    return result

def computeRealFields(spectral_result, filter=[]):
    #TODO: consider if one wants to apply an m=0 filter
    # generate the result of this function
    result = dict()
    
    # function used to implemented to generate the real representation of the field
    s = spectral_result['s']
    m = spectral_result['m']

    # generate the real grid for plotting
    mn = 2*(len(m)-1)
    phi = np.arange(0, mn )*2*np.pi /mn
    # append the values that close the poles
    s = np.hstack(([0.], s))
    phi = np.hstack((phi, [np.pi*2]))
    ss, pphi = np.meshgrid(s, phi)
    xx = np.cos(pphi) * ss
    yy = np.sin(pphi) * ss

    print(len(s), len(phi))
    # store the grid
    result['xx'] = xx
    result['yy'] = yy
    result['phi'] = pphi
    result['s'] = ss

    # carry out the inverse fourier transform
    for k in spectral_result.keys():

        # skip over all the non matrix fields
        if spectral_result[k].ndim <2:
            continue

        # truncate the spectrum if needed
        temp = spectral_result[k]
        for m in filter:
            temp[m,:] = 0.
        # compute the tranforms
        field = fft.irfft(temp ,axis=0)
        print('Shape after the transform:', field.shape)
        # attach the right columns in the right place
        field = np.vstack((field,field[0, :]))
        temp1 = field[:, 0].reshape((-1, 1))
        field = np.hstack((np.ones_like(temp1) * np.mean(temp1), field))
        result[k] = field
        print('Shape after the additions:', temp.shape, field.shape)
        
        
    
    return result


def computeZintegralOnTheFly(state, scale, zInt):
    """
    returns the z-integral of vorticity and the flow up to 
    a max spherical harmonic order Mmax and Worland Polynomial (Nmax)
    vort_int=computeZintegral(state, vortz_tor, vortz_pol, ur_pol, uphi_tor, uphi_pol, (Nmax, Lmax, Mmax, N_s), w, grid_s)
    """
    f = h5py.File(state, 'r')
    LL = f['/truncation/spectral/dim2D'].value + 1
    MM = f['/truncation/spectral/dim3D'].value + 1
    NN = f['/truncation/spectral/dim1D'].value + 1
    E = f['/physical/ekman'].value
    dataP = f['velocity/velocity_pol'].value
    dataT = f['velocity/velocity_tor'].value
    eta = f['/physical/rratio'].value
    f.close()

    # compute the diffeomorfism parameters between Tchebyshev and radius space
    a = .5
    b = .5*(1+eta)/(1-eta)

    # compute boundary layer
    d = 10.*E**.5
    riBoundary = eta/(1-eta)+d
    roBoundary = 1/(1-eta)-d

    # TODO: decide if the one needs to import the resolution from an argument
    # compute the outside  tangent cylinder part
    #NsPoints = 20 # NsPoints is the number of points int the radius of the gap

    NsPoints, Nmax, Lmax, Mmax = zInt.res
    # there are points also on top of the tangent cylinder

    # build the cylindrical radial grid
    s = np.linspace(0, roBoundary, 2 * NsPoints + 1)[1::2]
    s1 = s[s>=riBoundary]
    s2 = s[s<riBoundary]
    # compute the outside  tangent cylinder part
    ss1, no = np.meshgrid(s1, np.ones(2 * NsPoints))
    fs1 = (roBoundary ** 2 - s1 ** 2) ** .5
    x, w = leggauss(NsPoints * 2)
    no, zz1 = np.meshgrid(fs1, x)
    no, ww1 = np.meshgrid(fs1, w)
    zz1 *= fs1


    # compute the inside of the tangent cylinder
    ss2, no = np.meshgrid(s2, np.ones(2 * NsPoints))
    fs2 = ((roBoundary ** 2 - s2 ** 2) ** .5 - (riBoundary ** 2 - s2 ** 2) ** .5) / 2
    x, w = leggauss(NsPoints)
    w *= .5
    no, zz2 = np.meshgrid(fs2, x)
    no, ww2 = np.meshgrid(fs2, w)
    zz2 *= fs2
    means = ((roBoundary ** 2 - s2 ** 2) ** .5 + (riBoundary ** 2 - s2 ** 2) ** .5) / 2
    zz2 += means
    zz2 = np.vstack((zz2, -zz2))
    ww2 = np.vstack((ww2, ww2))

    # combine the 2 grids together
    ss = np.hstack((ss2, ss1))
    zz = np.hstack((zz2, zz1))
    ww = np.hstack((ww2, ww1))

    # prepare the grid for tchebyshev polynomials
    ttheta = np.arctan2(ss, zz)
    rr = (ss**2+zz**2)**.5
    xx = (rr - b)/a
    xx_th = np.cos(ttheta)
    ccos_theta = xx_th
    ssin_theta = np.sin(ttheta)

    # prepare the division weight
    fs = np.hstack([fs2*2, fs1]) * 2

    # prepare resolution for the integrator
    Ns = ss.shape[1]
    # edit: now Ns == NsPoints

    # generate the dictionary that maps the l,m to idx
    idx = idxlm(LL, MM)
    #idxM = idxlm(Lmax,Mmax)

    #Transform into complex data
    dataT=dataT[:,:,0]+dataT[:,:,1]*1j
    dataP=dataP[:,:,0]+dataP[:,:,1]*1j
    
    #Allocating memory for output
    vort_int = np.zeros((Mmax, Ns), dtype=complex)
    Ur_int = np.zeros((Mmax, Ns), dtype=complex)
    Uth_int = np.zeros((Mmax, Ns), dtype=complex)
    Uphi_int = np.zeros((Mmax, Ns), dtype=complex)
    Us_int = np.zeros((Mmax, Ns), dtype=complex)
    Uz_int = np.zeros((Mmax, Ns), dtype=complex)
    H_int = np.zeros((Mmax, Ns), dtype=complex)

    #main idea compute for each m the integral 
    #Int(f(s,z)dz) = sum(weight*vort_z(s,z)) = f(s)
    #vort_z function that returns the component of the z-vorticity
    # old

    for id_s in range(Ns):

        x = xx[:, id_s]
        w = ww[:, id_s]
        theta = ttheta[:, id_s]
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        xtheta = xx_th[:, id_s]
                            
        # compute the radial matrices
        Tn  = shell.proj_radial(Nmax, a, b, x) # evaluate the Tn
        dTndr = shell.proj_dradial_dr(Nmax, a, b, x)  # evaluate 1/r d/dr(r Tn)
        Tn_r = shell.proj_radial_r(Nmax, a, b, x)  # evaluate 1/r Tn
                            
        # produce the mapping for the tri-curl part
        Tn_r2 = shell.proj_radial_r2(Nmax, a, b, x) # evaluate 1/r**2 Tn
        d2Tndr2 = shell.proj_lapl(Nmax, a, b, x) # evaluate 1/r**2 dr r**2 dr


        for l in range(0, min(LL, Lmax) ):
            for m in range(0, min(l+1, MM, Mmax) ):

                # the computation is carried out over a vertical grid
                # assume that u_r, u_theta and u_phi are vectors
                # compute the theta evaluations
                plm = spherical.lplm(Lmax, l, m, xtheta)
                dplm = spherical.dplm(Lmax, l, m, xtheta)
                plm_sin = spherical.lplm_sin(Lmax, l, m, xtheta)

                # compute the transformed radial part
                ur_part = l*(l+1)*dot(Tn_r, dataP[idx[l, m], :Nmax])
                utor_part = dot(dTndr, dataP[idx[l, m], :Nmax])
                upol_part = dot(Tn, dataT[idx[l, m], :Nmax])
                omegar_part = l*(l+1)*dot(Tn_r, dataT[idx[l, m], :Nmax])
                omegator_part = dot(dTndr, dataT[idx[l, m], :Nmax])
                omegapol_part = -( dot(d2Tndr2, dataP[idx[l, m], :Nmax]) - l*(l+1)*dot(Tn_r2, dataP[idx[l, m], :Nmax]) )

                # compute the r-coordinate components
                u_r = ur_part * plm
                u_theta = dplm * upol_part + 1j * m * plm_sin * utor_part
                u_phi = 1j * m * plm_sin * upol_part - dplm * utor_part
                omega_r = omegar_part * plm
                omega_theta = dplm * omegator_part + 1j * m * plm_sin * omegapol_part
                #omega_phi = 1j * m * plm_sin * omegator_part - dplm * omegapol_part

                # convert in cylindrical coordinate components
                #u_z = u_r * cos_theta - u_theta * sin_theta
                u_s = u_r * sin_theta + u_theta * cos_theta
                omega_z = omega_r * cos_theta - omega_theta * sin_theta
                #omega_s = omega_r * sin_theta + omega_theta * cos_theta

                # For real data:
                #vort_z = zInt.vortz_tor[id_s, n, idxM[l, m], :] * dataP[idx[l, m], :] \
                #         + zInt.vortz_pol[id_s, n, idxM[l, m], :] * dataT[idx[l, m], :]
                # u_r = zInt.ur_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                #u_th = zInt.uth_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] \
                #        + zInt.uth_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]
                #u_phi = zInt.uphi_tor[id_s, n, idxM[l, m], :] * dataT[idx[l, m], n] \
                #        + zInt.uphi_pol[id_s, n, idxM[l, m], :] * dataP[idx[l, m], n]
                #u_s = zInt.us_tor[id_s, n, idxM[l, m], :] * dataT[idx[l, m], n] \
                #      + zInt.us_pol[id_s, n, idxM[l, m], :] * dataP[idx[l, m], n]
                #u_z = zInt.uz_tor[id_s, n, idxM[l,m], :] * dataT[idx[l,m], n] \
                #      + zInt.uz_pol[id_s, n, idxM[l,m], :] * dataP[idx[l,m], n]

                # don't forget to include the factor sqrt(1-s^2) to compute the integral
                # For real data:
                # Integrate
                vort_int[m, id_s] += sum(w * omega_z) / fs[id_s]
                # Nicolo: modified the weight to fs
                #Ur_int[m, id_s] += sum(w*u_r) / fs[id_s]
                #Uth_int[m, id_s] += sum(w*u_th) / fs[id_s]
                Uphi_int[m, id_s] += sum(w * u_phi) / fs[id_s]
                Us_int[m, id_s] += sum(w * u_s) / fs[id_s]
                #Uz_int[m, id_s] += sum(w*u_z) / fs[id_s]
                #H_int[m, id_s] += sum(w*u_z*vort_z) / fs[id_s]

    
    return ss,vort_int, Us_int, Uphi_int
