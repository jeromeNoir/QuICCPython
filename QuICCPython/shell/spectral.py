import numpy as np
import quicc_bind
from copy import copy
from QuICCPython import tools
from scipy.fftpack import dct, idct
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial.legendre import leggauss
from QuICCPython.shell.energy_tools import ortho_pol_q, ortho_pol_s, ortho_tor
from QuICCPython.shell.quicc_tools import shell_radius
from QuICCPython.shell.quicc_tools import shell


def make1DGrid(spec_state, gridType, specRes):
    """
    INPUT: 
    gridType: string; type of required associated grid, it relates to the
    type of spectral function
    specRes: int; spectral resolution to generate grid
    OUTPUT:
    grid: numpy.array; associated grid in physical space
    """

    # cast the string into lowercase
    gridType = gridType.lower()

    if gridType == 'fourier' or gridType=='f':
        #M: 1/3 rule 
        #physicalRes = 3/2 * 2 specRes = 3* specRes
        #L: uses the same 
        #physialRes = specRes 
        #N:
        #physicalRes = constant * specRes  
        spec_state.physRes.nPhi = 3*specRes
        grid = np.linspace(0,2*np.pi, spec_state.physRes.nPhi + 1)

    elif gridType == 'chebyshev' or gridType == 't':
        #applying 3/2 rule
        physRes = int(3/2 * specRes)
        #x=cos((2*[1:1:N]-1)*pi/(2*N))
        #check physical grid resolution
        grid=np.cos((2*np.arange(1,
                                 physRes+1)-1)*np.pi/(2*physRes))

    elif gridType == 'chebyshevshell' or gridType == 'ts':
        # applying the 3/2 x + 12 rule (comes quite close the
        # Philippe resolution)
        # Note: The 3/2 x + 12 rule is not always correct
        spec_state.physRes.nR = int(3/2 * specRes) + 12
        # retrieve the a and b parameters from the shell aspect
        # ratio
        spec_state.eta = spec_state.parameters.rratio
        spec_state.a, spec_state.b = .5, .5 * (1+spec_state.eta)/(1-spec_state.eta)
        x, w = np.polynomial.chebyshev.chebgauss(spec_state.physRes.nR)
        grid = x*spec_state.a + spec_state.b

    elif gridType == 'legendre' or gridType == 'l':
        # applying 3/2 rule
        spec_state.physRes.nTh = int(3/2 * specRes)
        grid = np.arccos(np.polynomial.legendre.leggauss(spec_state.physRes.nTh)[0])

    elif gridType == 'worland' or gridType == 'w':
        #TODO: Leo, check what the relation between physRes and
        # specRes this should be about 2*specRes Philippes thesis
        # 3/2 N + 3/4 L + 1
        spec_state.physRes.nR = specRes 
        nr = spec_state.physRes.nR
        grid = np.sqrt((np.cos(np.pi*(np.arange(0,2*nr)+0.5)/(2*nr)) + 1.0)/2.0)

    else:
        raise ValueError("Defined types are Fourier, Legendre, Chebyshev, ChebyshevShell and Worland")
        grid = None

    return grid

def makeMeridionalGrid(spec_state):
    """
    INPUT:
    None
    OUTPUT:
    X: np.matrix; first coordinate on a grid
    Y: np.matrix; second coordinate on a grid
    x: np.array; first grid
    y: np.array; second grid
    """
    
    # set default argument
    x = make1DGrid(spec_state, 'ChebyshevShell', spec_state.specRes.N)
    y = make1DGrid(spec_state, 'Legendre', spec_state.specRes.L)

    # make the 2D grid via Kronecker product
    R, Th = np.meshgrid(x, y)
    X = R* np.sin(Th)
    Y = R* np.cos(Th)
    
    return X, Y, x, y

def makeEquatorialGrid(spec_state):
    """
    INPUT:
    None
    OUTPUT:
    X: np.matrix; first coordinate on a grid
    Y: np.matrix; second coordinate on a grid
    x: np.array; first grid
    y: np.array; second grid
    """
    # set default argument
    x = make1DGrid(spec_state, 'ChebyshevShell', spec_state.specRes.N)
    y = make1DGrid(spec_state, 'Fourier', spec_state.specRes.M)

    # make the 2D grid via Kronecker product
    R, Phi = np.meshgrid(x, y)
    X = R*np.cos(Phi)
    Y = R*np.sin(Phi)

    return X, Y, x, y
  
def makeIsoradiusGrid(spec_state):
    """
    INPUT:
    None
    OUTPUT:
    X: np.matrix; first coordinate on a grid
    Y: np.matrix; second coordinate on a grid
    x: np.array; first grid
    y: np.array; second grid
    """
    # set default argument
    x = make1DGrid(spec_state, 'Legendre', spec_state.specRes.L)
    y = make1DGrid(spec_state, 'Fourier', spec_state.specRes.M)
    
    # necessary for matters of transforms (python need to know nR)
    make1DGrid(spec_state, 'ts', spec_state.specRes.N)
    
    # make the 2D grid via Kronecker product
    X, Y = np.meshgrid(x, y)

    return X, Y, x, y    

    
def getPointValue(spec_state, Xvalue, Yvalue, Zvalue,  field='velocity'):

    # assume that the argument are r=x theta=y and phi=z
    assert(len(Xvalue) == len(Yvalue))
    assert(len(Xvalue) == len(Zvalue))
    r = Xvalue
    theta = Yvalue
    phi = Zvalue

    # generate indexer
    # this generate the index lenght also
    spec_state.idx = idxlm(spec_state)
    ridx = {v: k for k, v in spec_state.idx.items()}

    # generate grid
    makeMeridionalGrid(spec_state)

    # pad the fields
    dataT = np.zeros((spec_state.nModes, spec_state.physRes.nR), dtype='complex')
    dataT[:,:spec_state.specRes.N] = getattr(spec_state.fields, field+'_tor')
    dataP = np.zeros((spec_state.nModes, spec_state.physRes.nR), dtype='complex')
    dataP[:,:spec_state.specRes.N] = getattr(spec_state.fields, field+'_pol')

    # prepare the output fields
    FR = np.zeros_like(r)
    FTheta = np.zeros_like(FR)
    FPhi = np.zeros_like(FR)
    FieldOut = [FR, FTheta, FPhi]

    # initialize the spherical harmonics
    spec_state.makeSphericalHarmonics(theta)
    x = (r - spec_state.b)/spec_state.a
    for i in range(spec_state.nModes):

        # get the l and m of the index
        l, m = ridx[i]

        # statement to redute the number of modes considered

        evaluate_mode(spec_state, l, m, FieldOut, dataT[i, :], dataP[i,
                                                              :], r, theta, phi, kron='points', x=x)

    return_value =  {'r': r, 'theta': theta, 'phi': phi, 'uR': FieldOut[0], 'uTheta': FieldOut[1], 'uPhi': FieldOut[2]}

    return return_value

# function creating a dictionary to index data for SLFl, WLFl,
# SLFm or WLFm geometries
def idxlm(spec_state):

    assert (spec_state.geometry == 'shell'), 'The idxlm dictionary is not implemented for the current geometry: '+ spec_state.geometry

    # initialize an empty dictionary
    idxlm = {}

    # initialize the index counter to 0
    ind = 0

    # decide if 'SLFl' or 'SLFm'
    if spec_state.ordering == b'SLFl':
        for l in range(spec_state.specRes.L):
            for m in range(min(l + 1, spec_state.specRes.M)):
                idxlm[(l,m)] = ind
                ind += 1

    elif spec_state.ordering == b'SLFm':
        for m in range(spec_state.specRes.M):
            for l in range(m, spec_state.specRes.L):
                idxlm[(l,m)] = ind
                ind += 1
    spec_state.nModes = ind
    return idxlm

def makeSphericalHarmonics(spec_state, theta):

    # raise error in case of wrong geometry
    assert (spec_state.geometry == 'shell'), 'makeSphericalHarmonics is not implemented for the geometry: '+spec_state.geometry

    # change the theta into x
    x = np.cos(theta)
    spec_state.xth = x

    # initialize storage to 0
    spec_state.Plm = np.zeros((spec_state.nModes, len(x)))
    spec_state.dPlm = np.zeros_like(spec_state.Plm)
    spec_state.Plm_sin = np.zeros_like(spec_state.Plm)

    # generate the reverse indexer
    ridx = {v: k for k, v in spec_state.idx.items()}
    for m in range(spec_state.specRes.M):

        # compute the assoc legendre
        temp1 = tools.plm(spec_state.specRes.L - 1, m, x)
        temp2 = tools.dplm(spec_state.specRes.L - 1, m, x)
        temp3 = tools.plm_sin(spec_state.specRes.L - 1, m, x)
        # assign the Plm to storage
        for l in range(m, spec_state.specRes.L):
            spec_state.Plm[spec_state.idx[l, m], :] = temp1[:, l-m]
            spec_state.dPlm[spec_state.idx[l, m], :] = temp2[:, l-m]
            spec_state.Plm_sin[spec_state.idx[l, m], :] = temp3[:, l-m]
            pass

        pass

# Lookup function to help the implementation of dPlm and Plm_sin 
def plm(spec_state, l, m, x = None):

    if l < m or m < 0 or l < 0:
        return np.zeros_like(spec_state.Plm[0,:])
    elif m > spec_state.specRes.M - 1:
        return np.zeros_like(spec_state.Plm[0, :])
        #temp = np.sqrt((2.0*l+1)/(l-m)) * np.sqrt((2.0*l-1.0)/(l+m))*spec_state.xth*spec_state.plm(l-1,m)-\
            #np.sqrt((2.0*l+1)/(2.0*l-3.0))*np.sqrt((l+m-1.0)/(l+m))*np.sqrt((l-m-1.0)/(l-m))\
            #* spec_state.plm(l-2, m)
        temp = tools.plm(spec_state.specRes.L-1, m, x)
        return temp[:, l - m]
    else:
        return spec_state.Plm[spec_state.idx[l, m], :]
    

# the function takes care of the looping over modes
def getMeridionalSlice(spec_state, phi0=0, field='velocity' ):

    assert (spec_state.geometry == 'shell'), 'makeMeridionalSlice is not implemented for the geometry: '+spec_state.geometry

    # generate indexer
    # this generate the index lenght also
    spec_state.idx = idxlm(spec_state)
    ridx = {v: k for k, v in spec_state.idx.items()}

    # generate grid
    X, Y, r, theta = makeMeridionalGrid(spec_state)

    # pad the fields
    dataT = spec_state.fields.velocity_tor
    dataP = spec_state.fields.velocity_pol
    
    # prepare the output fields
    FR = np.zeros((len(r), len(theta)))
    FTheta = np.zeros_like(FR)
    FPhi = np.zeros_like(FR)
    FieldOut = [FR, FTheta, FPhi]

    # initialize the spherical harmonics
    makeSphericalHarmonics(spec_state, theta)
    #eta = spec_state.parameters.rratio
    #a, b = .5, .5*(1+eta)/(1-eta)
    #x = (r - b)/a
    for i in range(spec_state.nModes):

        # get the l and m of the index
        l, m = ridx[i]

        # evaluate mode
        evaluate_mode(spec_state, l, m, FieldOut, dataT[i, :], dataP[i,
                                                                     :], r, theta, None, kron='meridional', phi0=phi0, field = field)

    return {'x': X, 'y': Y, 'uR': FieldOut[0], 'uTheta': FieldOut[1], 'uPhi': FieldOut[2]}


# the function takes care of the looping over modes
def getEquatorialSlice(spec_state, phi0=0, field = 'velocity'):

    assert (spec_state.geometry == 'shell'), 'makeEquatorialSlice is not implemented for the geometry: '+spec_state.geometry

    # generate indexer
    # this generate the index lenght also
    spec_state.idx = idxlm(spec_state)
    ridx = {v: k for k, v in spec_state.idx.items()}
    
    # generate grid
    X, Y, r, phi = makeEquatorialGrid(spec_state)
    spec_state.grid_r = r
    spec_state.grid_phi = phi
    
    # pad the fields
    dataT = spec_state.fields.velocity_tor
    dataP = spec_state.fields.velocity_pol
    
    # prepare the output fields
    FR = np.zeros((len(r), int(spec_state.physRes.nPhi/2)+1), dtype = 'complex')
    FTheta = np.zeros_like(FR)
    FPhi = np.zeros_like(FR)
    FieldOut = [FR, FTheta, FPhi]

    # initialize the spherical harmonics
    # only for the equatorial values
    makeSphericalHarmonics(spec_state, np.array([np.pi/2]))
    eta = spec_state.parameters.rratio
    spec_state.a, spec_state.b = .5, .5*(1+eta)/(1-eta)
    

    for i in range(spec_state.nModes):

        # get the l and m of the index
        l, m = ridx[i]

        # evaluate the mode update
        evaluate_mode(spec_state, l, m, FieldOut, dataT[i, :], dataP[i,
                                                                     :], r, None, phi, kron='equatorial', phi0=phi0, field = field)

    # carry out the Fourier Transform in phi direction
    field2 = []
    for i, f in enumerate(FieldOut):
        temp = f
        
        f = np.fft.irfft(temp, axis=1)
        f = f * len(f[0,:])
        f = np.hstack([f,np.column_stack(f[:,0]).T])
        field2.append(f)
    FieldOut = field2
    
    return {'x': X, 'y': Y, 'uR': FieldOut[0], 'uTheta': FieldOut[1], 'uPhi': FieldOut[2]}

# the function takes care of the looping over modes
def getIsoradiusSlice(spec_state, r=.5, phi0=0, field = 'velocity'):
    
    assert (spec_state.geometry == 'shell'), 'makeIsoradiusSlice is not implemented for the geometry: '+spec_state.geometry

    # generate indexer
    # this generate the index lenght also
    spec_state.idx = idxlm(spec_state)
    ridx = {v: k for k, v in spec_state.idx.items()}

    # generate grid
    TTheta, PPhi, theta, phi = makeIsoradiusGrid(spec_state)
    spec_state.grid_theta = theta
    spec_state.grid_phi = phi
    eta = spec_state.parameters.rratio
    
    # pad the fields
    dataT = spec_state.fields.velocity_tor
    dataP = spec_state.fields.velocity_pol
    
    # prepare the output fields
    # use the Fourier tranform approach
    FR = np.zeros((len(theta), int(spec_state.physRes.nPhi/2)+1), dtype = 'complex')
    FTheta = np.zeros_like(FR)
    FPhi = np.zeros_like(FR)
    FieldOut = [FR, FTheta, FPhi]

    # prepare the point to evaluate the radial functions
    spec_state.a, spec_state.b = .5, .5 * (1 + eta)/(1 - eta)
    r += eta/(1-eta)
    x = (r - spec_state.b)/spec_state.a
    
    # initialize the spherical harmonics
    makeSphericalHarmonics(spec_state, theta)

    for i in range(spec_state.nModes):

        # get the l and m of the index
        l, m = ridx[i]

        # update the field for the current mode
        evaluate_mode(spec_state, l, m, FieldOut, dataT[i, :], dataP[i,
                                                                     :], r, theta, phi, kron='isogrid', phi0=phi0, x = x, field = field)

    field2 = []
    for i, f in enumerate(FieldOut):
        temp = f
        
        f = np.fft.irfft(temp, axis=1)
        f = f * len(f[0,:])
        f = np.hstack([f,np.column_stack(f[:,0]).T])
        field2.append(f)

    FieldOut = field2
    return {'theta': TTheta, 'phi': PPhi, 'uR': FieldOut[0], 'uTheta': FieldOut[1], 'uPhi': FieldOut[2]}

def evaluate_mode(spec_state, l, m, *args, **kwargs):

    # raise exception if wrong geometry
    assert (spec_state.geometry == 'shell'), 'evaluate_mode is being used for the wrong geometry: '+spec_state.geometry

    
    
    # prepare the input data
    Field_r = args[0][0]
    Field_theta = args[0][1]
    Field_phi = args[0][2]
    modeT = copy(args[1])
    modeP = copy(args[2])
    r = args[3]
    theta = args[4]
    phi = args[5]
    phi0 = kwargs.get('phi0', 0.)
    field = kwargs['field']

    # define factor
    factor = 1. if m==0 else 2.
        
    x = kwargs.get('x', None)
    if x == None:
        x = (r - spec_state.b)/spec_state.a
        
    # assume that the mode is weighted like Philippe sets it
    modeP[1:] *= 2.
    modeT[1:] *= 2.

    if field == 'velocity' or field == 'magnetic':
        # assume a QST field
        # prepare the q_part
        modeP_r = cheb.chebval(x, modeP)/r
        q_part = modeP_r * l*(l+1)
        
        # prepare the s_part
        dP = np.zeros_like(modeP)
        d_temp = cheb.chebder(modeP)
        dP[:len(d_temp)] = d_temp
        s_part = modeP_r + cheb.chebval(x, dP)/spec_state.a
        
        # prepare the t_part
        t_part = cheb.chebval(x, modeT)

    elif field == 'vorticity' or field == 'current':
        # assume a QST field
        # prepare the q_part
        modeT_r = cheb.chebval(x, modeT)/r
        q_part = modeT_r * l*(l+1)

        # prepare the s_part
        dT = cheb.chebder(modeT)
        s_part = modeT_r + cheb.chebval(x, dT)/spec_state.a

        # prepare the t_part
        dP = cheb.chebder(modeP)
        d2P = cheb.chebder(dP)
        
        t_part = -(cheb.chebval(x, d2P)/spec_state.a**2 \
                   + 2*cheb.chebval(x, dP)/spec_state.a/r \
                   - l*(l+1) * cheb.chebval(x, modeP)/r**2)
        
    # depending on the kron type it changes how 2d data are formed
    if kwargs['kron'] == 'meridional':
        eimp = np.exp(1j *  m * phi0)

        idx_ = spec_state.idx[l, m]
        Field_r += np.real(tools.kron(q_part, spec_state.Plm[idx_, :]) * eimp) * factor

        Field_theta += np.real(tools.kron(s_part, spec_state.dPlm[idx_, :]) * eimp) * factor #2
        Field_theta += np.real(tools.kron(t_part, spec_state.Plm_sin[idx_, :]) * eimp * 1j * m) * factor# 2
        Field_phi += np.real(tools.kron(s_part, spec_state.Plm_sin[idx_, :]) * eimp * 1j * m) * factor#2
        Field_phi -= np.real(tools.kron(t_part, spec_state.dPlm[idx_, :]) * eimp) * factor#2

    elif kwargs['kron'] == 'points':
        eimp = np.exp(1j *  m * phi)

        idx_ = spec_state.idx[l, m]
        Field_r += np.real(q_part * spec_state.Plm[idx_, :] * eimp) * factor
        
        Field_theta += np.real(s_part * spec_state.dPlm[idx_, :] * eimp) * factor #2
        Field_theta += np.real(t_part * spec_state.Plm_sin[idx_, :] * eimp * 1j * m) * factor#2
        Field_phi += np.real(s_part * spec_state.Plm_sin[idx_, :] * eimp * 1j * m) * factor#2
        Field_phi -= np.real(t_part * spec_state.dPlm[idx_, :] * eimp) * factor#2

    elif kwargs['kron'] == 'equatorial':
        
        idx_ = spec_state.idx[l, m]

        Field_r[:, m] += spec_state.Plm[idx_, :] * q_part 

        Field_theta[:, m] += spec_state.dPlm[idx_, :] * s_part 
        Field_theta[:, m] += spec_state.Plm_sin[idx_, :] * 1j * m * t_part 
        Field_phi[:, m] += spec_state.Plm_sin[idx_, :]* 1j * m * s_part 
        Field_phi[:, m] -= spec_state.dPlm[idx_, :] * t_part 

    elif kwargs['kron'] == 'isogrid':

        idx_ = spec_state.idx[l, m]

        Field_r[:, m] += spec_state.Plm[idx_, :] * q_part

        Field_theta[:, m] += spec_state.dPlm[idx_, :] * s_part
        Field_theta[:, m] += spec_state.Plm_sin[idx_, :] * 1j * m * t_part
        Field_phi[:, m] += spec_state.Plm_sin[idx_, :]* 1j * m * s_part
        Field_phi[:, m] -= spec_state.dPlm[idx_, :] * t_part

    else:
        raise ValueError('Kron type not understood: '+kwargs['kron'])

    pass

# define the following dictionaries for the convertion of field names
# to tor/pol fields and the correct operator
field_spectral = {'velocity': 'velocity', 'vorticity': 'velocity', 'magnetic': 'magnetic', 'current': 'magnetic'}
field_operation = {'velocity': 'simple', 'vorticity': 'curl', 'magnetic': 'simple', 'current': 'curl'}
# The function compute_energy makes use of orthogonal energy norms
# to compute the energy of a field
def computeEnergy(spec_state, field='velocity', rmin = 0., rmax = 1.):
    # INPUT:
    # field_name: string, specifies the field type (velocity, magnetic)

    # init the storage
    tor_energy = 0
    pol_energy = 0

    # precompute the radial tranform parameters
    a, b = .5, .5 * (1 + spec_state.parameters.rratio) / (1 - spec_state.parameters.rratio)

    # compute volume
    ro = 1/(1-spec_state.parameters.rratio)
    ri = spec_state.parameters.rratio/(1-spec_state.parameters.rratio)
    rmin += ri
    rmax += ri
    vol = 4/3*np.pi*(rmax**3-rmin**3)

    # compute the "limits of integration"
    xmin = (rmin - b)/a
    xmax = (rmax - b)/a

    # generate idx indicer
    spec_state.idx = idxlm(spec_state)
    
    # obtain the 2 fields
    if field_operation[field] == 'simple':
        
        Tfield = getattr(spec_state.fields, field_spectral[field] + '_tor')
        Pfield = getattr(spec_state.fields, field_spectral[field] + '_pol')
        
    elif field_operation[field] == 'curl':

        Tfield = getattr(spec_state.fields, field_spectral[field] + '_pol')
        Pfield = getattr(spec_state.fields, field_spectral[field] + '_tor')

    else:
        raise NotImplementedError('Type of curling operation not implemented')
        
    # precompute the integral matrix
    nr = Tfield.shape[1]
    bc = {0: 0, 'cr':1}
    I1 = shell_radius.i1(nr + 1, a, b, bc)

    # loop first over f
    for l in range(spec_state.specRes.L):

        for m in range(min(l+1, spec_state.specRes.M) ):

            # compute factor
            factor = 1. if m==0 else 2.

            # obtain modes
            Tmode = Tfield[spec_state.idx[l, m], :]
            Pmode = Pfield[spec_state.idx[l, m], :]

            tor_energy += factor * l*(l+1) *\
            ortho_tor(len(Tmode), a, b, Tmode.real,
                      Tmode.real, xmin = xmin, xmax = xmax, I1 = I1,
                      operation = field_operation[field], l = l)
            tor_energy += factor * l*(l+1)*\
            ortho_tor(len(Tmode), a, b, Tmode.imag,
                      Tmode.imag, xmin = xmin, xmax = xmax, I1 = I1,
                      operation = field_operation[field], l = l)

            pol_energy += factor * (l*(l+1))**2 *\
            ortho_pol_q(len(Pmode), a, b, Pmode.real,
                        Pmode.real, xmin = xmin, xmax = xmax, I1 = I1)
            pol_energy += factor * (l*(l+1))**2 *\
            ortho_pol_q(len(Pmode), a, b, Pmode.imag,
                        Pmode.imag, xmin = xmin, xmax = xmax, I1 = I1)

            pol_energy += factor *  l*(l+1) *\
            ortho_pol_s(len(Pmode), a, b, Pmode.real,
                        Pmode.real, xmin = xmin, xmax = xmax, I1 = I1)
            pol_energy += factor *  l*(l+1) *\
            ortho_pol_s(len(Pmode), a, b, Pmode.imag,
                        Pmode.imag, xmin = xmin, xmax = xmax, I1 = I1)

    if field_operation[field] == 'curl':
        swap = tor_energy
        tor_energy = pol_energy
        pol_energy = swap
    
    tor_energy /= vol
    pol_energy /= vol
    return tor_energy + pol_energy, tor_energy, pol_energy

def computeModeProduct(spec_state, spectral_state, m, field_name = 'velocity'):

    # init the storage
    tor_product = 0
    pol_product = 0

    # precompute the radial tranform parameters
    a, b = .5, .5 * (1 + spec_state.parameters.rratio) / (1 - spec_state.parameters.rratio)

    # compute volume
    ro = 1/(1-spec_state.parameters.rratio)
    ri = spec_state.parameters.rratio/(1-spec_state.parameters.rratio)
    vol = 4/3*np.pi*(ro**3-ri**3)

    # generate idx indicer
    spec_state.idx = idxlm(spec_state)
    idxQ = idxlm(spectra_state)

    # obtain the 2 fields
    Tfield = getattr(spec_state.fields, field_name + '_tor')
    Pfield = getattr(spec_state.fields, field_name + '_pol')

    # obtain the 2 fields
    QTfield = getattr(spectral_state.fields, field_name + '_tor')
    QPfield = getattr(spectral_state.fields, field_name + '_pol')

    # loop first over f
    for l in range(m, spec_state.specRes.L):

        # compute factor
        factor = 2. if m==0 else 1.

        # obtain modes
        Tmode = Tfield[spec_state.idx[l, m], :]
        Pmode = Pfield[spec_state.idx[l, m], :]
        QTmode = QTfield[idxQ[l, m], :]
        QPmode = QPfield[idxQ[l, m], :]

        tor_product += factor * l*(l+1) *\
            ortho_tor(len(Tmode), a, b, Tmode.real,
                      QTmode.real)
        tor_product += factor * l*(l+1)*\
            ortho_tor(len(Tmode), a, b, Tmode.imag,
                      QTmode.imag)
        tor_product += 1j * factor * l*(l+1) *\
            ortho_tor(len(Tmode), a, b, Tmode.real,
                      QTmode.imag)
        tor_product -= 1j * factor * l*(l+1)*\
            ortho_tor(len(Tmode), a, b, Tmode.imag,
                      QTmode.real)

        pol_product += factor * (l*(l+1))**2 *\
            ortho_pol_q(len(Pmode), a, b, Pmode.real,
                        QPmode.real)
        pol_product += factor * (l*(l+1))**2 *\
            ortho_pol_q(len(Pmode), a, b, Pmode.imag,
                        QPmode.imag)
        pol_product += factor * (l*(l+1))**2 *\
            ortho_pol_q(len(Pmode), a, b, Pmode.real,
                        QPmode.imag)
        pol_product -= 1j * factor * (l*(l+1))**2 *\
            ortho_pol_q(len(Pmode), a, b, Pmode.imag,
                        QPmode.real)

        pol_product += factor *  l*(l+1) *\
            ortho_pol_s(len(Pmode), a, b, Pmode.real,
                        QPmode.real)
        pol_product += factor *  l*(l+1) *\
            ortho_pol_s(len(Pmode), a, b, Pmode.imag,
                        QPmode.imag)
        pol_product += 1j * factor *  l*(l+1) *\
            ortho_pol_s(len(Pmode), a, b, Pmode.real,
                        QPmode.imag)
        pol_product -= 1j * factor *  l*(l+1) *\
            ortho_pol_s(len(Pmode), a, b, Pmode.imag,
                        QPmode.real)

    tor_product /= (2*vol)
    pol_product /= (2*vol)
    return tor_product + pol_product

# generate wigner matrix
def dlmb(L):
    #INPUT:
    #L       Maximum angular degree
    #OUTPUT:
    #D       Lower right quarter of the Wigner D-matrix m>=0
    #d       Masters' concatenated output
    # Computes matrix elements for spherical harmonic polar rotation around
    # the y-axis over 90 degrees only. 
    #Rotation matrix
    #  D_{mm'}(a,b,g) = exp(-ima) d_{mm'}(b) exp(-im'g) 
    # but we factor the rotation itself into:
    #    R(a,b,g)=R(a-pi/2,-pi/2,b)R(0,pi/2,g+pi/2)
    # thus we only need to compute d_{mm'} for b=90.
    # After a code by T. Guy Masters.
    # See also McEwen, 2006.
    # Last modified by fjsimons-at-alum.mit.edu, 08/05/2008
                                  
    d=np.zeros(np.sum( ( np.array(range(L+1)) + 1 )**2 ) )
    # Initialize using D&T C.115.
    # l = 0 
    d[0] = 1
    # l = 1 
    if L >= 1:
        d[1] = 0
        d[2] = 1.0 / np.sqrt(2)
        d[3] = -1.0 / np.sqrt(2)
        d[4]= 0.5 ;
        
    #pointer index
    ind = 5
    #factor
    f1= 0.5
    #Lwait = 100
    #Recursions
    for l in range(2,L+1):
        lp1 = l + 1
        knd = ind + lp1 
        #print(knd)
        fl2p1 = l + lp1
        vect = np.array( range(1,l+1) )
        f = np.sqrt( vect * ( fl2p1 - vect) )
        f1 = f1 * ( 2.0 * l - 1.0 ) / ( 2.0 * l )
        #print f1
        d[knd-1] = -np.sqrt(f1)
        d[knd-2] = 0
        for i in range(2,l+1):
            j = knd-i
            #print('j=',j)
            d[j-1] = -f[i-2] * d[j+1] / f[i-1]
        #print d

        #Positive N (bottom triangle)
        f2 = f1
        g1 = l 
        g2 = lp1
        for N in range(1,l+1):
            knd = knd + lp1
            en2 = N + N
            g1 = g1 + 1
            g2 = g2 - 1
            f2 = f2 * g2 / g1
            #print(f2)
            d[knd - 1] = -np.sqrt(f2)
            d[knd - 2] = d[knd-1]*en2/f[0]
            #print d[knd-2]
            for i in range(2, l-N+1):
                j = knd-i
                d[j-1] = ( en2 * d[j] - f[i-2] * d[j+1] ) / f[i-1]
                #print d[j-1]

        #Fill upper triangle and fix signs
        for j in range(1,l+1):
            for m in range(j,l+1):
                d[ind+m*lp1+j-1]=d[ind+j*lp1+m-l-1]

        isn=1+np.mod(l,2)
        for n in range(0,l+1):
            knd=ind+n*lp1
            for i in range(isn,lp1+1,2):
                d[knd+i-1]=-d[knd+i-1]
        ind=ind+lp1*lp1;

    #Now let's rearrange the coefficients as 1x1, 2x2, 3x3 etc rotation
    #matrices.
    cst=1;
    D=np.empty(L+1,dtype=object)
    #Start of coefficient sequence; need transpose!
    for l in range(1,L+2):
        #Leo: This line doesn't work !!!
        #print l
        #print(len(d[cst-1:cst+l*l-1]))
        #print(np.reshape(d[cst-1:cst+l*l-1],(l,l)))
        D[l-1]=np.reshape(d[cst-1:cst+l*l-1],(l,l))
        cst=cst+l*l
        #print(cst)
    return (D,d)

def computeUniformVorticity(spec_state, rmin = None, rmax = None):
    #f = h5py.File(state, 'r') #read state
    #nR=f['/truncation/spectral/dim1D'].value+1
    nR = spec_state.specRes.N
    #Toroidal Velocity
    #dataT=f['/velocity/velocity_tor'].value
    #dataT = dataT[:,:,0] + 1j* dataT[:, :, 1]
    dataT = spec_state.fields.velocity_tor

    # obtain the map factor for the Tchebyshev polynomials
    #ro = f['/physical/ro'].value
    #rratio = f['/physical/rratio'].value
    ro = spec_state.parameters.ro
    rratio = spec_state.parameters.rratio
    a, b = shell_radius.linear_r2x(ro,rratio)
    
    # compute the weight of the operator
    ri = ro* rratio
    E = spec_state.parameters.ekman
    delta = E**.5*10
    if rmin == None:
        riBoundary = ri+delta
    else:
        riBoundary = rmin + ri
    if rmax == None:
        roBoundary = ro-delta
    else:
        roBoundary = rmax

    volume = (roBoundary**5 - riBoundary**5 )/ (5. * (3. / (4 *np.pi))**.5)

    # define boundary conditions
    # this is for the linear operators from the main Python/quicc functions
    bc = {0:0, 'cr':2}
    R2 = shell_radius.r2(nR+2, a, b, bc)
    bc['cr'] = 1
    R1 = shell_radius.r1(nR+3, a, b, bc)
    I1 = shell_radius.i1(nR+4, a, b, bc)

    Pi = shell.eval(nR+4, a, b, np.array([roBoundary, riBoundary]))
    
    proj_vec = Pi*I1*R1*R2
    proj_vec = np.array((proj_vec[0,:]-proj_vec[1,:])/volume)

    omegax = -np.real(np.dot(proj_vec, np.array(dataT[2,:])))[0]*2**.5
    omegay = np.imag(np.dot(proj_vec, np.array(dataT[2,:])))[0]*2**.5
    omegaz = np.real(np.dot(proj_vec, np.array(dataT[1,:])))[0]
        
    return np.array([omegax, omegay, omegaz])

def rotateStateWigner(spec_state, omega, field):
    #finout = h5py.File(state, 'r+') #read state
    #LL=finout['/truncation/spectral/dim2D'].value
    #MM=finout['/truncation/spectral/dim3D'].value
    #NN=finout['/truncation/spectral/dim1D'].value
    #Ro = finout['/physical/ro'].value

    LL = spec_state.specRes.L - 1
    MM = spec_state.specRes.M - 1
    NN = spec_state.specRes.N - 1
    Ro = spec_state.parameters.ro
    #Toroidal Velocity
    #data=np.array(f[field].value)
    #data=finout[field].value
    data = getattr(spec_state, field)
    
    # obtain Wigner matrices
    DD=dlmb(LL)

    ###Determine rotation axis from spectral coefficients
    #divide by normalization of n(1,1) mode 
    #(-1) from condon-shortley phase

    (alp, bta, gam) = computeEulerAngles(omega * Ro)

    #e(i*m*alpha) = cos(m*alpha) + i * sin(m*alpha)
    #calp = cos(m*alpha)
    #salp = sin(m*alpha)
    mmVect=np.array(range(0,int(MM)+1))
    llVect=np.array(range(0,int(LL)+1))

    calp=np.cos(mmVect*alp)
    salp=np.sin(mmVect*alp)

    #cbta = cos(l*bta)
    #sbta = sin(l*bta)
    cbta=np.cos(llVect*bta)
    sbta=np.sin(llVect*bta)

    # TODO: to be removed, in QuICC this weights are 1
    #Schmidt normalization factors
    NF=np.array([])
    ind =  0
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
                    #NF=np.append(NF,np.sqrt(2)/2) #Schmidt normalization for m != 0
                    NF = np.append(NF, 1.)  # Schmidt normalization for m != 0
                
                ind=ind+1;
            
    #ind = 529 (same as matlab)

    #Rotating EPM results

    for n in range(0, NN+1 ): #N Chebyshev polynomials

        # preallocate the memory
        alp=np.zeros((int(((LL+2)*(LL+1))/2),2))
        bta=np.zeros((int(((LL+2)*(LL+1))/2),2))
        
        ind=0
        for l in range(0, LL+1 ):

            for m in range(0, min(l,MM)+1 ): #Leo fix: m from 0 to l
                #print ('ind, l, m', ind, l , m )
                # Azimuthal rotation (Z) by (ALPHA - PI/2)
                #Dividing by normalisation 
                Cos = data[ind,n][0] / NF[ind] #index, n, real
                Sin = data[ind,n][1] / NF[ind] #index, n, imag
                #Rotate about Z by alpha
                #print(Cos, Sin, calp[m], salp[m])
                alp[ind,0] = Cos * calp[m] + Sin * salp[m]
                alp[ind,1] = Sin * calp[m] - Cos * salp[m]
                
                ind=ind+1
                
            ###### tilt by bta
            li=l+1
            i,j=np.meshgrid(range(1,li+1),range(1,li+1))
            #checkboard * 2 
            IC=(((i+j)+li%2)%2)*2
            IC[:,0]=1 #m=0
            #inverted checkboard *2
            IS=((((i+j)+li%2)%2)<1)*2
            IS[:,0]=1 #m=0
            
            # STEP 1: PASSIVE colatitudinal (Y) rotation over -PI/2
            Cp = np.dot( DD[0][l].T * IC, \
                         alp[ int((l+1)*(l+2) /2 -l-1) : int((l+1) * (l+2) / 2), 0]) #X Tor
            Sp = np.dot( DD[0][l].T * IS, \
                         alp[ int((l+1)*(l+2) /2 - l - 1) : int((l+1) * (l+2) / 2), 1]) #Y Tor

            
            # STEP 2: PASSIVE azimuthal (Z) rotation over BETA 
            Cpp = Cp * cbta[0:l+1].T + Sp * sbta[0:l+1].T
            Spp = Sp * cbta[0:l+1].T - Cp * sbta[0:l+1].T

            # STEP 3: PASSIVE colatitudinal (Y) rotation over PI/2
            Cpp = np.dot( DD[0][l] * IC, Cpp)
            Spp = np.dot( DD[0][l] * IS, Spp)
                        
            ###### STEP4: PASSIVE azimuthal rotation over (GAMMA +PI/2)
            bta[int((l+1)*(l+2)/2-l-1):int((l+1)*(l+2)/2),0] = Cpp
            bta[int((l+1)*(l+2)/2-l-1):int((l+1)*(l+2)/2),1] = Spp
     
        ind=0
        for l in range(0, LL+1 ):
            for m in range(0, min(l,MM)+1 ): #Leo fix: m from 0 to min(l,M)
                #print("ind, l , m ", ind, l, m)
                #Multiplying by Schmidt normalization 
                data[ind,n][0]=bta[ind,0]*NF[ind]
                data[ind,n][1]=bta[ind,1]*NF[ind]
                
                ind=ind+1
                
        #print('dataT:',dataT[6,0])

    rotatedState = data#(dataT, dataP, dataC)

    #finout[field].value[:] = data
    setattr(spec_state.fields, field, data)

    #finout.close()
    #return rotatedState
    pass
"""
def selectModes(state, modes, field):
    f = h5py.File(state, 'r+') #read state
    LL=int(np.array(f['/truncation/spectral/dim2D'], dtype='double'))
    MM=int(np.array(f['/truncation/spectral/dim3D'], dtype='double'))
    NN=int(np.array(f['/truncation/spectral/dim1D'], dtype='double'))

    # impose single modes on field
    data=f[field].value
    
    ind=0
    for l in range(0, LL+1 ):
        for m in range(0, min(l,MM)+1 ): #Leo fix: m from 0 to min(l,M)
            if m not in modes:

                # Set mode to 0
                data[ind,:][0]=0.
                data[ind,:][1]=0.
                
            ind=ind+1
                
    f[field][:] = data
    f.close()
    return data

def removeModes(state, modes, field):
    f = h5py.File(state, 'r+') #read state
    LL=f['/truncation/spectral/dim2D'].value + 1
    MM=f['/truncation/spectral/dim3D'].value + 1
    NN=f['/truncation/spectral/dim1D'].value + 1

    #Toroidal Velocity
    data=f[field].value

    ind=0
    for l in range(0, LL+1 ):
        for m in range(0, min(l,MM)+1 ): #Leo fix: m from 0 to min(l,M)
            if m in modes:
                data[ind,:][0]=0.
                data[ind,:][1]=0.
    
            ind=ind+1
                
    
    f[field][:] = data
    f.close()
    return data


def correctRotation(state, toBeCorrected = ['/velocity/velocity_tor','/velocity/velocity_pol']):
    
    try:
        omega = computeAverageVorticity(state)
        for field in toBeCorrected:
            rotateState(state, omega, field)

    except Exception as e:
        print(e)
        pass
"""
def subtractUniformVorticity(spec_state, omega):
    #finout = h5py.File(state, 'r+')
    #eta = finout['/physical/rratio'].value
    eta = spec_state.parameters.rratio
    a = .5
    b = .5*(1+eta)/(1-eta)

    #dataT = finout['/velocity/velocity_tor'].value
    dataT = spec_state.fields.velocity_tor
    #dataT = dataT[:, :, 0] + dataT[:, :, 1]*1j

    idx = idxlm(spec_state)
    dataT[idx[1, 0], 0] -= 2*(np.pi/3)**.5 * b * omega[2]
    dataT[idx[1, 0], 1] -= 2*(np.pi/3)**.5 * a * omega[2]*.5
    
    dataT[idx[1, 1], 0] -= 2*(2*np.pi/3)**.5 * b * (-omega[0] + omega[1]*1j)*.5
    dataT[idx[1, 1], 1] -= 2*(2*np.pi/3)**.5 * a * (-omega[0] + omega[1]*1j)*.5*.5

    """
    torField = finout['/velocity/velocity_tor'].value
    torField[:, :, 0] = dataT.real
    torField[:, :, 1] = dataT.imag
    finout['/velocity/velocity_tor'][:]=torField
    finout.close()
    """
    pass

def alignAlongFluid(state, omega):

    #finout = h5py.File(state, 'r+')

    #LL=finout['/truncation/spectral/dim2D'].value 
    #MM=finout['/truncation/spectral/dim3D'].value 
    #NN=finout['/truncation/spectral/dim1D'].value 
    #Ro = finout['/physical/ro'].value

    # make a copy of
    from copy import deepcopy
    spec_state = deepcopy(state)
    
    LL = spec_state.specRes.L - 1
    MM = spec_state.specRes.M - 1
    NN = spec_state.specRes.N - 1
    Ro = spec_state.parameters.ro
    # compute Euler angles
    (alpha, beta, gamma) = computeEulerAngles(omega * Ro)
    alpha = float(alpha)
    LL = int(LL)
    MM = int(MM)
        
    # loop over the fields
    for field in vars(spec_state.fields):
        Qlm = np.array(getattr(spec_state.fields, field))
        #Qlm = data[:, :, 0]+ 1j*data[:, :, 1]
        Slm = np.zeros_like(Qlm)
        Tlm = np.zeros_like(Qlm)

        # first rotation (around Z)
        #print(type(Qlm), Qlm.shape, Qlm.dtype, type(Slm), Slm.shape, Slm.dtype)
        Qlm = np.asfortranarray(Qlm)
        Slm = np.asfortranarray(Slm)
        Tlm = np.asfortranarray(Tlm)
        #print(type(alpha), type(LL), type(MM)) 
        
        quicc_bind.ZrotateFull(Qlm, Slm, alpha, LL, MM)

        # second rotation (around X)
        quicc_bind.XrotateFull(Slm, Tlm, beta, LL, MM)

        # third rotation (arond Z, back to original orientation)
        quicc_bind.ZrotateFull(Tlm, Qlm, gamma, LL, MM)

        """
        field_temp = finout[field].value
        field_temp[:, :, 0] = np.real(Qlm)
        field_temp[:, :, 1] = np.imag(Qlm)
        finout[field][:]=field_temp
        """
        setattr(spec_state.fields, field, Qlm)
        
    return spec_state
    
def computeEulerAngles(omega):
    # omega is in the mantle frame
    # the real omega_f in the frame of reference is omega + [0 0 1]
    #[phi0, theta0]: rotation axis of fluid
    omegax = omega[0]
    omegay = omega[1]
    omegaz = omega[2]
    phi0=np.arctan2(omegay, omegax)
    theta0=np.arctan2((omegax**2+omegay**2)**.5, (omegaz+1))

    #### determine axis of fluid from uniform vorticity
    # using thetat0 and phi0 determined from mean vorticity
    #Euler Angles
    #Alpha: 1st rotation about Z
    #R = R(alpha-pi/2, -pi/2, beta)* R(0,pi/2, gamma+pi/2)
    alpha=-phi0-np.pi/2
    #Beta: 2nd rotation about X
    beta=-theta0
    #Gamma: 3rd rotation about Z
    gamma=np.pi/2

    return (alpha, beta, gamma)

def processState(spec_state):
    # spec_state is now a QuICCPython data object

    # compute averate vorticity
    omega = computeAverageVorticity(spec_state)
    
    # copy the state file ( rotations and subtractions of states are done in place
    #filename = 'stateRotated.hdf5'
    #copyfile(state, filename)

    # subtract average vorticity
    subtractUniformVorticity(spec_state, omega)

    # to be rotated: '/velocity/velocity_tor', '/velocity/velocity_pol'
    #fields = ['/velocity/velocity_tor', '/velocity/velocity_pol']
    rotateState(filename, omega)
        
    pass


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
        
    def generateGrid(self, spec_state):
        # here state is needed to obtain the eta parameter    
    
        # import eta value and others
        # open hdf5 file
        #fin = h5py.File(state, 'r')
        #LL = fin['/truncation/spectral/dim2D'].value + 1
        #MM = fin['/truncation/spectral/dim3D'].value + 1
        #NN = fin['/truncation/spectral/dim1D'].value + 1
        LL = spec_state.specRes.L
        MM = spec_state.specRes.M
        NN = spec_state.specRes.N
        self.file_res = (NN, LL, MM)
        E = spec_state.parameters.ekman
        eta = spec_state.parameters.rratio
                
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
        # this value is (h^+ - h^-)/2
        fs2 = ((roBoundary ** 2 - s2 ** 2) ** .5 - (riBoundary ** 2 - s2 ** 2) ** .5) / 2
        x, w = leggauss(NsPoints)
        w *= .5
        no, zz2 = np.meshgrid(fs2, x)
        no, ww2 = np.meshgrid(fs2, w)
        zz2 *= fs2
        # this value is (h^+ + h^-)/2
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

def getZIntegrator(filename):
    zInt = Integrator()
    zInt.load(integrator)
    return zInt

def computeZIntegral(spec_state, field, nS, integrator = None, Lmax = None, Mmax = None, Nmax = None):

    assert field in ['uPhi', 'uS', 'vortZ', 'uZ'], field+' not possible as a field option for computeZIntegral'
    
    # read the state resolution
    LL = spec_state.specRes.L
    MM = spec_state.specRes.M
    NN = spec_state.specRes.N

    # import variables from state
    E = spec_state.parameters.ekman
    dataP = spec_state.fields.velocity_pol
    dataT = spec_state.fields.velocity_tor
    eta = spec_state.parameters.rratio

    # compute the diffeomorfism parameters between Tchebyshev and radius space
    a = .5
    b = .5*(1+eta)/(1-eta)

    # decide wether to load an integrator from memory
    if integrator == None:

        # if no integrator is specified, construct one on the spot
        if Lmax == None:
            Lmax = LL
        if Mmax == None:
            Mmax = MM
        if Nmax == None:
            Nmax = NN

        Nmax = min(Nmax, NN)
        Lmax = min(Lmax, LL)
        Mmax = min(Mmax, MM)
        zInt = Integrator((nS, Nmax, Lmax, Mmax))
        zInt.generateGrid(spec_state)

    else:

        # if integrator string is give, load an integrator
        zInt = getIntegrator(integrator)
        # assume the grid is already written
        
    NsPoints, Nmax, Lmax, Mmax = zInt.res
    print(zInt.res)
    
    ss = zInt.ss
    s = ss[0, :]
    zz = zInt.zz
    ttheta = zInt.ttheta
    ww = zInt.ww
    rr = zInt.rr
    xx = zInt.xx
    xx_th = zInt.ccos_theta
    ssin_theta = zInt.ssin_theta
    fs = zInt.fs
    # prepare resolution for the integrator
    Ns = ss.shape[1]
    # edit: now Ns == NsPoints

    # generate the dictionary that maps the l,m to idx
    idx = idxlm(spec_state)

    # prepare the reshape for the entire grid
    x = np.reshape(xx, (-1))
    w = np.reshape(ww, (-1))
    r = np.reshape(rr, (-1))
    theta = np.reshape(ttheta, (-1))
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    #xtheta = xx_th[:, id_s]
    xtheta = np.reshape(xx_th, (-1))
    
    #Allocating memory for output
    if field == 'vortZ':
        vortZ_int = np.zeros((len(x), Mmax), dtype=complex)
    
    #uR_int = np.zeros_like(vortZ_int)
    #uTh_int = np.zeros_like(vortZ_int)
    if field == 'uPhi':
        uPhi_int = np.zeros((len(x), Mmax), dtype=complex)
    
    if field == 'uS':
        uS_int = np.zeros((len(x), Mmax), dtype=complex)
    
    if field == 'uZ':
        uZ_int = np.zeros((len(x), Mmax), dtype=complex)
    
    #H_int = np.zeros_like(vortZ_int)

    """
    old syntax
    for l in range(0, min(Lmax) ):
        for m in range(0, min(l+1, Mmax) ):
    """
    for m in range(Mmax):

        # precompute the mode in storage
        plm_storage = tools.plm(Lmax, m, xtheta)
        dplm_storage = tools.dplm(Lmax, m, xtheta)
        plm_sin_storage = tools.plm_sin(Lmax, m, xtheta)

        # take into account the factor for the fourier transform
        factor = 1. if m==0 else 2.

        for l in range(m, Lmax):
            
            plm = plm_storage[:, l - m]
            dplm = dplm_storage[:, l - m]
            plm_sin = plm_sin_storage[:, l-m]
            
            # store mode nicely
            modeP = copy(dataP[idx[l, m], :Nmax])
            modeT = copy(dataT[idx[l, m], :Nmax])
            
            # use the DCT weighting
            modeP[1:] *= 2.
            modeT[1:] *= 2.
            
            # compute the transformed radial part
            P_r = cheb.chebval(x, modeP)/r
            dmodeP = cheb.chebder(modeP)
            tempP = cheb.chebval(x, dmodeP)/a
            
            if field == 'uS' or field == 'uPhi' or field == 'uZ':
                ur_part = l*(l+1)*P_r    
                upol_part = P_r + tempP
                utor_part = cheb.chebval(x, modeT)

            if field == 'vortZ':
                T_r = cheb.chebval(x, modeT)/r

                omegar_part = l*(l+1)*T_r
                dmodeT = cheb.chebder(modeT)
                omegator_part = T_r + cheb.chebval(x, dmodeT)/a
                ddmodeP = cheb.chebder(dmodeP)
                omegapol_part = -(cheb.chebval(x, ddmodeP)/a**2 + 2*tempP/r - l*(l+1)/r * P_r )
            
            # compute the r-coordinate components
            if field == 'uS' or field == 'uZ':
                u_r = ur_part * plm * factor
                u_theta = dplm * upol_part + 1j * m * plm_sin * utor_part * factor
            
            if field == 'uPhi':
                u_phi = 1j * m * plm_sin * upol_part - dplm * utor_part * factor

            if field == 'vortZ':
                omega_r = omegar_part * plm * factor
                omega_theta = dplm * omegator_part + 1j * m * plm_sin * omegapol_part * factor
            #omega_phi = 1j * m * plm_sin * omegator_part - dplm * omegapol_part * factor
            
            # convert in cylindrical coordinate components
            if field == 'uZ':
                u_z = u_r * cos_theta - u_theta * sin_theta

            if field == 'uS':
                u_s = u_r * sin_theta + u_theta * cos_theta
            if field == 'vortZ':
                omega_z = omega_r * cos_theta - omega_theta * sin_theta
            #omega_s = omega_r * sin_theta + omega_theta * cos_theta
            
            # weight the computed fields with the associated quadrature weight
            if field == 'vortZ':
                vortZ_int[:, m] += w * omega_z #/ fs[id_s]
            # Nicolo: modified the weight to fs
            #uR_int[:, m] += w * u_r #/ fs[id_s]
            #uTh_int[:, m] += w * u_th #/ fs[id_s]
            if field == 'uPhi':
                uPhi_int[:, m] += w * u_phi #/ fs[id_s]

            if field == 'uS':
                uS_int[:, m] += w * u_s #/ fs[id_s]

            if field == 'uZ':
                uZ_int[:, m] += w * u_z #/ fs[id_s]
            #H_int[:, m] += w * u_z * vort_z #/ fs[id_s]

    # perform the integration:
    # this happens over 2 steps, first reshape the data, then
    # sum over the z axis
    if field == 'vortZ':
        vortZ_temp = np.reshape(vortZ_int, (xx.shape[0], xx.shape[1], Mmax) )
        vortZ_int = np.sum(vortZ_temp, axis=0).T
        FField = vortZ_int
    #uR_temp =  np.reshape(uR_int, (xx.shape[0], xx.shape[1], Mmax) )
    #uR_int = np.sum(uR_temp, axis=0)
    #uTh_temp =  np.reshape(uTh_int, (xx.shape[0], xx.shape[1], Mmax) )
    #uTh_int = np.sum(uTh_temp, axis=0)
    if field == 'uPhi':
        uPhi_temp =  np.reshape(uPhi_int, (xx.shape[0], xx.shape[1], Mmax) )
        uPhi_int = np.sum(uPhi_temp, axis=0).T
        FField = uPhi_int

    if field == 'uS':
        uS_temp =  np.reshape(uS_int, (xx.shape[0], xx.shape[1], Mmax) )
        uS_int = np.sum(uS_temp, axis=0).T
        FField = uS_int

    if field == 'uZ':
        uZ_temp =  np.reshape(uZ_int, (xx.shape[0], xx.shape[1], Mmax) )
        uZ_int = np.sum(uZ_temp, axis=0).T
        FField = uZ_int
    #H_temp =  np.reshape(H_int, (xx.shape[0], xx.shape[1], Mmax) )
    #H_int = np.sum(H_temp, axis=0)

    #result = {'s': s, 'm': np.arange(Mmax), 'vortZ': vortZ_int, 'uPhi': uPhi_int, 'uS': uS_int, 'uZ':uS_int}
    result = {'s': s, 'm': np.arange(Mmax), field: FField}
    return result
