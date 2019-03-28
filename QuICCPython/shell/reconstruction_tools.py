import numpy as np
import tools
#import h5py
from scipy.fftpack import dct, idct
from numpy.polynomial import chebyshev as cheb
#import sys, os
#env = os.environ.copy()
#sys.path.append(env['HOME']+'/workspace/QuICC/Python/')
#from quicc.projection.shell_energy import ortho_pol_q, ortho_pol_s, ortho_tor
from energy_tools import ortho_pol_q, ortho_pol_s, ortho_tor

    
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
    x = spec_state.make1DGrid('ChebyshevShell', spec_state.specRes.N)
    y = spec_state.make1DGrid('Legendre', spec_state.specRes.L)

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
    x = spec_state.make1DGrid('ChebyshevShell', spec_state.specRes.N)
    y = spec_state.make1DGrid('Fourier', spec_state.specRes.M)

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
    x = spec_state.make1DGrid('Legendre', spec_state.specRes.L)
    y = spec_state.make1DGrid('Fourier', spec_state.specRes.M)
    
    # necessary for matters of transforms (python need to know nR)
    spec_state.make1DGrid('ts', spec_state.specRes.N)
    
    # make the 2D grid via Kronecker product
    X, Y = np.meshgrid(x, y)

    return X, Y, x, y    

    
def makePointValue(spec_state, Xvalue, Yvalue, Zvalue,  field='velocity'):

    # assume that the argument are r=x theta=y and phi=z
        assert(len(Xvalue) == len(Yvalue))
    assert(len(Xvalue) == len(Zvalue))
    r = Xvalue
    theta = Yvalue
    phi = Zvalue

    # generate indexer
    # this generate the index lenght also
    spec_state.idx = spec_state.idxlm()
    ridx = {v: k for k, v in spec_state.idx.items()}

    # generate grid
    spec_state.makeMeridionalGrid()

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

    return_value =  {'r': r, 'theta': theta, 'phi': phi, 'U_r': FieldOut[0], 'U_theta': FieldOut[1], 'U_phi': FieldOut[2]}

    return return_value

# function creating a dictionary to index data for SLFl, WLFl,
# SLFm or WLFm geometries
def idxlm(spec_state):

    assert (spec_state.geometry == 'shell'), 'The idxlm dictionary is not implemented for the current geometry', spec_state.geometry

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
        temp = tools.plm(spec_state.specRes.L-1, m, x)

        # assign the Plm to storage
        for l in range(m, spec_state.specRes.L):
            spec_state.Plm[spec_state.idx[l, m], :] = temp[:, l-m]
            pass

        pass

    # compute dPlm and Plm_sin
    for i in range(spec_state.nModes):
        l, m = ridx[i]
        spec_state.dPlm[i, :] = -.5 * (((l+m)*(l-m+1))**0.5 * spec_state.plm(l,m-1) -
                                 ((l-m)*(l+m+1))**.5 * spec_state.plm(l, m+1, x) )

        if m!=0:
            spec_state.Plm_sin[i, :] = -.5/m * (((l-m)*(l-m-1))**.5 *
                                          spec_state.plm(l-1, m+1, x) + ((l+m)*(l+m-1))**.5 *
                                        spec_state.plm(l-1, m-1)) * ((2*l+1)/(2*l-1))**.5
        else:
            spec_state.Plm_sin[i, :] = spec_state.plm(l, m)/(1-x**2)**.5

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
def makeMeridionalSlice(spec_state, p=0, modeRes = (120,120) ):

    assert (spec_state.geometry == 'shell'), 'makeMeridionalSlice is not implemented for the geometry: '+spec_state.geometry

    # generate indexer
    # this generate the index lenght also
    spec_state.idx = spec_state.idxlm()
    ridx = {v: k for k, v in spec_state.idx.items()}

    # generate grid
    X, Y, r, theta = spec_state.makeMeridionalGrid()

    # pad the fields
    dataT = np.zeros((spec_state.nModes, spec_state.physRes.nR), dtype='complex')
    dataT[:,:spec_state.specRes.N] = spec_state.fields.velocity_tor
    dataP = np.zeros((spec_state.nModes, spec_state.physRes.nR), dtype='complex')
    dataP[:,:spec_state.specRes.N] = spec_state.fields.velocity_pol

    # prepare the output fields
    FR = np.zeros((len(r), len(theta)))
    FTheta = np.zeros_like(FR)
    FPhi = np.zeros_like(FR)
    FieldOut = [FR, FTheta, FPhi]

    # initialize the spherical harmonics
    spec_state.makeSphericalHarmonics(theta)

    for i in range(spec_state.nModes):

        # get the l and m of the index
        l, m = ridx[i]

        # evaluate mode
        evaluate_mode(spec_state, l, m, FieldOut, dataT[i, :], dataP[i,
                                                              :], r, theta, None, kron='meridional', phi0=p)

    return {'x': X, 'y': Y, 'U_r': FieldOut[0], 'U_theta': FieldOut[1], 'U_phi': FieldOut[2]}


# the function takes care of the looping over modes
def makeEquatorialSlice(spec_state, phi0=0 ):

    assert (spec_state.geometry == 'shell'), 'makeEquatorialSlice is not implemented for the geometry: '+spec_state.geometry

    # generate indexer
    # this generate the index lenght also
    spec_state.idx = spec_state.idxlm()
    ridx = {v: k for k, v in spec_state.idx.items()}
    
    # generate grid
    X, Y, r, phi = spec_state.makeEquatorialGrid()
    spec_state.grid_r = r
    spec_state.grid_phi = phi
    # pad the fields
    dataT = np.zeros((spec_state.nModes, spec_state.physRes.nR), dtype='complex')
    dataT[:,:spec_state.specRes.N] = spec_state.fields.velocity_tor
    dataP = np.zeros((spec_state.nModes, spec_state.physRes.nR), dtype='complex')
    dataP[:,:spec_state.specRes.N] = spec_state.fields.velocity_pol

    # prepare the output fields
    FR = np.zeros((len(r), int(spec_state.physRes.nPhi/2)+1), dtype = 'complex')
    FTheta = np.zeros_like(FR)
    FPhi = np.zeros_like(FR)
    FieldOut = [FR, FTheta, FPhi]

    # initialize the spherical harmonics
    # only for the equatorial values
    spec_state.makeSphericalHarmonics(np.array([np.pi/2]))

    for i in range(spec_state.nModes):

        # get the l and m of the index
        l, m = ridx[i]

        # evaluate the mode update
        evaluate_mode(spec_state, l, m, FieldOut, dataT[i, :], dataP[i,
                                                                    :], r, None, phi, kron='equatorial', phi0=phi0)

    # carry out the Fourier Transform in phi direction
    field2 = []
    for i, f in enumerate(FieldOut):
        temp = f
        if i > 0:
            temp[0,:] *= 2.
        f = np.fft.irfft(temp, axis=1)
        f = f * len(f[0,:])
        f = np.hstack([f,np.column_stack(f[:,0]).T])
        field2.append(f)
    FieldOut = field2
    
    return {'x': X, 'y': Y, 'U_r': FieldOut[0], 'U_theta': FieldOut[1], 'U_phi': FieldOut[2]}

# the function takes care of the looping over modes
def makeIsoradiusSlice(spec_state, r=None, phi0=0 ):

    assert (spec_state.geometry == 'shell'), 'makeIsoradiusSlice is not implemented for the geometry: '+spec_state.geometry

    # generate indexer
    # this generate the index lenght also
    spec_state.idx = idxlm(spec_state)
    ridx = {v: k for k, v in spec_state.idx.items()}

    """
    if modeRes == None:
        modeRes=(spec_state.specRes.L, spec_state.specRes.M)
    """

    # generate grid
    TTheta, PPhi, theta, phi = spec_state.makeIsoradiusGrid()
    spec_state.grid_theta = theta
    spec_state.grid_phi = phi
    # pad the fields
    dataT = np.zeros((spec_state.nModes, spec_state.physRes.nR), dtype='complex')
    dataT[:,:spec_state.specRes.N] = spec_state.fields.velocity_tor
    dataP = np.zeros((spec_state.nModes, spec_state.physRes.nR), dtype='complex')
    dataP[:,:spec_state.specRes.N] = spec_state.fields.velocity_pol


    # prepare the output fields
    #FR = np.zeros((len(theta), len(phi)))
    # attempt the Fourier tranform approach
    FR = np.zeros((len(theta), int(spec_state.physRes.nPhi/2)+1), dtype = 'complex')
    FTheta = np.zeros_like(FR)
    FPhi = np.zeros_like(FR)
    FieldOut = [FR, FTheta, FPhi]

    # prepare the "collocation point"
    if r == None:
        x = 0
        r = .5*(spec_state.eta+1)/(1-spec_state.eta)
    else:
        spec_state.a, spec_state.b = .5, .5 * (1+spec_state.eta)/(1-spec_state.eta)
        x = (r - spec_state.b)/spec_state.a


    # initialize the spherical harmonics
    spec_state.makeSphericalHarmonics(theta)

    for i in range(spec_state.nModes):

        # get the l and m of the index
        l, m = ridx[i]

        # update the field for the current mode
        evaluate_mode(spec_state, l, m, FieldOut, dataT[i, :], dataP[i,
                                                                     :], r, theta, phi, kron='isogrid', phi0=phi0, x=x)

    field2 = []
    for i, f in enumerate(FieldOut):
        temp = f
        if i > 0:
            temp[:,0] *= 2.
        f = np.fft.irfft(temp, axis=1)
        f = f * len(f[0,:])
        f = np.hstack([f,np.column_stack(f[:,0]).T])
        field2.append(f)

    FieldOut = field2
    return {'theta': TTheta, 'phi': PPhi, 'U_r': FieldOut[0], 'U_theta': FieldOut[1], 'U_phi': FieldOut[2]}

def evaluate_mode(spec_state, l, m, *args, **kwargs):

    # raise exception if wrong geometry
    assert (spec_state.geometry == 'shell'), 'evaluate_mode is being used for the wrong geometry: '+spec_state.geometry

    # prepare the input data
    Field_r = args[0][0]
    Field_theta = args[0][1]
    Field_phi = args[0][2]
    modeT = args[1]
    modeP = args[2]
    r = args[3]
    theta = args[4]
    phi = args[5]
    phi0 = kwargs['phi0']


    # define factor
    factor = 1. if m==0 else 2.

    if kwargs['kron'] == 'isogrid' or kwargs['kron'] == 'points':
        x = kwargs['x']
        
        # assume that the mode is weighted like Philippe sets it
        modeP[1:] *= 2.
        modeT[1:] *= 2.

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

        
    else: # hence either merdiional or equatorial

        # prepare the q_part
        modeP_r = idct(modeP, type = 2)/r
        q_part = modeP_r * l*(l+1)

        # prepare the s_part
        dP = np.zeros_like(modeP)
        d_temp = cheb.chebder(modeP)
        dP[:len(d_temp)] = d_temp
        s_part = modeP_r + idct(dP, type = 2)/spec_state.a

        # prepare the t_part
        t_part = idct(modeT, type = 2)
        
    # depending on the kron type it changes how 2d data are formed
    if kwargs['kron'] == 'meridional':
        eimp = np.exp(1j *  m * phi0)

        idx_ = spec_state.idx[l, m]
        Field_r += np.real(tools.kron(q_part, spec_state.Plm[idx_, :]) * eimp) * factor

        Field_theta += np.real(tools.kron(s_part, spec_state.dPlm[idx_, :]) * eimp) * 2
        Field_theta += np.real(tools.kron(t_part, spec_state.Plm_sin[idx_, :]) * eimp * 1j * m) * 2
        Field_phi += np.real(tools.kron(s_part, spec_state.Plm_sin[idx_, :]) * eimp * 1j * m) * 2
        Field_phi -= np.real(tools.kron(t_part, spec_state.dPlm[idx_, :]) * eimp) * 2

    elif kwargs['kron'] == 'points':
        eimp = np.exp(1j *  m * phi)

        idx_ = spec_state.idx[l, m]
        Field_r += np.real(q_part * spec_state.Plm[idx_, :] * eimp) * factor
        
        Field_theta += np.real(s_part * spec_state.dPlm[idx_, :] * eimp) * 2
        Field_theta += np.real(t_part * spec_state.Plm_sin[idx_, :] * eimp * 1j * m) * 2
        Field_phi += np.real(s_part * spec_state.Plm_sin[idx_, :] * eimp * 1j * m) * 2
        Field_phi -= np.real(t_part * spec_state.dPlm[idx_, :] * eimp) * 2

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
                
# The function compute_energy makes use of orthogonal energy norms
# to compute the energy of a field
def compute_energy(spec_state, field_name='velocity'):
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
    vol = 4/3*np.pi*(ro**3-ri**3)

    # generate idx indicer
    spec_state.idx = spec_state.idxlm()

    # obtain the 2 fields
    Tfield = getattr(spec_state.fields, field_name + '_tor')
    Pfield = getattr(spec_state.fields, field_name + '_pol')

    # loop first over f
    for l in range(spec_state.specRes.L):

        for m in range(min(l+1, spec_state.specRes.M) ):

            # compute factor
            factor = 2. if m==0 else 1.

            # obtain modes
            Tmode = Tfield[spec_state.idx[l, m], :]
            Pmode = Pfield[spec_state.idx[l, m], :]

            tor_energy += factor * l*(l+1) *\
            ortho_tor(len(Tmode), a, b, Tmode.real,
                      Tmode.real)
            tor_energy += factor * l*(l+1)*\
            ortho_tor(len(Tmode), a, b, Tmode.imag,
                      Tmode.imag)

            pol_energy += factor * (l*(l+1))**2 *\
            ortho_pol_q(len(Pmode), a, b, Pmode.real,
                        Pmode.real)
            pol_energy += factor * (l*(l+1))**2 *\
            ortho_pol_q(len(Pmode), a, b, Pmode.imag,
                        Pmode.imag)

            pol_energy += factor *  l*(l+1) *\
            ortho_pol_s(len(Pmode), a, b, Pmode.real,
                        Pmode.real)
            pol_energy += factor *  l*(l+1) *\
            ortho_pol_s(len(Pmode), a, b, Pmode.imag,
                        Pmode.imag)

    tor_energy /= (2*vol)
    pol_energy /= (2*vol)
    return tor_energy + pol_energy, tor_energy, pol_energy

def compute_mode_product(spec_state, spectral_state, m, field_name = 'velocity'):

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
    spec_state.idx = spec_state.idxlm()
    idxQ = spectral_state.idxlm()

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
