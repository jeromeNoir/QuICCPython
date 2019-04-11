import EpmPython.sphere as sphere
import h5py
import numpy as np
from scipy.fftpack import dct, idct
from numpy.polynomial import chebyshev as cheb
import sys, os
#env = os.environ.copy()
#LEO: remove all QuICC dependency
#sys.path.append(env['HOME']+'/quicc-github/QuICC/Python/')
import EpmPython.sphere.spectral as spectral
#get it from crossover_master_release
#from quicc.projection.shell_energy import ortho_pol_q, ortho_pol_s, ortho_tor
#import integrateWorland as wor 

class BaseState:
    def __init__(self, filename): #, geometry, file_type='QuICC'):
        self.filename = filename 

        # PhysicalState('state0000.hdf5',geometry='Sphere'
        fin = h5py.File(filename, 'r')
        self.fin = fin

        attrs = list(self.fin.attrs.keys())
       
        if attrs[1] == "version": # and file_type.lower()!= 'epm':
            raise RunTimeError("maybe your file is QuICC")
                            
        #fin.close() 
    pass

class SpectralState(BaseState):
    
    def __init__(self, filename): # , geometry, file_type='QuICC'):
                
        # apply the read of the base class
        BaseState.__init__(self, filename) #:, geometry, file_type=file_type)
        fin = self.fin
        
        # initialize the .parameters object
        self.parameters = lambda: None
        setattr(self.parameters, 'time', fin['RunParameters/Time'][()])
        setattr(self.parameters, 'timestep', fin['RunParameters/Step'][()])

        # find the spectra
        self.fields = lambda:None
        
        for group in fin.keys():

            for subg in fin[group]:

                # we check to import fields which are at least 2-dimensional tensors
                field = fin[group][subg]
                
                if isinstance(field, h5py.Group):
                    continue
                if len(field[()].shape)<2:
                    continue
                    
                field_temp = field[:]

                #if self.geometry == 'sphere' or self.geometry == 'shell':

                if field.dtype=='complex128':
                    field = np.array(field_temp, dtype = 'complex128')
                else:
                    field = np.array(field_temp[:,:,0]+1j*field_temp[:,:,1])

                #else:
                #    field = np.array(field_temp[:, :, :, 0] + 1j*field_temp[:, :, :, 1])
                    
                
                # set attributes
                #if self.isEPM:
                # cast the subgroup to lower and transforms e.g.
                # VelocityTor to velocity_tor
                # TODO: fix for codensity and mag_field
                subg = subg.lower()
                tmplist = list(subg)
                tmplist.insert(-3,'_')
                subg = ''.join(tmplist)
                
                setattr(self.fields, subg, field)
                
        #if self.isEPM:
        for at in fin['PhysicalParameters'].keys():
            setattr(self.parameters, at, fin['PhysicalParameters'][at][()])

        self.readResolution(fin)

        fin.close() 
        
        
    def readResolution(self, fin):
        """
        This function sets the geometry specific attributes for the
        spectral resolution
        INPUT:
        fin: h5py object; the  current in reading hdf5 buffer
        OUTPUT:
        None
        """
        # init the self.specRes object
        self.specRes = lambda: None
        
        # read defined resolution
        N = fin['/Truncation/N'][()] + 1
        L = fin['/Truncation/L'][()] + 1
        M = fin['/Truncation/M'][()] + 1
        setattr(self.specRes, 'N', N)
        setattr(self.specRes, 'L', L)
        setattr(self.specRes, 'M', M)

        #ordering 
        setattr(self, 'ordering', b'WLFl')

        # init the self.physRes object for future use
        self.physRes = lambda: None

        # end of function
        pass
    
    def make1DGrid(self, gridType, specRes):
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
            self.physRes.nPhi = 3*specRes
            grid = np.linspace(0,2*np.pi, self.physRes.nPhi + 1)
            
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
            self.physRes.nR = int(3/2 * specRes) + 12
            # retrieve the a and b parameters from the shell aspect
            # ratio
            self.eta = self.parameters.rratio
            self.a, self.b = .5, .5 * (1+self.eta)/(1-self.eta)
            x, w = np.polynomial.chebyshev.chebgauss(self.physRes.nR)
            grid = x*self.a + self.b
            
        elif gridType == 'legendre' or gridType == 'l':
            # applying 3/2 rule
            self.physRes.nTh = int(3/2 * specRes)
            grid = np.arccos(np.polynomial.legendre.leggauss(self.physRes.nTh)[0])
            
        elif gridType == 'worland' or gridType == 'w':
            #TODO: Leo, check what the relation between physRes and
            # specRes this should be about 2*specRes Philippes thesis
            # 3/2 N + 3/4 L + 1
            # e.g: N=10, L=10, nR = 29; N=10, L=20, nR = 36
            # 3*(N+1)//2 + 1 + 3*(L+1)//4 + 1  + 3 = 29
            # self.physRes.nR = specRes
            self.physRes.nR = (3*(specRes.N+1))//2+1 + 3*(specRes.L+1)//4+1 + 3
            nr = self.physRes.nR
            grid = np.sqrt((np.cos(np.pi*(np.arange(0,2*nr)+0.5)/(2*nr)) + 1.0)/2.0)
        
        else:
            raise ValueError("Defined types are Fourier, Legendre, Chebyshev, ChebyshevShell and Worland")
            grid = None
            
        return grid

    def makeMeridionalGrid(self):
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
        #x, y = None, None
        #if self.geometry == 'shell':
        #    x = self.make1DGrid('ChebyshevShell', self.specRes.N)
        #    y = self.make1DGrid('Legendre', self.specRes.L)

        #elif self.geometry == 'sphere':
        x = self.make1DGrid('Worland', self.specRes)
        y = self.make1DGrid('Legendre', self.specRes.L)

        #elif self.geometry == 'cartesian':
        #    print('Please use the makeVerticalGrid function for a cartesian geometry')
        #    pass

        #else:
        #    raise RuntimeError('Meridional grid for unknown geometry')
        # make the 2D grid via Kronecker product
        R, Th = np.meshgrid(x, y)

        X = R* np.sin(Th)
        Y = R* np.cos(Th)
        return X, Y, x, y


    def makeEquatorialGrid(self):
        """
        INPUT:
        None
        OUTPUT:
        X: np.matrix; first coordinate on a grid
        Y: np.matrix; second coordinate on a grid
        x: np.array; first grid
        y: np.array; second grid
        """

        x = self.make1DGrid('Worland', self.specRes)
        y = self.make1DGrid('Fourier', self.specRes.M)

        # make the 2D grid via Kronecker product
        R, Phi = np.meshgrid(x, y)
        X = R*np.cos(Phi)
        Y = R*np.sin(Phi)

        return X, Y, x, y

    def makeIsoradiusGrid(self):
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
        x, y = None, None
        if self.geometry == 'shell' or self.geometry == 'sphere':
            x = self.make1DGrid('Legendre', self.specRes.L)
            y = self.make1DGrid('Fourier', self.specRes.M)

            # necessary for matters of transforms (python need to know nR)
            self.make1DGrid('ts', self.specRes.N)
        elif self.geometry == 'cartesian':
            raise NotImplementedError('Iso-radius grids are not possible for Cartesian geometries')
                        

        else:
            raise RuntimeError('Meridional grid for unknown geometry')
        # make the 2D grid via Kronecker product
        X, Y = np.meshgrid(x, y)

        return X, Y, x, y    

    # function creating a dictionary to index data for SLFl, WLFl,
    # SLFm or WLFm geometries
    def idxlm(self):

        # initialize an empty dictionary
        idxlm = {}

        # initialize the index counter to 0
        ind = 0

        # decide if 'SLFl' or 'SLFm'
        if self.ordering == b'SLFl' or self.ordering == b'WLFl':
            for l in range(self.specRes.L):
                for m in range(min(l + 1, self.specRes.M)):
                    idxlm[(l,m)] = ind
                    ind += 1

        elif self.ordering == b'SLFm' or self.ordering == b'WLFm':
            for m in range(self.specRes.M):
                for l in range(m, self.specRes.L):
                    idxlm[(l,m)] = ind
                    ind += 1
        self.nModes = ind
        return idxlm

    def makeSphericalHarmonics(self, theta):
        # change the theta into x
        x = np.cos(theta)
        self.xth = x

        # initialize storage to 0
        self.Plm = np.zeros((self.nModes, len(x)))
        self.dPlm = np.zeros_like(self.Plm)
        self.Plm_sin = np.zeros_like(self.Plm)

        # generate the reverse indexer
        ridx = {v: k for k, v in self.idx.items()}
        for m in range(self.specRes.M):
            
            # compute the assoc legendre
            temp = sphere.plm(self.specRes.L-1, m, x)
            #temp = wor.plm(self.specRes.L-1, m, x) #Leo: this implementation doesn't work 

            # assign the Plm to storage
            for l in range(m, self.specRes.L):
                self.Plm[self.idx[l, m], :] = temp[:, l-m]
                pass
            
            pass

        # compute dPlm and Plm_sin
        for i in range(self.nModes):
            l, m = ridx[i]
            self.dPlm[i, :] = -.5 * (((l+m)*(l-m+1))**0.5 * self.plm(l,m-1) -
                                     ((l-m)*(l+m+1))**.5 * self.plm(l, m+1, x) )

            if m!=0:
                self.Plm_sin[i, :] = -.5/m * (((l-m)*(l-m-1))**.5 *
                                              self.plm(l-1, m+1, x) + ((l+m)*(l+m-1))**.5 *
                                            self.plm(l-1, m-1)) * ((2*l+1)/(2*l-1))**.5
            else:
                self.Plm_sin[i, :] = self.plm(l, m)/(1-x**2)**.5

        pass

    # Lookup function to help the implementation of dPlm and Plm_sin 
    def plm(self, l, m, x = None):

        if l < m or m < 0 or l < 0:
            return np.zeros_like(self.Plm[0,:])
        elif m > self.specRes.M - 1:
            return np.zeros_like(self.Plm[0, :])
            #temp = np.sqrt((2.0*l+1)/(l-m)) * np.sqrt((2.0*l-1.0)/(l+m))*self.xth*self.plm(l-1,m)-\
                #np.sqrt((2.0*l+1)/(2.0*l-3.0))*np.sqrt((l+m-1.0)/(l+m))*np.sqrt((l-m-1.0)/(l-m))\
                #* self.plm(l-2, m)
            temp = sphere.plm(self.specRes.L-1, m, x)
            return temp[:, l - m]
        else:
            return self.Plm[self.idx[l, m], :]
        

    def getMeridionalSlice(self, p=0, modeRes = (120,120) ):
        """
        the function takes care of the looping over modes
        TODO: Nico add comments and change syntax   
        """
        
        merSlice = spectral.getMeridionalSlice(self, p, modeRes)

        return merSlice 

    # the function takes care of the looping over modes
    def getEquatorialSlice(self, p=0, modeRes = (120,120) ):

        eqSlice = spectral.getEquatorialSlice(self, p, modeRes)

        return eqSlice

    # TODO: Leo port this to spectral.py  
    # The function compute_energy makes use of orthogonal energy norms
    # to compute the energy of a field
    def compute_energy(self, field_name='velocity'):
        # INPUT:
        # field_name: string, specifies the vector field type (velocity, magnetic) 
        # switch over different geometries
        # potential: symmetric and antisymmetric

        if self.geometry == 'shell':

            # init the storage
            tor_energy = 0
            pol_energy = 0

            # precompute the radial tranform parameters
            # r = a*x + b
            a, b = .5, .5 * (1 + self.parameters.rratio) / (1 - self.parameters.rratio)

            # compute volume for scaling 
            ro = 1/(1-self.parameters.rratio)
            ri = self.parameters.rratio/(1-self.parameters.rratio)
            vol = 4/3*np.pi*(ro**3-ri**3)
            
            # generate idx indicer
            self.idx = self.idxlm()

            # obtain the 2 fields all modes 
            Tfield = getattr(self.fields, field_name + '_tor')
            Pfield = getattr(self.fields, field_name + '_pol')

            # loop first over modes
            for l in range(self.specRes.L):

                for m in range(min(l+1, self.specRes.M) ):

                    # compute factor because we compute only m>=0
                    # TODO: Nico 2 for m==0 and 1 for m>0?
                    factor = 2. if m==0 else 1.

                    # obtain (l,m) modes
                    Tmode = Tfield[self.idx[l, m], :]
                    Pmode = Pfield[self.idx[l, m], :]

                    #(\sqrt{l*(l+1)})^2: energy norm prefactor for Spherical Harmonics
                    #ortho_tor(length, a, b, Field.real, Field.real) : computes the inner product (Field.real,Field.real)
                    #t, q, s : energy
                    # real and imag are done separately, 
                    # it should be equivalent to compute inner product with complex conjugate 
                    tor_energy += factor * l*(l+1) *\
                    ortho_tor(len(Tmode), a, b, Tmode.real,
                              Tmode.real)
                    tor_energy += factor * l*(l+1)*\
                    ortho_tor(len(Tmode), a, b, Tmode.imag,
                              Tmode.imag)

                    #(l*(l+1))^2: integral over r of 1/r \mathcal{L}^2 P 
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
            return tor_energy, pol_energy, tor_energy + pol_energy
                    

    def compute_mode_product(self, spectral_state, m, field_name='velocity'):
        #TODO: Nico, add description 
        # switch over different geometries
        # \int_V q m\dagger \cdot u dV 
        # projection of velocity field onto a given mode with respect to a given mode
        if self.geometry == 'shell':
            
            # init the storage
            tor_product = 0
            pol_product = 0
            
            # precompute the radial tranform parameters
            a, b = .5, .5 * (1 + self.parameters.rratio) / (1 - self.parameters.rratio)
            
            # compute volume
            ro = 1/(1-self.parameters.rratio)
            ri = self.parameters.rratio/(1-self.parameters.rratio)
            vol = 4/3*np.pi*(ro**3-ri**3)
            
            # generate idx indicer
            self.idx = self.idxlm()
            idxQ = spectral_state.idxlm()

            # obtain the 2 fields
            Tfield = getattr(self.fields, field_name + '_tor')
            Pfield = getattr(self.fields, field_name + '_pol')

            # obtain the 2 fields
            QTfield = getattr(spectral_state.fields, field_name + '_tor')
            QPfield = getattr(spectral_state.fields, field_name + '_pol')

            # loop first over f
            for l in range(m, self.specRes.L):
                
                # compute factor
                factor = 2. if m==0 else 1.
                
                # obtain modes
                Tmode = Tfield[self.idx[l, m], :]
                Pmode = Pfield[self.idx[l, m], :]
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
            
class PhysicalState(BaseState):
    def __init__(self, filename):#, geometry, file_type='QuICC'):
        
        # apply the read of the base class
        BaseState.__init__(self, filename)#, geometry, file_type='QuICC')
        fin = self.fin
        # read the grid
        for at in fin['FSOuter/grid']:
            setattr(self, at.replace(' ', '_'), fin['FSOuter/grid'][at][:])
            
        # find the fields
        self.fields = lambda:None
        #temp = object()
        for group in fin['FSOuter'].keys():
            if group == 'Codensity':
                field = np.array(fin['FSOuter'][group][:])
                setattr(self.fields, group.lower(), field)
                continue

            for subg in fin['FSOuter'][group]:
                # we check to import fields which are at least 2-dimensional tensors
                field = np.array(fin['FSOuter'][group][subg])
                if len(field.shape)<2:
                    continue

                # set attributes
                setattr(self.fields, subg.lower(), field)

        fin.close()


    # define function for equatorial plane visualization from the visState0000.hdf5 file
    def makeMeridionalSlice(self, phi=None, fieldname = 'velocity'):

        # some parameters just in case
        eta = self.parameters.rratio
        ri = eta/(1-eta)
        ro = 1/(1-eta)
        a, b = .5, .5*(1+eta)/(1-eta)

        # find the grid in radial and meridional direction
        r = self.grid_r #.value[idx_r]
        theta =  self.grid_theta
        
        rr, ttheta = np.meshgrid(self.grid_r, self.grid_theta)
        X = rr*np.sin(ttheta)
        Y = rr*np.cos(ttheta)

        if phi == None:
            # select the 0 meridional plane value for 
            idx_phi0 = (self.grid_phi==0)
            
            Field1 = np.mean(getattr(self.fields, fieldname+'r')[:, :, idx_phi0], axis=2)
            Field2 = np.mean(getattr(self.fields, fieldname+'t')[:, :, idx_phi0], axis=2)
            Field3 = np.mean(getattr(self.fields, fieldname+'p')[:, :, idx_phi0], axis=2)
        else:
            phi = self.grid_phi
            Field1 = interp1d(phi, getattr(self.fields, fieldname+'r'), axis=2)(phi)
            Field2 = interp1d(phi, getattr(self.fields, fieldname+'t'), axis=2)(phi)
            Field3 = interp1d(phi, getattr(self.fields, fieldname+'p'), axis=2)(phi)
            
        return X, Y, [Field1, Field2, Field3]


    
    # define function for equatorial plane visualization from the visState0000.hdf5 file
    def makeEquatorialSlice(self, fieldname = 'velocity'):
                        
        # select the r grid in the bulk of the flow
        #idx_r = (fopen['mesh/grid_r']>ri+delta) & (fopen['mesh/grid_r']<ro-delta)
        #idx_r = (fopen['mesh/grid_r'].value>0.)
        
        # select the 0 meridional plane value for
        theta_grid = self.grid_theta
        neq = int(len(theta_grid)/2)
        if len(theta_grid) % 2 == 0:
            idx_theta = (theta_grid == theta_grid[neq-1]) | (theta_grid == theta_grid[neq])
            print(theta_grid[neq-1]-np.pi/2, theta_grid[neq]-np.pi/2)
        else:
            idx_theta = (theta_grid == theta_grid[neq])
            
        # find the grid in radial and meridional direction
        #r = fopen['mesh/grid_r'].value[idx_r]
        r = self.grid_r
        phi = self.grid_phi
        
        rr, pphi = np.meshgrid(r, phi)
        # compute the values for x and y
        X = np.cos(pphi)*rr
        Y = np.sin(pphi)*rr
        
        Field1 = np.mean(getattr(self.fields, fieldname+'r')[:,idx_theta,:], axis=1)
        Field2 = np.mean(getattr(self.fields, fieldname+'t')[:,idx_theta,:], axis=1)
        Field3 = np.mean(getattr(self.fields, fieldname+'p')[:,idx_theta,:], axis=1)

        return X, Y, [Field1, Field2, Field3]

    def PointValue(file, geometry, field, Xvalue, Yvalue, Zvalue):
        if geometry == 'shell' or geometry == 'sphere':
            print('This is not finished yet!')
            pass

        elif geometry == 'cartesian':
            my_state = SpectralState(file,geometry)
            spectral_coeff = getattr(my_state.fields,field)
            [nx , ny , nz ] = spectral_coeff.shape

            spectral_coeff[:,0,:] = 2*spectral_coeff[:,0,:]
            spectral_coeff_cc = spectral_coeff[:, :, :].real - 1j*spectral_coeff[:, :, :].imag
            total = np.zeros((int(nx), int(ny*2 -1 ), int(nz)), dtype=complex)
            
            for i in range(0,ny-1):
                total[:,ny-1-i,:] = spectral_coeff_cc[:,i+1,:]            
            
            total[:,(ny-1):,:] = spectral_coeff[:,:,:]
            [nx2 , ny2 , nz2 ] =total.shape

            Proj_cheb = SpectralState.Cheb_eval(nz2, 1.0, 0, Zvalue)
            Proj_fourier_x = sphere.fourier_eval(nx2, Xvalue)
            Proj_fourier_y = sphere.fourier_eval(ny2, Yvalue)

            value1 = np.dot(total, Proj_cheb.T)
            value2 = np.dot(value1, Proj_fourier_y.T)
            value3 = np.dot(value2.T,Proj_fourier_x.T )

        return float(2*value3.real)
