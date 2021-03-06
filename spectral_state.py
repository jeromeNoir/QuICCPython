from base_state import BaseState
import tools
import h5py
import numpy as np
from scipy.fftpack import dct, idct
from numpy.polynomial import chebyshev as cheb
import sys, os
env = os.environ.copy()
sys.path.append(env['HOME']+'/quicc-github/QuICC/Python/')
#get it from crossover_master_release
#from quicc.projection.shell_energy import ortho_pol_q, ortho_pol_s, ortho_tor
import integrateWorland as wor 

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
        x, y = None, None
        if self.geometry == 'shell':
            x = self.make1DGrid('ChebyshevShell', self.specRes.N)
            y = self.make1DGrid('Legendre', self.specRes.L)

        elif self.geometry == 'sphere':
            x = self.make1DGrid('Worland', self.specRes)
            y = self.make1DGrid('Legendre', self.specRes.L)

        elif self.geometry == 'cartesian':
            print('Please use the makeVerticalGrid function for a cartesian geometry')
            pass

        else:
            raise RuntimeError('Meridional grid for unknown geometry')
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
        # set default argument
        x, y = None, None
        if self.geometry == 'shell':
            x = self.make1DGrid('ChebyshevShell', self.specRes.N)
            y = self.make1DGrid('Fourier', self.specRes.M)

        elif self.geometry == 'sphere':
            x = self.make1DGrid('Worland', self.specRes)
            y = self.make1DGrid('Fourier', self.specRes.M)

        elif self.geometry == 'cartesian':
            print('Please use the makeHorizontalGrid function for a cartesian geometry')
            pass

        else:
            raise RuntimeError('Meridional grid for unknown geometry')
        # make the 2D grid via Kronecker product
        R, Phi = np.meshgrid(x, y)
        X = R*np.cos(Phi)
        Y = R*np.sin(Phi)

        return X, Y, x, y

    def makeHorizontalGrid(self):
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
            print('Please use the makeEquatorialGrid function for sphere or shell geometry')
            pass

        elif self.geometry == 'cartesian':
            x = self.make1DGrid('Fourier', self.specRes.kx)
            y = self.make1DGrid('Fourier', self.specRes.ky)

        else:
            raise RuntimeError('Horizontal grid for unknown geometry')
        # make the 2D grid via Kronecker product
        X, Y = np.meshgrid(x, y)

        return X, Y, x, y

    def makeVerticalGrid(self, direction):
        """
        INPUT:
        direction: string, 'x' or 'y', describes the direction we want the slice on
        OUTPUT:
        X: np.matrix; first coordinate on a grid
        Y: np.matrix; second coordinate on a grid
        x: np.array; first grid
        y: np.array; second grid
        """
        # set default argument
        x, y = None, None
        if self.geometry == 'shell' or self.geometry == 'sphere':
            raise RuntimeError('Please use the makeMeridionalGrid function for sphere or shell geometry')
            pass

        elif self.geometry == 'cartesian':
            if direction=='x':
                #'Grid for (z,x)'
                x = self.make1DGrid('Chebyshev', self.specRes.N)
                y = self.make1DGrid('Fourier', self.specRes.kx)
            elif direction == 'y':
                #'Grid for (z,y)'
                x = self.make1DGrid('Chebyshev', self.specRes.N)
                y = self.make1DGrid('Fourier', self.specRes.ky)
            else:
                raise RuntimeError('direction does not make sense - should be either x or y')
        else:
            raise RuntimeError('Horizontal grid for unknown geometry')
        # make the 2D grid via Kronecker product
        X, Y = np.meshgrid(x, y)

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

    def makeHorizontalSlice(file, geometry, field, level):
        """
        INPUT:
        file 
        geometry
        field: string, descriptor for the spectral field wanted e.g. /velocity/velocityz
        level: double, \in [-1,1] to denote the level you want in Z
        OUTPUT:
        real_field: np.matrix, representing the real data of the field
        """

        if geometry == 'shell' or geometry == 'sphere':
            raise NotImplementedError('Please use the makeEquatorialSlice function for a sphere or shell geometry')
            
        elif geometry == 'cartesian':
            my_state = SpectralState(file,geometry)
            spectral_coeff = getattr(my_state.fields,field)
            [nx , ny , nz ] = spectral_coeff.shape
            x = np.array([level])
            PI = tools.cheb_eval(nz, 1.0, 0, x);
            
            spectral_coeff[:,0,:] = 2* spectral_coeff[:,0,:]
            padfield = np.zeros((int((nx+1)*3/2), int((ny+1)*3/2), nz  ), dtype=complex)
            
            padfield[:(int((nx+1)/2)+1), :ny, :] = spectral_coeff[:(int((nx+1)/2)+1),:,:]
            padfield[-(int((nx)/2)):, :ny, :] = spectral_coeff[-(int((nx)/2)):, :, :]
            real_field = np.fft.irfft2(np.dot(padfield, PI.T))
            
            [nx2 , ny2  ] = real_field.shape
            real_field = real_field*nx2*ny2

        else:
            raise RuntimeError('Unknown geometry')

        return real_field

    def makeVerticalSlice(file, geometry, field, direction, level):
        """
            INPUT:
            geometry
            field
            direction = either 'x' or 'y'
            level = should be a value (0, 2pi) you want in the specified direction
            OUTPUT:
            2D array of the slice
            """

        if geometry == 'shell' or geometry == 'sphere':
            raise NotImplementedError('Please use the makeMeridionalSlice function for a sphere or shell geometry')
            pass
        
        elif geometry == 'cartesian':
            my_state = SpectralState(file,geometry)
            spectral_coeff = getattr(my_state.fields,field)
            [nx , ny , nz ] = spectral_coeff.shape

            if direction == 'x':
                PI = tools.fourier_eval(nx, level);

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
                
            elif direction == 'y':
                PI = tools.fourier_eval(2*ny-1, level);
                spectral_coeff[:,0,:] = 2* spectral_coeff[:,0,:]
                spectral_coeff_cc = spectral_coeff[:, :, :].real - 1j*spectral_coeff[:, :, :].imag

                total = np.zeros((int(nx), int(ny*2 -1 ), int(nz)), dtype=complex)
                for i in range(0,ny-1):
                    total[:,ny-1-i,:] = spectral_coeff_cc[:,i+1,:]
                total[:,:(ny),:] = spectral_coeff[:,:,:]

                test = np.zeros((nx,nz), dtype=complex)

                for i in range(0,nz):
                    test[:,i] = np.ndarray.flatten(np.dot(total[:,:,i],PI.T))

                padfield = np.zeros((int((nx+1)*3/2), int(nz*3/2)), dtype=complex)
                #Leo: check with M
                padfield[:int((nx+1)/2)+1,  :nz] = test[:int((nx+1)/2)+1,:]
                padfield[-int(nx/2):, :nz] = test[-int(nx/2):,  :]

                real_field = np.zeros((int((nx+1)*3/2), int(nz*3/2)),  dtype=complex)
                real_field2 = np.zeros((int((nx+1)*3/2), int(nz*3/2)),  dtype=complex)

                for i in range(0, int((nx+1)*3/2)):
                    real_field[i,:] = idct(padfield[i,:])

                for i in range(0,int(nz*3/2)):
                    real_field2[:,i] = np.fft.ifft(real_field[:,i])
                    
                [nx2 , nz2] = real_field2.shape
                real_field2 = real_field2*nx2*2
                    
            else:
                raise RuntimeError('Direction for vertical slice given incorrectly')

        else:
            raise RuntimeError('error in the makeVerticalSlice function')

        return real_field2
    
    def PointValue(self, Xvalue, Yvalue, Zvalue,  field='velocity'):

        if self.geometry == 'shell' or self.geometry == 'sphere':

            # assume that the argument are r=x theta=y and phi=z
            assert(len(Xvalue) == len(Yvalue))
            assert(len(Xvalue) == len(Zvalue))
            r = Xvalue
            theta = Yvalue
            phi = Zvalue
            
            # generate indexer
            # this generate the index lenght also
            self.idx = self.idxlm()
            ridx = {v: k for k, v in self.idx.items()}
            
            # generate grid
            self.makeMeridionalGrid()
            
            # pad the fields
            dataT = np.zeros((self.nModes, self.physRes.nR), dtype='complex')
            dataT[:,:self.specRes.N] = getattr(self.fields, field+'_tor')
            dataP = np.zeros((self.nModes, self.physRes.nR), dtype='complex')
            dataP[:,:self.specRes.N] = getattr(self.fields, field+'_pol')
            
            # prepare the output fields
            FR = np.zeros_like(r)
            FTheta = np.zeros_like(FR)
            FPhi = np.zeros_like(FR)
            FieldOut = [FR, FTheta, FPhi]
            
            # initialize the spherical harmonics
            self.makeSphericalHarmonics(theta)
            x = (r - self.b)/self.a
            for i in range(self.nModes):
                
                # get the l and m of the index
                l, m = ridx[i]
                
                # statement to redute the number of modes considered
            
                self.evaluate_mode(l, m, FieldOut, dataT[i, :], dataP[i,
                                                                      :], r, theta, phi, kron='points', x=x)
                
            return_value =  {'r': r, 'theta': theta, 'phi': phi, 'U_r': FieldOut[0], 'U_theta': FieldOut[1], 'U_phi': FieldOut[2]}
                        
        elif self.geometry == 'cartesian':
            
            spectral_coeff = getattr(self.fields,field)
            [nx , ny , nz ] = spectral_coeff.shape

            spectral_coeff[:,0,:] = 2*spectral_coeff[:,0,:]
            spectral_coeff_cc = spectral_coeff[:, :, :].real - 1j*spectral_coeff[:, :, :].imag
            total = np.zeros((int(nx), int(ny*2 -1 ), int(nz)), dtype=complex)
            
            for i in range(0,ny-1):
                total[:,ny-1-i,:] = spectral_coeff_cc[:,i+1,:]            
            
            total[:,(ny-1):,:] = spectral_coeff[:,:,:]
            [nx2 , ny2 , nz2 ] =total.shape

            Proj_cheb = SpectralState.Cheb_eval(nz2, 1.0, 0, Zvalue)
            Proj_fourier_x = tools.fourier_eval(nx2, Xvalue)
            Proj_fourier_y = tools.fourier_eval(ny2, Yvalue)

            value1 = np.dot(total, Proj_cheb.T)
            value2 = np.dot(value1, Proj_fourier_y.T)
            value3 = np.dot(value2.T,Proj_fourier_x.T )

        return float(2*value3.real)


    # function creating a dictionary to index data for SLFl, WLFl,
    # SLFm or WLFm geometries
    def idxlm(self):

        if self.geometry != 'shell' and self.geometry != 'sphere':
            raise  NotImplementedError('The idxlm dictionary is not implemented for the current geometry', self.geometry)
        
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

        # raise error in case of wrong geometry
        if not (self.geometry == 'shell' or self.geometry == 'sphere'):
            raise NotImplementedError('makeSphericalHarmonics is not implemented for the geometry: '+self.geometry)

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
            temp = tools.plm(self.specRes.L-1, m, x)
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
            temp = tools.plm(self.specRes.L-1, m, x)
            return temp[:, l - m]
        else:
            return self.Plm[self.idx[l, m], :]
        

    def makeMeridionalSlice(self, p=0, modeRes = (120,120) ):
        """
        the function takes care of the looping over modes
        TODO: Nico add comments and change syntax   
        """

        if not (self.geometry == 'shell' or self.geometry == 'sphere'):
            raise NotImplementedError('makeMeridionalSlice is not implemented for the geometry: '+self.geometry)

        # generate indexer
        # this generate the index lenght also
        self.idx = self.idxlm()
        # returns (l,m) from index 
        ridx = {v: k for k, v in self.idx.items()}

        """
        if modeRes == None:
            modeRes=(self.specRes.L, self.specRes.M)
        """
        
        # generate grid
        # X, Y: meshgrid for plotting in cartesian coordinates
        # r, theta: grid used for evaluation 
        X, Y, r, theta = self.makeMeridionalGrid()
        
        # pad the fields to apply 3/2 rule 
        dataT = np.zeros((self.nModes, self.physRes.nR), dtype='complex')
        dataT[:,:self.specRes.N] = self.fields.velocity_tor
        dataP = np.zeros((self.nModes, self.physRes.nR), dtype='complex')
        dataP[:,:self.specRes.N] = self.fields.velocity_pol
        
        # prepare the output fields in radial, theta, and phi 
        FR = np.zeros((len(r), len(theta)))
        FTheta = np.zeros_like(FR)
        FPhi = np.zeros_like(FR)
        FieldOut = [FR, FTheta, FPhi]
                
        # initialize the spherical harmonics
        self.makeSphericalHarmonics(theta)

        for i in range(self.nModes):
            
            # get the l and m of the index
            l, m = ridx[i]

            # statement to redute the number of modes considered
            #if l> modeRes[0] or m> modeRes[1]:
            #continue
            self.evaluate_mode(l, m, FieldOut, dataT[i, :], dataP[i,
                                                                  :], r, theta, None, kron='meridional', phi0=p)

        return {'x': X, 'y': Y, 'U_r': FieldOut[0], 'U_theta': FieldOut[1], 'U_phi': FieldOut[2]}


    # the function takes care of the looping over modes
    def makeEquatorialSlice(self, p=0, modeRes = (120,120) ):

        if not (self.geometry == 'shell' or self.geometry == 'sphere'):
            raise NotImplementedError('makeEquatorialSlice is not implemented for the geometry: '+self.geometry)

        # generate indexer
        # this generate the index lenght also
        self.idx = self.idxlm()
        ridx = {v: k for k, v in self.idx.items()}

        #if modeRes == None:
        #    modeRes=(self.specRes.L, self.specRes.M)
        # generate grid
        X, Y, r, phi = self.makeEquatorialGrid()
        self.grid_r = r
        self.grid_phi = phi
        # pad the fields
        dataT = np.zeros((self.nModes, self.physRes.nR), dtype='complex')
        dataT[:,:self.specRes.N] = self.fields.velocity_tor
        dataP = np.zeros((self.nModes, self.physRes.nR), dtype='complex')
        dataP[:,:self.specRes.N] = self.fields.velocity_pol
        
        # prepare the output fields
        FR = np.zeros((len(r), int(self.physRes.nPhi/2)+1), dtype = 'complex')
        FTheta = np.zeros_like(FR)
        FPhi = np.zeros_like(FR)
        FieldOut = [FR, FTheta, FPhi]
                
        # initialize the spherical harmonics
        # only for the equatorial values
        self.makeSphericalHarmonics(np.array([np.pi/2]))

        for i in range(self.nModes):
            # get the l and m of the index
            l, m = ridx[i]
            # statement to reduce the number of modes considered
            #if l> modeRes[0] or m> modeRes[1]:
            #    continue
            #Core function to evaluate (l,m) modes summing onto FieldOut             
            self.evaluate_mode(l, m, FieldOut, dataT[i, :], dataP[i,:], r, None, phi, kron='equatorial', phi0=p)

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
        #return X, Y, field2
        return {'x': X, 'y': Y, 'U_r': FieldOut[0], 'U_theta': FieldOut[1], 'U_phi': FieldOut[2]}

    # the function takes care of the looping over modes
    def makeIsoradiusSlice(self, r=None, p=0, modeRes = (120,120) ):

        if not (self.geometry == 'shell' or self.geometry == 'sphere'):
            raise NotImplementedError('makeMeridionalSlice is not implemented for the geometry: '+self.geometry)

        # generate indexer
        # this generate the index lenght also
        self.idx = self.idxlm()
        ridx = {v: k for k, v in self.idx.items()}

        """
        if modeRes == None:
            modeRes=(self.specRes.L, self.specRes.M)
        """
        
        # generate grid
        TTheta, PPhi, theta, phi = self.makeIsoradiusGrid()
        self.grid_theta = theta
        self.grid_phi = phi
        # pad the fields
        dataT = np.zeros((self.nModes, self.physRes.nR), dtype='complex')
        dataT[:,:self.specRes.N] = self.fields.velocity_tor
        dataP = np.zeros((self.nModes, self.physRes.nR), dtype='complex')
        dataP[:,:self.specRes.N] = self.fields.velocity_pol

        
        # prepare the output fields
        #FR = np.zeros((len(theta), len(phi)))
        # attempt the Fourier tranform approach
        FR = np.zeros((len(theta), int(self.physRes.nPhi/2)+1), dtype = 'complex')
        FTheta = np.zeros_like(FR)
        FPhi = np.zeros_like(FR)
        FieldOut = [FR, FTheta, FPhi]

        # prepare the "collocation point"
        if r == None:
            x = 0
            r = .5*(self.eta+1)/(1-self.eta)
        else:
            self.a, self.b = .5, .5 * (1+self.eta)/(1-self.eta)
            x = (r - self.b)/self.a
            
            
        # initialize the spherical harmonics
        self.makeSphericalHarmonics(theta)
        
        for i in range(self.nModes):
            
            # get the l and m of the index
            l, m = ridx[i]

            # statement to redute the number of modes considered
            #if l> modeRes[0] or m> modeRes[1]:
            #continue
            self.evaluate_mode(l, m, FieldOut, dataT[i, :], dataP[i,
                                                                  :], r, theta, phi, kron='isogrid', phi0=p, x=x)

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

    def evaluate_mode(self, l, m, *args, **kwargs):
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
        if not (self.geometry == 'shell' or self.geometry == 'sphere'):
            raise NotImplementedError('makeMeridionalSlice is not implemented for the geometry: '+self.geometry)

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

        if kwargs['kron'] == 'isogrid' or kwargs['kron'] == 'points':
            x = kwargs['x']
            if self.geometry == 'shell':
                # assume that the mode is weighted like Philippe sets it
                modeP[1:] *= 2.
                modeT[1:] *= 2.
                # prepare the q_part
                modeP_r = cheb.chebval(x, modeP)/r
                q_part = modeP_r * l*(l+1)
                
                # prepare the s_part
                dP = np.zeros_like(modeP)
                d_temp = cheb.chebder(modeP)
                dP[:len(d_temp)] = d_temp
                s_part = modeP_r + cheb.chebval(x, dP)/self.a
                                
                # prepare the t_part
                t_part = cheb.chebval(x, modeT)
                                
            elif self.geometry == 'sphere':
                #TODO: Leo
                q_part = None
                s_part = None
                t_part = None
        
        else: # hence either meridional or equatorial
            if self.geometry == 'shell':
                # prepare the q_part
                # q = l*(l+1) * Poloidal / r            
                # idct: inverse discrete cosine transform
                # type = 2: type of transform 
                # q = Field_radial 
                # prepare the q_part
                # idct = \sum c_n * T_n(xj)

                modeP_r = idct(modeP, type = 2)/r
                q_part = modeP_r * l*(l+1)
                
                # prepare the s_part
                # s_part = orthogonal to Field_radial and Field_Toroidal
                # s_part = (Poloidal)/r + d(Poloidal)/dr = 1/r d(Poloidal)/dr
                dP = np.zeros_like(modeP)
                d_temp = cheb.chebder(modeP)
                dP[:len(d_temp)] = d_temp
                s_part = modeP_r + idct(dP, type = 2)/self.a
                
                # prepare the t_part
                # t_part = Field_Toroidal                
                t_part = idct(modeT, type = 2)
                
            elif self.geometry == 'sphere':
                #TODO: Leo, Where do you get this???
                #print('modeP:', modeP.shape, modeP) #32
                #print('modeT:', modeT.shape, modeT) #32
                #print('r:',r.shape, r) # 64
                #print('specRes:',self.specRes)
                modeP_r = np.zeros_like(r)
                dP = np.zeros_like(r)
                t_part = np.zeros_like(r)

            for n in range(self.specRes.N):
                modeP_r=modeP_r+modeP[n]*wor.W(n, l, r)/r
                dP = dP+modeP[n]*wor.diffW(n, l, r)
                t_part = t_part + modeT[n]*wor.W(n,l,r)

            q_part =  modeP_r*l*(l+1)#same stuff
            s_part = modeP_r + dP 
        
        # depending on the kron type it changes how 2d data are formed
        # Mapping from qst to r,theta, phi 
        # phi0: azimuthal angle of meridional plane. It's also the phase shift of the flow with respect to the mantle frame, 
        if kwargs['kron'] == 'meridional':
            eimp = np.exp(1j *  m * phi0)
            
            idx_ = self.idx[l, m]
            Field_r += np.real(tools.kron(q_part, self.Plm[idx_, :]) *
                               eimp) * factor
            
            Field_theta += np.real(tools.kron(s_part, self.dPlm[idx_, :]) * eimp) * 2
            Field_theta += np.real(tools.kron(t_part, self.Plm_sin[idx_, :]) * eimp * 1j * m) * 2#factor
            Field_phi += np.real(tools.kron(s_part, self.Plm_sin[idx_, :]) * eimp * 1j * m) * 2#factor 
            Field_phi -= np.real(tools.kron(t_part, self.dPlm[idx_, :]) * eimp) * 2

        elif kwargs['kron'] == 'points':
            eimp = np.exp(1j *  m * phi)
            
            idx_ = self.idx[l, m]
            Field_r += np.real(q_part * self.Plm[idx_, :] * eimp) * factor
            Field_theta += np.real(s_part * self.dPlm[idx_, :] * eimp) * 2
            Field_theta += np.real(t_part * self.Plm_sin[idx_, :] * eimp * 1j * m) * 2#factor
            Field_phi += np.real(s_part * self.Plm_sin[idx_, :] * eimp * 1j * m) * 2#factor 
            Field_phi -= np.real(t_part * self.dPlm[idx_, :] * eimp) * 2

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
            idx_ = self.idx[l, m]
            
            #Field_r += np.real(tools.kron(self.Plm[idx_, :], eimp) * q_part) * factor
            Field_r[:, m] += self.Plm[idx_, :] * q_part

            #Field_theta += np.real(tools.kron(self.dPlm[idx_, :], eimp) * s_part) * 2
            Field_theta[:, m] += self.dPlm[idx_, :] * s_part
            #Field_theta += np.real(tools.kron(self.Plm_sin[idx_, :], eimp * 1j * m) * t_part) * 2#factor
            Field_theta[:, m] += self.Plm_sin[idx_, :] * 1j * m * t_part
            #Field_phi += np.real(tools.kron(self.Plm_sin[idx_, :], eimp * 1j * m) * s_part) * 2#factor
            Field_phi[:, m] += self.Plm_sin[idx_, :]* 1j * m * s_part
            #Field_phi -= np.real(tools.kron(self.dPlm[idx_, :], eimp) * t_part) * 2
            Field_phi[:, m] -= self.dPlm[idx_, :] * t_part
            
        elif kwargs['kron'] == 'isogrid':
            #eimp = np.exp(1j *  m * (phi0 + phi))
            
            idx_ = self.idx[l, m]
            
            #Field_r += np.real(tools.kron(self.Plm[idx_, :], eimp) * q_part) * factor
            Field_r[:, m] += self.Plm[idx_, :] * q_part

            #Field_theta += np.real(tools.kron(self.dPlm[idx_, :], eimp) * s_part) * 2
            Field_theta[:, m] += self.dPlm[idx_, :] * s_part
            #Field_theta += np.real(tools.kron(self.Plm_sin[idx_, :], eimp * 1j * m) * t_part) * 2#factor
            Field_theta[:, m] += self.Plm_sin[idx_, :] * 1j * m * t_part
            #Field_phi += np.real(tools.kron(self.Plm_sin[idx_, :], eimp * 1j * m) * s_part) * 2#factor
            Field_phi[:, m] += self.Plm_sin[idx_, :]* 1j * m * s_part
            #Field_phi -= np.real(tools.kron(self.dPlm[idx_, :], eimp) * t_part) * 2
            Field_phi[:, m] -= self.dPlm[idx_, :] * t_part
            
        else:
            raise ValueError('Kron type not understood: '+kwargs['kron'])

        pass
                
            
                

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
            
 
