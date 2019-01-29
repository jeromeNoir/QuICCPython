import h5py
import numpy as np

class BaseState:
    
    def __init__(self, filename, geometry, file_type='QuICC'):
        
        # PhysicalState('state0000.hdf5',geometry='Sphere'
        fin = h5py.File(filename, 'r')
        self.fin = fin

        
        attrs = list(self.fin.attrs.keys())
        
        if attrs[2] == "Version" and file_type.lower()!= 'quicc':
            raise RunTimeError("maybe your file is EPM")
            
        if attrs[1] == "version" and file_type.lower()!= 'epm':
            raise RunTimeError("maybe your file is QuICC")
                            
        # TODO: distinguish between EPM and QuICC
        self.isEPM = False
        
        if file_type.lower()!='quicc':
            self.isEPM = True
        # assuming using QuICC state syntax
           
            
        
        # initialize the .parameters object
        self.parameters = lambda: None
                   
        # hardcode set time and timestep
        if not self.isEPM:
            setattr(self.parameters, 'time', fin['run/time'].value)
            setattr(self.parameters, 'timestep', fin['run/timestep'].value)
            for at in fin['physical'].keys():
                setattr(self.parameters, at, fin['physical'][at].value)
        else:
            setattr(self.parameters, 'time', fin['RunParameters/Time'].value)
            setattr(self.parameters, 'timestep', fin['RunParameters/Step'].value)
        
        # import attributes

        
        #self.geometry = fin.attrs['type']
        self.geometry = geometry.lower()
        if(self.geometry not in list(['cartesian', 'shell', 'sphere'])):
            raise RuntimeError("I'm sorry but we only deal with Cartesian, Shell and Sphere, try again later")
        # geometry so far is
    pass

class PhysicalState(BaseState):
    
    
    def __init__(self, filename, geometry, file_type='QuICC'):
        
        # apply the read of the base class
        BaseState.__init__(self, filename, geometry, file_type='QuICC')
        fin = self.fin
        # read the grid
        if not self.isEPM:
            for at in fin['mesh']:
                setattr(self, at, fin['mesh'][at].value)
            
            # find the fields
            self.fields = lambda:None
            #temp = object()
            for group in fin.keys():

                for subg in fin[group]:

                    # we check to import fields which are at least 2-dimensional tensors
                    field = np.array(fin[group][subg])
                    if len(field.shape)<2:
                        continue

                    # set attributes
                    setattr(self.fields, subg, field)
                    
        else:
            for at in fin['FSOuter/grid']:
                setattr(self, at, fin['FSOuter/grid'][at].value)
            
            # find the fields
            self.fields = lambda:None
            #temp = object()
            for group in fin['FSOuter'].keys():

                for subg in fin['FSOuter'][group]:

                    # we check to import fields which are at least 2-dimensional tensors
                    field = np.array(fin['FSOuter'][group][subg])
                    if len(field.shape)<2:
                        continue

                    # set attributes
                    setattr(self.fields, subg, field)
                
        
class SpectralState(BaseState):
    
    def __init__(self, filename, geometry, file_type='QuICC'):
                
        # apply the read of the base class
        BaseState.__init__(self, filename, geometry, file_type='QuICC')
        fin = self.fin
            
        # find the spectra
        self.fields = lambda:None
        
        for group in fin.keys():

            for subg in fin[group]:

                # we check to import fields which are at least 2-dimensional tensors
                print(fin[group][subg])
                field = fin[group][subg]
                
                if isinstance(field, h5py.Group):
                    continue
                if len(field.value.shape)<2:
                    continue
                    
                field_temp = field[:]

                if self.geometry == 'sphere' or self.geometry == 'shell':
                    field = np.array(field_temp[:,:,0]+1j*field_temp[:,:,1])

                else:
                    field = np.array(field_temp[:, :, :, 0] + 1j*field_temp[:, :, :, 1])
                    
                
                # set attributes
                if self.isEPM:
                    # cast the subgroup to lower and transforms e.g.
                    # VelocityTor to velocity_tor
                    subg = subg.lower()
                    tmplist = list(subg)
                    tmplist.insert(-3,'_')
                    subg = ''.join(tmplist)
                
                setattr(self.fields, subg, field)
                
        if self.isEPM:
            for at in fin['PhysicalParameters'].keys():
                setattr(self.parameters, at, fin['PhysicalParameters'][at].value)

        self.readResolution(fin)
        
    def readResolution(self, fin):
        """
        This function sets the geometry specific attributes for the
        spectral resoltion
        INPUT:
        fin: h5py object; the  current in reading hdf5 buffer
        OUTPUT:
        None
        """
        # init the self.specRes object
        self.specRes = lambda: None
        if self.geometry == 'shell' or self.geometry == 'sphere':
            # read defined resolution
            N = fin['/truncation/spectral/dim1D'].value + 1
            L = fin['/truncation/spectral/dim2D'].value + 1
            M = fin['/truncation/spectral/dim3D'].value + 1
            setattr(self.specRes, 'N', N)
            setattr(self.specRes, 'L', L)
            setattr(self.specRes, 'M', M)
        elif self.geometry == 'cartesian':
            #TODO: Meredith, implement the names for spectral resolution
            pass

        # init the self.physRes object for future use
        self.physRes = lambda: None

        #TODO: Leo, implement the logic for EPM files

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
            physRes = 3*specRes
            grid = np.linspace(0,2*np.pi, physRes + 1)
            
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
            physRes = int(3/2 * specRes) + 12
            # retrieve the a and b parameters from the shell aspect
            # ratio
            eta = self.parameters.rratio
            a, b = .5, .5 * (1+eta)/(1-eta)
            x, w = np.polynomial.chebyshev.chebgauss(physRes)
            grid = x*a + b
            
        elif gridType == 'legendre' or gridType == 'l':
            # applying 3/2 rule
            physRes = int(3/2 * specRes)
            grid = np.arccos(np.polynomial.legendre.leggauss(physRes)[0])
            
        elif gridType == 'worland' or gridType == 'w':
            #TODO: Leo, check what the relation between physRes and
            # specRes this should be about 2*specRes Philippes thesis
            # 3/2 N + 3/4 L + 1
            physRes = specRes 
            nr = physRes
            grid = np.sqrt((np.cos(np.pi*(np.arange(0,2*nr)+0.5)/(2*nr)) + 1.0)/2.0)
        
        else:
            raise ValueError("Defined types are Fourier, Legendre, Chebyshev, ChebyshevShell and Worland")
            grid = 0
            
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
            x = self.make1DGrid('Worland', self.specRes.N)
            y = self.make1DGrid('Legendre', self.specRes.L)

        elif self.geometry == 'cartesian':
            #TODO: Meredith, implement the name of the 2 spectral
            #resolution needed
            x = self.make1DGrid(..., ...)
            y = self.make1DGrid(..., ...)

        else:
            raise RuntimeError('Meridional grid for unknown geometry')
        # make the 2D grid via Kronecker product
        X, Y = np.meshgrid(x, y)

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
            x = self.make1DGrid('Worland', self.specRes.N)
            y = self.make1DGrid('Fourier', self.specRes.M)

        elif self.geometry == 'cartesian':
            #TODO: Meredith, implement the name of the 2 spectral
            #resolution needed
            x = self.make1DGrid(..., ...)
            y = self.make1DGrid(..., ...)

        else:
            raise RuntimeError('Meridional grid for unknown geometry')
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

        elif self.geometry == 'cartesian':
            raise NotImplementedError('Iso-radius grids are not possible for Cartesian geometries')
                        

        else:
            raise RuntimeError('Meridional grid for unknown geometry')
        # make the 2D grid via Kronecker product
        X, Y = np.meshgrid(x, y)

        return X, Y, x, y

