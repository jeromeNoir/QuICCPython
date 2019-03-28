import h5py
import numpy as np
from numpy.polynomial import chebyshev as cheb

class BaseState:
    
    def __init__(self, filename, geometry, file_type='QuICC'):
        
        # PhysicalState('state0000.hdf5',geometry='Sphere'
        fin = h5py.File(filename, 'r')
        self.fin = fin

        # check that the file is QuICC generated
        attrs = list(self.fin.attrs.keys())
        #assert (attrs[1] == "version"), 'File structure not compatible with QuICC, possible EPM file'
                            
        # initialize the .parameters object
        self.parameters = lambda: None
                   
        # hardcode set time and timestep
        setattr(self.parameters, 'time', fin['run/time'].value)
        setattr(self.parameters, 'timestep', fin['run/timestep'].value)
        for at in fin['physical'].keys():
            setattr(self.parameters, at, fin['physical'][at].value)
        
        # store the geometry string as a lower case (makes it possible to compare, compare is always in lowercase)
        self.geometry = geometry.lower()
        if(self.geometry not in list(['cartesian', 'shell', 'sphere'])):
            raise RuntimeError("Only Cartesian, Shell and Sphere geometries are currently supported for QuICC generated files.")
        # geometry so far is
    pass


class PhysicalState(BaseState):
    
    def __init__(self, filename, geometry, file_type='QuICC'):
        
        # apply the read of the base class
        BaseState.__init__(self, filename, geometry, file_type='QuICC')
        fin = self.fin
        # read the grid

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
                    

class SpectralState(BaseState):
    
    def __init__(self, filename, geometry, file_type='QuICC'):
                
        # apply the read of the base class
        BaseState.__init__(self, filename, geometry, file_type='QuICC')
        fin = self.fin
            
        # initialize the data fields
        self.fields = lambda:None

        # loop over each group/subgroup of the hdf5 file
        for group in fin.keys():
            for subg in fin[group]:

                # we check to import fields which are at least 2-dimensional tensors
                field = fin[group][subg]

                # skip  the instances which are not groups
                if isinstance(field, h5py.Group):
                    continue
                
                # skip the groups which are not at least matrices
                if len(field.value.shape)<2:
                    continue
                    
                field_temp = field[:]

                if self.geometry == 'sphere' or self.geometry == 'shell':

                    if field.dtype=='complex128':
                        field = np.array(field_temp, dtype = 'complex128')
                    else:
                        field = np.array(field_temp[:,:,0]+1j*field_temp[:,:,1])

                else:
                    field = np.array(field_temp[:, :, :, 0] + 1j*field_temp[:, :, :, 1])
                    
                # set attributes
                setattr(self.fields, subg, field)
                
        self.readResolution()
        
    def readResolution(self):
        """
        This function sets the geometry specific attributes for the
        spectral resoltion, it gets the information from the self.fin 
        object.
        INPUT:
        None
        OUTPUT:
        None
        """
        # init the self.specRes object
        self.specRes = lambda: None
        if self.geometry == 'shell' or self.geometry == 'sphere':

            # read defined resolution
            N = self.fin['/truncation/spectral/dim1D'].value + 1
            L = self.fin['/truncation/spectral/dim2D'].value + 1
            M = self.fin['/truncation/spectral/dim3D'].value + 1
            setattr(self.specRes, 'N', N)
            setattr(self.specRes, 'L', L)
            setattr(self.specRes, 'M', M)

            # with the resolution read the ordering type
            setattr(self, 'ordering', self.fin.attrs['type'])
            
        elif self.geometry == 'cartesian':

            N = self.fin['/truncation/spectral/dim1D'].value + 1
            kx = self.fin['/truncation/spectral/dim2D'].value + 1
            ky = self.fin['/truncation/spectral/dim3D'].value + 1
            setattr(self.specRes, 'N', N)
            setattr(self.specRes, 'kx', kx)
            setattr(self.specRes, 'ky', ky)
            setattr(self.specRes, 'kx', kx)

        else:
            raise NotImplementedError('Geometry unknown: ', self.geometry)
        
        # init the self.physRes object for future use
        self.physRes = lambda: None

