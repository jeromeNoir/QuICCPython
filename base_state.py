import h5py
import numpy as np
from numpy.polynomial import chebyshev as cheb

class BaseState:
    
    def __init__(self, filename, geometry, file_type='QuICC'):
        
        # PhysicalState('state0000.hdf5',geometry='Sphere'
        fin = h5py.File(filename, 'r')
        self.fin = fin

        
        attrs = list(self.fin.attrs.keys())
       
        if len(attrs) > 2 : 
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
            setattr(self.parameters, 'time', fin['run/time'][()])
            setattr(self.parameters, 'timestep', fin['run/timestep'][()])
            for at in fin['physical'].keys():
                setattr(self.parameters, at, fin['physical'][at].value)
        else:
            setattr(self.parameters, 'time', fin['RunParameters/Time'][()])
            setattr(self.parameters, 'timestep', fin['RunParameters/Step'][()])
        
        # import attributes

        
        #self.geometry = fin.attrs['type']
        self.geometry = geometry.lower()
        if(self.geometry not in list(['cartesian', 'shell', 'sphere'])):
            raise RuntimeError("I'm sorry but we only deal with Cartesian, Shell and Sphere, try again later")
        # geometry so far is
    pass
        
