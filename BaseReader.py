import h5py
import numpy as np

    
class BaseState:
    
    def __init__(self, filename, geometry, file_type='QuICC'):
        
        # PhysicalState('state0000.hdf5',geometry='Sphere'
        fin = h5py.File(filename, 'r')
        self.fin = fin

        # TODO: distinguish between EPM and QuICC
        if file_type.lower()!='quicc':
            self.isEPM = True
        # assuming using QuICC state syntax
        
        attrs = list(self.fin.attrs.keys())
        if attrs[1] == 'version':
            self.dtype='QuICC'
        elif attrs[2] == 'Version':
            self.dtype='EPM'       
            
        
        # initialize the .parameters object
        self.parameters = lambda: None
                   
        # hardcode set time and timestep
        setattr(self.parameters, 'time', fin['run/time'].value)
        setattr(self.parameters, 'timestep', fin['run/timestep'].value)
        
        # import attributes
        
        for at in fin['physical'].keys():
            
            setattr(self.parameters, at, fin['physical'][at].value)
        
        #self.geometry = fin.attrs['type']
        self.geometry = geometry
        # geometry so far is
    pass

class PhysicalState(BaseState):
    
    
    def __init__(self):
        
        # apply the read of the base class
        BaseState.__init__(self,filename)
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
    
    def __init__(self):
                
        # apply the read of the base class
        BaseState.__init__(self,filename)
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
                
                field = np.array(field_temp[:,:,0]+1j*field_temp[:,:,1])
                                                               
                
                # set attributes
                setattr(self.fields, subg, field)
                
