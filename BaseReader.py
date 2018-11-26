import h5py
import numpy as np




"""
class BaseReader:
    
    ...
    
    pass
"""
    
class BaseState:
    
    def __init__(self):
        pass
    
    def read(self, filename):
    
        fin = h5py.File(filename, 'r')
        self.fin = fin

        # TODO: distinguish between EPM and QuICC
        # assuming using QuICC state syntax
        
        self.dtype='QuICC'
                   
        # hardcode set time and timestep
        setattr(self, 'time', fin['run/time'].value)
        setattr(self, 'timestep', fin['run/timestep'].value)
        
        # import attributes
        
        for at in fin['physical'].keys():
            
            setattr(self, at, fin['physical'][at].value)
        
        self.geometry = fin.attrs['type']
    pass

class ConvertedState(BaseState):
    
    
    def __init__(self):
        BaseState.__init__(self)
        pass
    
    def read(self, filename):
        
        # apply the read of the base class
        BaseState.read(self,filename)
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
                
        #setattr(self, Fields, temp)
        
class UnconvertedState(BaseState):
    
    def __init__(self):
        BaseState.__init__(self)
        pass
    
    def read(self, filename):
        
        # apply the read of the base class
        BaseState.read(self,filename)
        fin = self.fin
            
        # find the fields
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
                
                # set attributes
                setattr(self.fields, subg, field)
                
        #setattr(self, Fields, temp)