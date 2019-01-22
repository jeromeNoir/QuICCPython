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
        if !self.isEPM:
            setattr(self.parameters, 'time', fin['run/time'].value)
            setattr(self.parameters, 'timestep', fin['run/timestep'].value)
            for at in fin['physical'].keys():
                setattr(self.parameters, at, fin['physical'][at].value)
        else:
            setattr(self.parameters, 'time', fin['RunParameters/Time'].value)
            setattr(self.parameters, 'timestep', fin['RunParameters/Step'].value)
        
        # import attributes

        
        #self.geometry = fin.attrs['type']
        self.geometry = geometry
        if(geometry.lower() is not in list('cartesian', 'shell', 'sphere')):
            raise RuntimeError("I'm sorry but we only deal with Cartesian, Shell and Sphere, try again later")
        # geometry so far is
    pass

class PhysicalState(BaseState):
    
    
    def __init__(self, filename, geometry, file_type='QuICC'):
        
        # apply the read of the base class
        BaseState.__init__(self, filename, geometry, file_type='QuICC')
        fin = self.fin
        # read the grid
        if !self.isEPM:
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
                
                field = np.array(field_temp[:,:,0]+1j*field_temp[:,:,1])
                                                               
                
                # set attributes
                if self.isEPM:
                    # cast the subgroup to lower and transforms e.g.
                    # VelocityTor to velocity_tor
                    subg = subg.lower()
                    tmplist = list(subg)
                    tmplist.insert(-3,'_')
                    subg = ''.join(tmplist)
                setattr(self.fields, subg, field)
        #TODO: FOR EPM 
        #    for at in fin['physical'].keys():
        #        setattr(self.parameters, at, fin['physical'][at].value)
                
