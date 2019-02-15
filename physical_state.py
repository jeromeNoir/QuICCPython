from BaseState import BaseState

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
