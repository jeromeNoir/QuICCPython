The branch devel is supposed to self-contain the mess of change of architecture before merging to the "new master"

# QuICCPython

Instalation:

- install dependencies:
    shtns:
    - hg clone https://bitbucket.org/nschaeff/shtns
    - cd shtns
    - ./configure 
    - make 
    - make install 

- install QuICCPython
    - cd ../
    - python3 -m pip install ./


To read the HDF5:

readSpectral(<fileName>)
readPhysical(<fileName>)

readPhysical:(Meredith / Magical Leo)

read EPM and QuICC hdf5 - it is a mapping between what is in the hdf5 and python. Fields are created as needed!!!
structure of the ouput:   
    
    myData=readPhysical...
    
    grid output: 
                if the grid is cartesian
    myData.grid.x -> 3D array
               .y
               .z
               ...
               
               if the grid is spherical create only 
               
               .r
               .theta
               .phi...
               
    Parameters: create the fields as needed!!!
    
    myData.parameters.time
                     .timeStep
                     .Ek
                     .Ra...
                     IMPORTANT geometry
    
    myData.velocity.u
                   .v
                   .w...or for spherical it should look like
                   .ur
                   .utheta
                   .uphi
                   
     otherfields created as needed:
     
     myData.vorticity
     myData.temperature
     myData.magneticField...
     
     
     
     
     
    TODO: how to turn the dictionary keys into field names (set attribute)
    
    readSpectral:(Magical Leo)
    
    myData.parameters....IMPORTANT geometry
    myData.velocity.poloidal -> 2x(spectral dimension) 
    myData.velocity.poroidal
    myData.temperature....
    
Reconstruction of Data:
    getUniformVorticity(myData)
    alignedToFluidRotationAxis(myData)
    geostrophicComponent(myData)
    
    
    
Visualisation of Data.
        
