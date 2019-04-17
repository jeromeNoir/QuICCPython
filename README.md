# QuICCPython

QuICCPython is a postprocessing software developed (almost natively)
in Python. It provides routines for the manipulation of stored HDF5 files
generated from QuICC runs
(https://github.com/QuICC/QuICC.git). Several routines are also available to
perform computation of kinetic and magnetic energy, uniform vorticity
of the flow. Generation of slices for the different geometries
supported by QuICC is also supported.



## Instalation:

### Install dependencies

QuICCPython requires a C++ compiled module to compute associated
Legendre functions, Worland polynomials (Marti and Jackson, 2016), and
perform rotation of the spectrum (Gimbutas and Greengard, 2009). To
provide this requirements, please follow this steps

- install SHTns (Schäffer, 2013):

You are gonna need mercurial for
this task. Download SHTns in your workspace or a download folder.

    hg clone https://bitbucket.org/nschaeff/shtns
    cd shtns
    ./configure 
    make 
    make install

- install pybind11

    - export MACOSX_DEPLOYMENT_TARGET=10.9 (** For Anaconda-Mac users)
    - python3 -m pip install pybind11 
- clone QuICCPython 
    - cd ../
    - git clone git@github.com:jeromeNoir/QuICCPython.git
    - cd QuICCPython
- get Eigen sources 
    - git submodule init
    - git submodule update
- install QuICCPython
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
        
