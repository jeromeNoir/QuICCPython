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

You are gonna need mercurial for this task. Download SHTns in your
workspace or a download folder.

    hg clone https://bitbucket.org/nschaeff/shtns
    cd shtns
    ./configure 
    make 
    sudo make install

- install pybind11

Pybind11 is required to link C++ executable into a Python module. Mac
user using python through anaconda require to set the following
environment variable:

    export MACOSX_DEPLOYMENT_TARGET=10.9

Pybind11 is available through pip

    python3 -m pip install pybind11

- download QuICCPython

Next you will download QuICCPython and obtain the submodules

    git clone https://github.com/jeromeNoir/QuICCPython.git
    cd QuICCPython
    git submodule init
    git submodule update

- install quicc-bind

You're gonna need to compile the C++ sources and link them to a python
module using pybind11. To do this, run

    python3 -m pip install ./
	

# Usage

## Reading files
To read the HDF5:

    from QuICCPython import read
    
    myData = read.SpectralState(<fileName>)  
    myData = read.PhysicalState(<fileName>)


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
        
