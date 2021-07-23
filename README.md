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
    From Jerome: the following doesn't work for me
    	-git submodule init
    	-git submodule update
    	instead I do:
    	-sudo git submodule init
    	-I upload manually the folder "eigen-git-mirror @ 1456fe2" on our git 
    	-copy it into the QuICCPyhton directory and rename it "eigen-git-mirror"

- install quicc-bind

You're gonna need to compile the C++ sources and link them to a python
module using pybind11. To do this, run

    python3 -m pip install ./
	
# Usage

## Reading files
HDF5 files can be opened with:
	
```python
from QuICCPython import read

myDataSpectral = read.SpectralState(<fileName>)
myDataPhysical = read.PhysicalState(<fileName>)
```
	
Once opened

```python
myData.parameters. ...
```
	
contains the parameters used to run the simulation, and

```python
myData.fields. ...
```
	
contains the states, either in spectral coefficient of physical
values, of the simulation.


## Obtaining slices and line cuts

Once the data are open  it is possible to obtain slices of the various
field components for plotting. The specific implementation is
dependent firstly on whether the data is in its spectral or physical
representation. Secondly, it depends on the geometry. The exact module
can be found in *QuICCPython.<geometry>.<specrtal/physical>*. For
example one can obtain meridional slices of the simulation state with:
	
```python
from QuICCPython.shell.spectral import getMeridionalSlice
meridionalSlice = getMeridionalSlice(myData, phi0 = np.pi/2)
```
	
The returned value is a python dictionary with keys

```python
meridionalSlice['x',
	'y',
	'uR',
	'uTheta',
	'uPhi']
```
	
thanks to which it is possible to plot with ease, with

```python
from matplotlib import pyplot as plt
plt.contourf(meridionalSlice['x'], meridionalSlice['y'], meridionalSlice['uPhi'])
```

Furthermore it is possible to store the results to memory with

```python
from scipy.io import savemat
savemat('myFile.mat', mdict = meridionalSlice)
```

For further examples, please consult the notebooks
*shellExample.ipynb*, *sphereExample.ipynb* and
*cartesianExample.ipynb* in the main folder.
