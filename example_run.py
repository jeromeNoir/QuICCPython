from BaseReader import SpectralState, PhysicalState

# This should be the path to the state file
# e.g.
#filename = 'PATH_TO_STATE_FILE'
filename = '/Users/meredith/Desktop/state_convert.hdf5'

# Use either SpectralState() or PhysicalState() depending on the file you are reading
my_state_physical = PhysicalState(filename)

#Leo: change this to 
#Add examples for every case
#Spherical: Precession, Convection
#Shell: Nuttation, 
#Cartesian: 
#TODO: See if it's possible to truncate the spectral resolution for the PhysicalState?
#Ultimately, for a publication one will provide a Python Notebook to generate all the figures

filename = './state0000.hdf5'
#my_state_spectral=SpectralState(filename)

# Read time and timestep 
time = my_state.parameters.time
print('time: ', time)
timestep = my_state.parameters.timestep
print('timestep: ', timestep)

# Read problem specific parameters from: a.parameters.PARAM_OF_CHOICE
# Example:  my_state_spectral.parameters.ekman, my_state_spectral.parameters.omega
# Ra = my_state_physical.parameters.rayleigh
# print('Ra: ', Ra)

# Read mesh from a.parameters.GRID_OPTIONS
# Example:  my_state_physical.parameters.grid_x, my_state_physical.parameters.grid_r

# Read fields from my_state_physical.fields.FIELDS
# Example:  my_state_physical.fields.velocityz, my_state_spectral.fields.velocity_tor

