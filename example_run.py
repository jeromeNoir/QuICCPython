from BaseReader import BaseState, SpectralState, PhysicalState

# This should be the path to the state file
filename = 'PATH_TO_STATE_FILE'
filename = '/Users/meredith/Desktop/state_convert.hdf5'

# Use either SpectralState() or PhysicalState() depending on the file you are reading
a = PhysicalState()

a.read(filename)

# Read time and timestep 
time = a.parameters.time
print('time: ', time)
timestep = a.parameters.timestep
print('timestep: ', timestep)

# Read problem specific parameters from: a.parameters.PARAM_OF_CHOICE
# Example:  a.parameters.ekman, a.parameters.omega
# Ra = a.parameters.rayleigh
# print('Ra: ', Ra)

# Read mesh from a.parameters.GRID_OPTIONS
# Example:  a.parameters.grid_x, a.parameters.grid_r

# Read fields from a.fields.FIELDS
# Example:  a.fields.velocityz, a.fields.velocity_tor

