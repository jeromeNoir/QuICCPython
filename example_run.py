from BaseReader import BaseState, ConvertedState, UnconvertedState

# This should be the path to the state file
filename = 'PATH_TO_STATE_FILE'

# Use either SpectralState() or PhysicalState() depending on the file you are reading
a = ConvertedState()

a.read(filename)

# Read time and timestep 
a.parameters.time
a.parameters.timestep

# Read problem specific parameters from: a.parameters.PARAM_OF_CHOICE
# Example:  a.parameters.ekman, a.parameters.omega

# Read mesh from a.parameters.GRID_OPTIONS
# Example:  a.parameters.grid_x, a.parameters.grid_r

# Read fields from a.fields.FIELDS
# Example:  a.fields.velocityz, a.fields.velocity_tor

