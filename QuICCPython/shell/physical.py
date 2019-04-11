import numpy as np
from scipy.interpolate import interp1d, interpn

# define the following dictionaries for the convertion of field names
# to the names used by QuICC for storage in physical fields
field_storage = {'velocity': 'velocity', 'vorticity': 'velocity_curl', 'magnetic': 'magnetic', 'current': 'magnetic_curl'}
field_presentation = {'velocity': 'u', 'vorticity': 'vort', 'magnetic': 'b', 'current': 'j'}

# define function for equatorial plane visualization from the visState0000.hdf5 file
def getMeridionalSlice(phys_state, phi=None, field = 'velocity'):

    # some parameters just in case
    eta = phys_state.parameters.rratio
    ri = eta/(1-eta)
    ro = 1/(1-eta)
    a, b = .5, .5*(1+eta)/(1-eta)

    # find the grid in radial and meridional direction
    r = phys_state.grid_r #.value[idx_r]
    theta =  phys_state.grid_theta

    rr, ttheta = np.meshgrid(r, theta)
    X = rr*np.sin(ttheta)
    Y = rr*np.cos(ttheta)

    fields = field_storage[field]
    if phi == None:
        # select the 0 meridional plane value for 
        idx_phi0 = (phys_state.grid_phi==0)

        Field1 = np.mean(getattr(phys_state.fields, fields+'_r')[:, :, idx_phi0], axis=2)
        Field2 = np.mean(getattr(phys_state.fields, fields+'_theta')[:, :, idx_phi0], axis=2)
        Field3 = np.mean(getattr(phys_state.fields, fields+'_phi')[:, :, idx_phi0], axis=2)
    else:
        phi = phys_state.grid_phi
        Field1 = interp1d(phi, getattr(phys_state.fields_storage, fields+'_r'), axis=2)(phi)
        Field2 = interp1d(phi, getattr(phys_state.fields_storage, fields+'_theta'), axis=2)(phi)
        Field3 = interp1d(phi, getattr(phys_state.fields_storage, fields+'_phi'), axis=2)(phi)

    fieldp = field_presentation[field]
    result = {'x': X, 'y': Y, fieldp+'R': Field1.T, fieldp+'Theta': Field2.T, fieldp+'Phi': Field3.T}
    return result



# define function for equatorial plane visualization from the visState0000.hdf5 file
def getEquatorialSlice(phys_state, field = 'velocity'):

    # select the r grid in the bulk of the flow
    #idx_r = (fopen['mesh/grid_r']>ri+delta) & (fopen['mesh/grid_r']<ro-delta)
    #idx_r = (fopen['mesh/grid_r'].value>0.)

    # select the 0 meridional plane value for
    theta_grid = phys_state.grid_theta
    neq = int(len(theta_grid)/2)
    if len(theta_grid) % 2 == 0:
        idx_theta = (theta_grid == theta_grid[neq-1]) | (theta_grid == theta_grid[neq])
    else:
        idx_theta = (theta_grid == theta_grid[neq])

    # find the grid in radial and meridional direction
    #r = fopen['mesh/grid_r'].value[idx_r]
    r = phys_state.grid_r
    phi = phys_state.grid_phi

    rr, pphi = np.meshgrid(r, phi)
    # compute the values for x and y
    X = np.cos(pphi)*rr
    Y = np.sin(pphi)*rr

    fields = field_storage[field]
    Field1 = np.mean(getattr(phys_state.fields, fields+'_r')[:,idx_theta,:], axis=1)
    Field2 = np.mean(getattr(phys_state.fields, fields+'_theta')[:,idx_theta,:], axis=1)
    Field3 = np.mean(getattr(phys_state.fields, fields+'_phi')[:,idx_theta,:], axis=1)

    fieldp = field_presentation[field]
    result = {'x': X, 'y': Y, fieldp+'R': Field1.T, fieldp+'Theta': Field2.T, fieldp+'Phi': Field3.T}
    return result

# define function for equatorial plane visualization from the visState0000.hdf5 file
def getIsoradiusSlice(phys_state, r=.5, field = 'velocity'):

    # some parameters just in case
    eta = phys_state.parameters.rratio
    ri = eta/(1-eta)
    ro = 1/(1-eta)
    a, b = .5, .5*(1+eta)/(1-eta)
    r += ri
     # find the grid in radial and meridional direction
    theta = phys_state.grid_theta
    phi = phys_state.grid_phi

    TTheta, PPhi = np.meshgrid(theta, phi)
    r_grid = phys_state.grid_r

    # evaluate the fields
    fields = field_storage[field]
    Field1 = interp1d(r_grid, getattr(phys_state.fields, fields+'_r'), axis=0)(r)
    Field2 = interp1d(r_grid, getattr(phys_state.fields, fields+'_theta'), axis=0)(r)
    Field3 = interp1d(r_grid, getattr(phys_state.fields, fields+'_phi'), axis=0)(r)

    fieldp = field_presentation[field]
    result = {'theta': TTheta, 'phi': PPhi, fieldp+'R': Field1.T, fieldp+'Theta': Field2.T, fieldp+'Phi': Field3.T}
    return result

# define function for equatorial plane visualization from the visState0000.hdf5 file
def getPointValue(phys_state, x1, x2, x3, field = 'velocity'):

    assert (phys_state.geometry == 'shell'), "Tools not implemented for geometry"+phys_state.geometry

    # find the grid in radial and meridional direction
    x1_grid = phys_state.grid_r
    x2_grid = phys_state.grid_theta
    x3_grid = phys_state.grid_phi

    fields = field_storage[field]
    Field1 = interp1d((x1_grid, x2_grid, x3_grid), getattr(phys_state.fields, fields+'_r'), (x1, x2, x3))
    Field2 = interp1d((x1_grid, x2_grid, x3_grid), getattr(phys_state.fields, fields+'_theta'), (x1, x2, x3))
    Field3 = interp1d((x1_grid, x2_grid, x3_grid), getattr(phys_state.fields, fields+'_phi'), (x1, x2, x3))
    fieldp = field_presentation[field]
    result = {'r': x1, 'theta':x2, 'phi': x3, fieldp+'R': Field1.T, fieldp+'Theta': Field2.T, fieldp+'Phi': Field3.T}
    return result
