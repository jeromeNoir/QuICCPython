import numpy as np
# define function for equatorial plane visualization from the visState0000.hdf5 file
def getMeridionalSlice(state, phi=None, fieldname = 'velocity'):

    # find the grid in radial and meridional direction
    r = state.r_axis #.value[idx_r]
    theta =  state.t_axis
    
    rr, ttheta = np.meshgrid(state.r_axis, state.t_axis)
    X = rr*np.sin(ttheta)
    Y = rr*np.cos(ttheta)

    if phi == None:
        # select the 0 meridional plane value for 
        idx_phi0 = (state.p_axis==0)
        
        Field1 = np.mean(getattr(state.fields, fieldname+'r')[:, :, idx_phi0], axis=2)
        Field2 = np.mean(getattr(state.fields, fieldname+'t')[:, :, idx_phi0], axis=2)
        Field3 = np.mean(getattr(state.fields, fieldname+'p')[:, :, idx_phi0], axis=2)
    else:
        phi = state.paxis
        Field1 = interp1d(phi, getattr(state.fields, fieldname+'r'), axis=2)(phi)
        Field2 = interp1d(phi, getattr(state.fields, fieldname+'t'), axis=2)(phi)
        Field3 = interp1d(phi, getattr(state.fields, fieldname+'p'), axis=2)(phi)
        
    return X, Y, [Field1, Field2, Field3]


# define function for equatorial plane visualization from the visState0000.hdf5 file
def getEquatorialSlice(state, fieldname = 'velocity'):
                    
    # select the r grid in the bulk of the flow
    #idx_r = (fopen['mesh/grid_r']>ri+delta) & (fopen['mesh/grid_r']<ro-delta)
    #idx_r = (fopen['mesh/grid_r'].value>0.)
    
    # select the 0 meridional plane value for
    theta_grid = state.t_axis
    neq = int(len(theta_grid)/2)
    if len(theta_grid) % 2 == 0:
        idx_theta = (theta_grid == theta_grid[neq-1]) | (theta_grid == theta_grid[neq])
        print(theta_grid[neq-1]-np.pi/2, theta_grid[neq]-np.pi/2)
    else:
        idx_theta = (theta_grid == theta_grid[neq])
        
    # find the grid in radial and meridional direction
    #r = fopen['mesh/grid_r'].value[idx_r]
    r = state.r_axis
    phi = state.p_axis
    
    rr, pphi = np.meshgrid(r, phi)
    # compute the values for x and y
    X = np.cos(pphi)*rr
    Y = np.sin(pphi)*rr
    
    Field1 = np.mean(getattr(state.fields, fieldname+'r')[:,idx_theta,:], axis=1)
    Field2 = np.mean(getattr(state.fields, fieldname+'t')[:,idx_theta,:], axis=1)
    Field3 = np.mean(getattr(state.fields, fieldname+'p')[:,idx_theta,:], axis=1)

    return X, Y, [Field1, Field2, Field3]
