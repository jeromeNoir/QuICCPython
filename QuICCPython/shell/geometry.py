import numpy as np

def makeOptimalIWCut(state, nPoints, isGeostrophicCut = False):

    # obtain spherical shell aspect ratio
    eta = state.parameters.rratio
    
    # obtain oscillation frequency
    try:

        omega_n = state.parameters.omega
    except AttributeError as e:

        omega_n = 0
        pass

    # define phi_0 in the frame of nutation
    phi_0 = np.pi/2 + omega_n * state.parameters.time
    
    # obtain propagation angle of inertial waves
    theta_prime = np.pi/2 - np.arccos(np.abs(omega_n)/2)
    if isGeostrophicCut:
        theta_prime = 0

    rmid = 1/2*(1+eta)/(1-eta)
    zmin = np.cos(theta_prime) * rmid
    smin = np.sin(theta_prime) * rmid
    zmax = zmin - np.sin(theta_prime) * 1.5 * eta/(1-eta)
    smax = smin + np.cos(theta_prime) * 1.5 * eta/(1-eta)
    zmin -= np.sin(theta_prime) * .5 * eta/(1-eta)
    smin += np.cos(theta_prime) * .5 * eta/(1-eta)

    # generate the cartesian grids
    z = np.linspace(zmin, zmax, nPoints)
    s = np.linspace(smin, smax, nPoints)

    # generate the grids in spherical coordinates
    r = (s**2 + z**2)**.5
    theta = np.arctan2(s, z)
    phi = np.ones_like(r)* phi_0

    return r, theta, phi

def makeUpperTCIWCut(state, nPoints, isGeostrophicCut = False):

    # obtain spherical shell aspect ratio
    eta = state.parameters.rratio
    
    # obtain oscillation frequency
    try:

        omega_n = state.parameters.omega
    except AttributeError as e:

        omega_n = 0
        pass

    # define phi_0 in the frame of nutation
    phi_0 = np.pi/2 + omega_n * state.parameters.time
    
    # obtain propagation angle of inertial waves
    theta_prime = np.pi/2 - np.arccos(np.abs(omega_n)/2)
    if isGeostrophicCut:
        theta_prime = 0
        
        
    zmid = 1/2*(1/np.sin(theta_prime) + np.sin(theta_prime)) * eta / (1-eta)
    smid = 1/2*(np.cos(theta_prime))*eta/(1-eta)
    
    distance = np.min([1,1 - np.cos(np.arctan(1/np.tan(theta_prime)/2))]) * eta / (1-eta) *.9
    print(distance)
    # distance = np.max([distance, .15])
    zmax = zmid + np.sin(theta_prime) * distance
    smax = smid + np.cos(theta_prime) * distance
    zmin = zmid - np.sin(theta_prime) * distance
    smin = smid - np.cos(theta_prime) * distance

    # generate the cartesian grids
    z = np.linspace(zmin, zmax, nPoints)
    s = np.linspace(smin, smax, nPoints)

    # generate the grids in spherical coordinates
    r = (s**2 + z**2)**.5
    theta = np.arctan2(s, z)
    phi = np.ones_like(r)* phi_0

    return r, theta, phi
