import sys, os
env = os.environ.copy()
sys.path.append(env['HOME'] + '/workspace/QuICCPython/')
from scipy.io import savemat
from QuICCPython.read import SpectralState
from QuICCPython.shell.spectral import computeZIntegral, getMeridionalSlice,\
    getEquatorialSlice, getIsoradiusSlice, processState

# define the main process file
def main(filename):

    # open the state
    state = SpectralState(filename, 'shell')
    
    # turn the file in the frame of rotation and subtract unif vorticity'
    omega = computeUniformVorticity(state)

    # rotate the states to put them in the fluid axis
    Ro = state.parameters.ro 
    rotatedState = alignAlongFluidAxis(state, Ro*omega)
    

    E = state.parameters.ekman

    Ns = 100
    spectralUPhi = computeZIntegral(rotatedState, 'uPhi', Ns, maxM = 100)
    savemat('GeostrophicUPhi.mat', mdict = spectralUPhi)

    # output the fields in the boundary layers
    innerBound = getIsoradiusSlice(rotatedstate, r = 1.* E **.5)
    savemat('IBoundaryFlowPhys.mat', mdict = innerBound)

    return
    
if __name__=="__main__":
    main(sys.argv[1])
