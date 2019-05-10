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
    processedState = processState(state)    

    E = state.parameters.ekman
    #Ns = 3/E**.3
    Ns = 100
    spectralUS = computeZIntegral(processedState, 'uS', Ns, maxM = 100)
    savemat('GeostrophicUS.mat', mdict = spectralUS)
    
    spectralUPhi = computeZIntegral(processedState, 'uPhi', Ns, maxM = 100)
    savemat('GeostrophicUPhi.mat', mdict = spectralUPhi)
    
    spectralVortZ = computeZIntegral(processedState, 'vortZ', Ns, maxM = 100)
    savemat('GeostrophicVortZ.mat', mdict = spectralVortZ)

    
    meridFields = getMeridionalSlice(state)
    savemat('MeridionalFlowPhys.mat', mdict = meridFields)
    equatFields = getEquatorialSlice(state)
    savemat('EquatorialFlowPhys.mat', mdict = equatFields)
    midFields = getIsoradiusSlice(state)
    savemat('MidradiusFlowPhys.mat', mdict = midFields)

    # output the fields in the boundary layers
    innerBound = getIsoradiusSlice(state, r = 2.* E **.5)
    savemat('IBoundaryFlowPhys.mat', mdict = innerBound)
    outerBound = getIsoradiusSlice(state, r = 1 - 2. * E**.5)
    savemat('OBoundaryFlowPhys.mat', mdict = outerBound)
                
    return
    
if __name__=="__main__":
    main(sys.argv[1])
