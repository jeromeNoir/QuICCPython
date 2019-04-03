"""
This module includes important functions to rotate an hdf5 state 
and perform the integral of vorticity from the spectral velocity field
"""
import numpy as np
import h5py
import sys, os
env = os.environ.copy()
sys.path.append(env['HOME'] + '/workspace/QuICC/Python/')
from quicc.geometry.spherical import shell_radius
from quicc.projection import shell
import QuICC as pybind
from shutil import copyfile

# generate wigner matrix
def dlmb(L):
    #INPUT:
    #L       Maximum angular degree
    #OUTPUT:
    #D       Lower right quarter of the Wigner D-matrix m>=0
    #d       Masters' concatenated output
    # Computes matrix elements for spherical harmonic polar rotation around
    # the y-axis over 90 degrees only. 
    #Rotation matrix
    #  D_{mm'}(a,b,g) = exp(-ima) d_{mm'}(b) exp(-im'g) 
    # but we factor the rotation itself into:
    #    R(a,b,g)=R(a-pi/2,-pi/2,b)R(0,pi/2,g+pi/2)
    # thus we only need to compute d_{mm'} for b=90.
    # After a code by T. Guy Masters.
    # See also McEwen, 2006.
    # Last modified by fjsimons-at-alum.mit.edu, 08/05/2008
                                  
    d=np.zeros(np.sum( ( np.array(range(L+1)) + 1 )**2 ) )
    # Initialize using D&T C.115.
    # l = 0 
    d[0] = 1
    # l = 1 
    if L >= 1:
        d[1] = 0
        d[2] = 1.0 / np.sqrt(2)
        d[3] = -1.0 / np.sqrt(2)
        d[4]= 0.5 ;
        
    #pointer index
    ind = 5
    #factor
    f1= 0.5
    #Lwait = 100
    #Recursions
    for l in range(2,L+1):
        lp1 = l + 1
        knd = ind + lp1 
        #print(knd)
        fl2p1 = l + lp1
        vect = np.array( range(1,l+1) )
        f = np.sqrt( vect * ( fl2p1 - vect) )
        f1 = f1 * ( 2.0 * l - 1.0 ) / ( 2.0 * l )
        #print f1
        d[knd-1] = -np.sqrt(f1)
        d[knd-2] = 0
        for i in range(2,l+1):
            j = knd-i
            #print('j=',j)
            d[j-1] = -f[i-2] * d[j+1] / f[i-1]
        #print d

        #Positive N (bottom triangle)
        f2 = f1
        g1 = l 
        g2 = lp1
        for N in range(1,l+1):
            knd = knd + lp1
            en2 = N + N
            g1 = g1 + 1
            g2 = g2 - 1
            f2 = f2 * g2 / g1
            #print(f2)
            d[knd - 1] = -np.sqrt(f2)
            d[knd - 2] = d[knd-1]*en2/f[0]
            #print d[knd-2]
            for i in range(2, l-N+1):
                j = knd-i
                d[j-1] = ( en2 * d[j] - f[i-2] * d[j+1] ) / f[i-1]
                #print d[j-1]

        #Fill upper triangle and fix signs
        for j in range(1,l+1):
            for m in range(j,l+1):
                d[ind+m*lp1+j-1]=d[ind+j*lp1+m-l-1]

        isn=1+np.mod(l,2)
        for n in range(0,l+1):
            knd=ind+n*lp1
            for i in range(isn,lp1+1,2):
                d[knd+i-1]=-d[knd+i-1]
        ind=ind+lp1*lp1;

    #Now let's rearrange the coefficients as 1x1, 2x2, 3x3 etc rotation
    #matrices.
    cst=1;
    D=np.empty(L+1,dtype=object)
    #Start of coefficient sequence; need transpose!
    for l in range(1,L+2):
        #Leo: This line doesn't work !!!
        #print l
        #print(len(d[cst-1:cst+l*l-1]))
        #print(np.reshape(d[cst-1:cst+l*l-1],(l,l)))
        D[l-1]=np.reshape(d[cst-1:cst+l*l-1],(l,l))
        cst=cst+l*l
        #print(cst)
    return (D,d)

def computeAverageVorticity(state):
    f = h5py.File(state, 'r') #read state
    nR=f['/truncation/spectral/dim1D'].value+1

    #Toroidal Velocity
    dataT=f['/velocity/velocity_tor'].value
    dataT = dataT[:,:,0] + 1j* dataT[:, :, 1]

    # obtain the map factor for the Tchebyshev polynomials
    ro = f['/physical/ro'].value
    rratio = f['/physical/rratio'].value
    a, b = shell_radius.linear_r2x(ro,rratio)
    
    # compute the weight of the operator
    ri = ro* rratio
    delta = (f['/physical/ekman'].value)**.5*10
    riBoundary = ri+delta
    roBoundary = ro-delta
    volume = ((ro-delta)**5 - (ri+delta)**5 )/ (5. * (3. / (4 *np.pi))**.5)

    # define boundary conditions
    bc = {0:0, 'cr':2}
    R2 = shell_radius.r2(nR+2, a, b, bc)
    bc['cr'] = 1
    R1 = shell_radius.r1(nR+3, a, b, bc)
    I1 = shell_radius.i1(nR+4, a, b, bc)

    Pi = shell.eval(nR+4, a, b, np.array([roBoundary, riBoundary]))
    
    proj_vec = Pi*I1*R1*R2
    proj_vec = np.array((proj_vec[0,:]-proj_vec[1,:])/volume)

    omegax = -np.real(np.dot(proj_vec, np.array(dataT[2,:])))[0]*2**.5
    omegay = np.imag(np.dot(proj_vec, np.array(dataT[2,:])))[0]*2**.5
    omegaz = np.real(np.dot(proj_vec, np.array(dataT[1,:])))[0]
        
    return np.array([omegax, omegay, omegaz])




def rotateState(state, omega, field):
    finout = h5py.File(state, 'r+') #read state
    LL=finout['/truncation/spectral/dim2D'].value
    MM=finout['/truncation/spectral/dim3D'].value
    NN=finout['/truncation/spectral/dim1D'].value
    Ro = finout['/physical/ro'].value
    
    #Toroidal Velocity
    #data=np.array(f[field].value)
    data=finout[field].value
    
    ##Computing rotation axis and allocating memory 
    DD=dlmb(LL)

    ###Determine rotation axis from spectral coefficients
    #divide by normalization of n(1,1) mode 
    #(-1) from condon-shortley phase

    (alp, bta, gam) = computeEulerAngles(omega * Ro)

    #e(i*m*alpha) = cos(m*alpha) + i * sin(m*alpha)
    #calp = cos(m*alpha)
    #salp = sin(m*alpha)
    mmVect=np.array(range(0,int(MM)+1))
    llVect=np.array(range(0,int(LL)+1))

    calp=np.cos(mmVect*alp)
    salp=np.sin(mmVect*alp)

    #cbta = cos(l*bta)
    #sbta = sin(l*bta)
    cbta=np.cos(llVect*bta)
    sbta=np.sin(llVect*bta)
    
    #Schmidt normalization factors
    NF=np.array([])
    ind =  0
    for l in range(0, int(LL)+1):
            for m in range(0,min(l,MM)+1):
                if (m==0):
                    #print(l,m)
                    #NF[ind]=1.0
                    #NF=1.0
                    NF=np.append(NF,1.0) #Schmidt normalization for m = 0 
                else:
                    #print(l,m)
                    #NF[ind]=np.sqrt(2)/2;
                    #NF=np.sqrt(2)/2
                    #NF=np.append(NF,np.sqrt(2)/2) #Schmidt normalization for m != 0
                    NF = np.append(NF, 1.)  # Schmidt normalization for m != 0
                
                ind=ind+1;
            
    #ind = 529 (same as matlab)

    #Rotating EPM results

    for n in range(0, NN+1 ): #N Chebyshev polynomials

        # preallocate the memory
        alp=np.zeros((int(((LL+2)*(LL+1))/2),2))
        bta=np.zeros((int(((LL+2)*(LL+1))/2),2))
        
        ind=0
        for l in range(0, LL+1 ):

            for m in range(0, min(l,MM)+1 ): #Leo fix: m from 0 to l
                #print ('ind, l, m', ind, l , m )
                # Azimuthal rotation (Z) by (ALPHA - PI/2)
                #Dividing by normalisation 
                Cos = data[ind,n][0] / NF[ind] #index, n, real
                Sin = data[ind,n][1] / NF[ind] #index, n, imag
                #Rotate about Z by alpha
                #print(Cos, Sin, calp[m], salp[m])
                alp[ind,0] = Cos * calp[m] + Sin * salp[m]
                alp[ind,1] = Sin * calp[m] - Cos * salp[m]
                
                ind=ind+1
                
            ###### tilt by bta
            li=l+1
            i,j=np.meshgrid(range(1,li+1),range(1,li+1))
            #checkboard * 2 
            IC=(((i+j)+li%2)%2)*2
            IC[:,0]=1 #m=0
            #inverted checkboard *2
            IS=((((i+j)+li%2)%2)<1)*2
            IS[:,0]=1 #m=0
            
            # STEP 1: PASSIVE colatitudinal (Y) rotation over -PI/2
            Cp = np.dot( DD[0][l].T * IC, \
                         alp[ int((l+1)*(l+2) /2 -l-1) : int((l+1) * (l+2) / 2), 0]) #X Tor
            Sp = np.dot( DD[0][l].T * IS, \
                         alp[ int((l+1)*(l+2) /2 - l - 1) : int((l+1) * (l+2) / 2), 1]) #Y Tor

            
            # STEP 2: PASSIVE azimuthal (Z) rotation over BETA 
            Cpp = Cp * cbta[0:l+1].T + Sp * sbta[0:l+1].T
            Spp = Sp * cbta[0:l+1].T - Cp * sbta[0:l+1].T

            # STEP 3: PASSIVE colatitudinal (Y) rotation over PI/2
            Cpp = np.dot( DD[0][l] * IC, Cpp)
            Spp = np.dot( DD[0][l] * IS, Spp)
                        
            ###### STEP4: PASSIVE azimuthal rotation over (GAMMA +PI/2)
            bta[int((l+1)*(l+2)/2-l-1):int((l+1)*(l+2)/2),0] = Cpp
            bta[int((l+1)*(l+2)/2-l-1):int((l+1)*(l+2)/2),1] = Spp
     
        ind=0
        for l in range(0, LL+1 ):
            for m in range(0, min(l,MM)+1 ): #Leo fix: m from 0 to min(l,M)
                #print("ind, l , m ", ind, l, m)
                #Multiplying by Schmidt normalization 
                data[ind,n][0]=bta[ind,0]*NF[ind]
                data[ind,n][1]=bta[ind,1]*NF[ind]
                
                ind=ind+1
                
        #print('dataT:',dataT[6,0])

    rotatedState = data#(dataT, dataP, dataC)

    finout[field].value[:] = data

    finout.close()
    #return rotatedState
    pass

def selectModes(state, modes, field):
    f = h5py.File(state, 'r+') #read state
    LL=int(np.array(f['/truncation/spectral/dim2D'], dtype='double'))
    MM=int(np.array(f['/truncation/spectral/dim3D'], dtype='double'))
    NN=int(np.array(f['/truncation/spectral/dim1D'], dtype='double'))

    # impose single modes on field
    data=f[field].value
    
    ind=0
    for l in range(0, LL+1 ):
        for m in range(0, min(l,MM)+1 ): #Leo fix: m from 0 to min(l,M)
            if m not in modes:

                # Set mode to 0
                data[ind,:][0]=0.
                data[ind,:][1]=0.
                
            ind=ind+1
                
    f[field][:] = data
    f.close()
    return data

def removeModes(state, modes, field):
    f = h5py.File(state, 'r+') #read state
    LL=f['/truncation/spectral/dim2D'].value + 1
    MM=f['/truncation/spectral/dim3D'].value + 1
    NN=f['/truncation/spectral/dim1D'].value + 1

    #Toroidal Velocity
    data=f[field].value

    ind=0
    for l in range(0, LL+1 ):
        for m in range(0, min(l,MM)+1 ): #Leo fix: m from 0 to min(l,M)
            if m in modes:
                data[ind,:][0]=0.
                data[ind,:][1]=0.
    
            ind=ind+1
                
    
    f[field][:] = data
    f.close()
    return data

def correctRotation(state, toBeCorrected = ['/velocity/velocity_tor','/velocity/velocity_pol']):
    
    try:
        omega = computeAverageVorticity(state)
        for field in toBeCorrected:
            rotateState(state, omega, field)

    except Exception as e:
        print(e)
        pass

def subtract_uniform_vorticity(state, omega):
    finout = h5py.File(state, 'r+')
    eta = finout['/physical/rratio'].value
    a = .5
    b = .5*(1+eta)/(1-eta)

    dataT = finout['/velocity/velocity_tor'].value
    dataT = dataT[:, :, 0] + dataT[:, :, 1]*1j
    
    dataT[1, 0] -= 2*(np.pi/3)**.5 * b * omega[2]
    dataT[1, 1] -= 2*(np.pi/3)**.5 * a * omega[2]*.5
    
    dataT[2, 0] -= 2*(2*np.pi/3)**.5 * b * (-omega[0] + omega[1]*1j)*.5
    dataT[2, 1] -= 2*(2*np.pi/3)**.5 * a * (-omega[0] + omega[1]*1j)*.5*.5
        
    torField = finout['/velocity/velocity_tor'].value
    torField[:, :, 0] = dataT.real
    torField[:, :, 1] = dataT.imag
    finout['/velocity/velocity_tor'][:]=torField
    finout.close()
    pass

def rotate_state(state, omega, fields):

    finout = h5py.File(state, 'r+')

    LL=finout['/truncation/spectral/dim2D'].value 
    MM=finout['/truncation/spectral/dim3D'].value 
    NN=finout['/truncation/spectral/dim1D'].value 
    Ro = finout['/physical/ro'].value
    # compute Euler angles
    (alpha, beta, gamma) = computeEulerAngles(omega * Ro)

    # loop over the fields
    for field in fields:
        data = finout[field].value
        Qlm = data[:, :, 0]+ 1j*data[:, :, 1]
        Slm = np.zeros_like(Qlm)
        Tlm = np.zeros_like(Qlm)

        # first rotation (around Z)
        #print(type(Qlm), Qlm.shape, Qlm.dtype, type(Slm), Slm.shape, Slm.dtype)
        alpha = float(alpha)
        LL = int(LL)
        MM = int(MM)
        Qlm = np.asfortranarray(Qlm)
        Slm = np.asfortranarray(Slm)
        Tlm = np.asfortranarray(Tlm)
        #print(type(alpha), type(LL), type(MM)) 
        
        pybind.ZrotateFull(Qlm, Slm, alpha, LL, MM)

        # second rotation (around X)
        pybind.XrotateFull(Slm, Tlm, beta, LL, MM)

        # third rotation (arond Z, back to original orientation)
        pybind.ZrotateFull(Tlm, Qlm, gamma, LL, MM)
        
        field_temp = finout[field].value
        field_temp[:, :, 0] = np.real(Qlm)
        field_temp[:, :, 1] = np.imag(Qlm)
        finout[field][:]=field_temp
        
    pass
    
def computeEulerAngles(omega):
    # omega is in the mantle frame
    # the real omega_f in the frame of reference is omega + [0 0 1]
    #[phi0, theta0]: rotation axis of fluid
    omegax = omega[0]
    omegay = omega[1]
    omegaz = omega[2]
    phi0=np.arctan2(omegay, omegax)
    theta0=np.arctan2((omegax**2+omegay**2)**.5, (omegaz+1))

    #### determine axis of fluid from uniform vorticity
    # using thetat0 and phi0 determined from mean vorticity
    #Euler Angles
    #Alpha: 1st rotation about Z
    #R = R(alpha-pi/2, -pi/2, beta)* R(0,pi/2, gamma+pi/2)
    alpha=-phi0-np.pi/2
    #Beta: 2nd rotation about X
    beta=-theta0
    #Gamma: 3rd rotation about Z
    gamma=np.pi/2

    return (alpha, beta, gamma)

def process_state(state):

    # compute averate vorticity
    omega = computeAverageVorticity(state)
    
    # copy the state file ( rotations and subtractions of states are done in place
    filename = 'stateRotated.hdf5'
    copyfile(state, filename)

    # subtract average vorticity
    subtract_uniform_vorticity(filename, omega)

    # to be rotated: '/velocity/velocity_tor', '/velocity/velocity_pol'
    fields = ['/velocity/velocity_tor', '/velocity/velocity_pol']
    rotate_state(filename, omega, fields)
        
    pass
    
