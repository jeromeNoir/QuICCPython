from matplotlib import pyplot as plt
import numpy as np
from numpy import fft
from scipy.interpolate import griddata

def computeRealFields(spectral_result, filter=[]):
    #TODO: consider if one wants to apply an m=0 filter
    # generate the result of this function
    result = dict()

    # copy spectral results so that it doesnt delete the data
    
    # function used to implemented to generate the real representation of the field
    s = spectral_result['s']
    m = spectral_result['m']

    # generate the real grid for plotting
    mn = 2*(len(m)-1)
    phi = np.arange(0, mn )*2*np.pi /mn
    # append the values that close the poles
    s = np.hstack(([0.], s))
    phi = np.hstack((phi, [np.pi*2]))
    ss, pphi = np.meshgrid(s, phi)
    xx = np.cos(pphi) * ss
    yy = np.sin(pphi) * ss

    # store the grid
    result['x'] = xx
    result['y'] = yy
    result['phi'] = pphi
    result['s'] = ss

    # carry out the inverse fourier transform
    for k in spectral_result.keys():

        # skip over all the non matrix fields
        if spectral_result[k].ndim <2:
            continue

        # truncate the spectrum if needed
        temp = spectral_result[k]
        for m in filter:
            temp[m,:] = 0.
        # compute the tranforms
        field = fft.irfft(temp ,axis=0)
        
        # attach the right columns in the right place
        field = np.vstack((field,field[0, :]))
        temp1 = field[:, 0].reshape((-1, 1))
        field = np.hstack((np.ones_like(temp1) * np.mean(temp1), field))
        result[k] = field
                    
    return result


def streamplot(real_data):

    # find the grid of s and phi
    s = real_data['s'][0, :]
    phi = real_data['phi'][:, 0]
                
    # find the grid of s and phi
    s = real_data['s'][0, :]   
    phi = real_data['phi'][:, 0]

    uu = real_data['uS']
    vv = real_data['uPhi']
    speed = (uu**2 + vv**2)**.5
    ss, pphi = np.meshgrid(s, phi)
    xx = ss * np.cos(pphi)
    yy = ss * np.sin(pphi)
    
    # rotate the velocity
    u = uu * np.cos(pphi) - vv*np.sin(pphi)
    v = vv * np.cos(pphi) + uu* np.sin(pphi)
    
    x = np.linspace(xx.min(), xx.max(), 50)
    y = np.linspace(yy.min(), yy.max(), 50)
    
    xi, yi = np.meshgrid(x,y)
    
    #then, interpolate your data onto this grid:
    
    px = xx.flatten()
    py = yy.flatten()
    pu = u.flatten()
    pv = v.flatten()
    pspeed = speed.flatten()
    
    gu = griddata((px,py), pu, (xi,yi))
    gv = griddata((px,py), pv, (xi,yi))
    gspeed = griddata((px,py), pspeed, (xi,yi))
    
    lw = 3*gspeed/np.nanmax(gspeed)

    c = plt.streamplot(x,y,gu,gv, density=2.5, linewidth=lw)
