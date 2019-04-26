from matplotlib import pyplot as plt
import numpy as np
from numpy import fft
from scipy.interpolate import griddata

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
