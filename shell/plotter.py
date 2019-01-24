# -*- coding: utf-8 -*-
""" Implementation of shell_graphical.

author: Nicolò Lardelli
data: 07.02.18
This visualizer works with all SphericalShellsModels
1) Instantiate plotter:  
plotter = ShellPlotter('statename.hdf5')
2) Plot and return data data:
(x, y, ur, uth, uph) = plotter.plot(mode ='meridional', type='simple') 

"""

import h5py
from projection_tools import spherical, shell
import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial import legendre as leg
from matplotlib import pyplot as pp
from matplotlib import rc
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('text.latex', preamble=r'\usepackage{bm}')

def rank_1_matrix(a, b):
    # Input:
    # a: column vector
    # b: row vector

    a = np.reshape(a, (-1, 1))
    b = np.reshape(b, (-1, 1))

    return np.kron(a, b.T)


class ShellPlotter:

    def __init__(self, filename, *args, **kwargs):

        # open file and object
        self.fopen = h5py.File(filename, 'r')

        # determine type
        self.file_type = self.fopen.attrs['type']

        # retrieve parameters for visualization
        self.rratio = self.fopen['/physical/rratio'].value
        try:
            self.ro = self.fopen['/physical/ro'].value
        except KeyError as e:
            self.ro = 1/(1-self.rratio)
        self.E = self.fopen['/physical/ekman'].value

        # retrieve the runtime and the frequency
        self.time = self.fopen['/run/time'].value
        try:
            self.fN = self.fopen['/physical/omega'].value
        except:
            # assume there are no time depenencies
            self.fN = 0

        # define the phi_0 variable (meriodional point used to plot meridional sections)
        self.phi_0 = self.fN * self.time

        # retrieve the spectral resolution
        self.nN = self.fopen['truncation/spectral/dim1D'].value + 1
        self.nL = self.fopen['truncation/spectral/dim2D'].value + 1
        self.nM = self.fopen['truncation/spectral/dim3D'].value + 1

        # produce the mapping
        self.a, self.b = .5, .5 * (1 + self.rratio)/(1 - self.rratio)
        self.ri = self.ro * self.rratio

        # define title dictionary
        self.title_dict = {'simple': r'$\bm{u}_', 'curl': r'$\left(\bm{\nabla}\times\bm{u}\right)_'}

        # introduce the truncation quantities Nmax Lmax Mmax
        # the default values are (50,100,100)
        self.Nmax = kwargs.get('Nmax', 50)
        self.Lmax = kwargs.get('Lmax', 100)
        self.Mmax = kwargs.get('Nmax', 100)

        tempTor =self.fopen['/velocity/velocity_tor'][:]
        self.toroidalField = tempTor[:,:,0]+1j*tempTor[:,:,1]

        tempPol = self.fopen['/velocity/velocity_pol'][:]
        self.poloidalField = tempPol[:,:,0]+1j*tempPol[:,:,1]



    def loop_over(self, *args, **kwargs ):

        # loop_over takes care of the looping of the various modes and distinguishes between SLFm and SLFl

        if self.file_type == b'SLFl':

            idx = 0
            for l in range(self.nL):
                if l >self.Lmax:
                    continue
                for m in range(min(l + 1, self.nM)):
                    if m> self.Mmax:
                        continue
                    if m in kwargs['modes']:
                        self.evaluate_mode(l, m, idx, *args, **kwargs)

                    idx += 1

        elif self.file_type == b'SLFm':

            idx = 0
            for m in range(self.nM):

                if m > self.Mmax:
                    continue

                for l in range(m, self.nL):

                    if l > self.Lmax:
                        continue

                    if m in kwargs['modes']:
                        self.evaluate_mode(l, m, idx, *args, **kwargs)

                    idx += 1
        else:
            raise RuntimeError('Unknown  file type ' + self.file_type)

    def plot(self,mSelect='all', **kwargs):
        """
        Plot spherical shell data
        plot(mSelect = 'all', **kwargs)
        Call signatures 
        plot(mode ='meridional', type='simple') 
        plot(mode ='equatorial')
        plot(mode = 'line')
        """


        # prepare the modes
        if mSelect == 'all':
            modes = np.arange(self.nM)
        else:
            modes = np.array(mSelect)

        kwargs['modes']=modes
        # produce the grid for plotting
        xx_r, ww = cheb.chebgauss( self.Nmax)
        rr = self.a * xx_r + self.b

        # produce grid for bulk of the flow only
        delta = self.E ** .5
        #delta=0.
        if self.E >1e-4:
            delta = 0.01

        if kwargs.get('noBoundary', False)==True:
            delta=0.0

        # first if decision block over the radial grid
        if kwargs['mode']=='boundaries':


            rr = rr[(rr < self.ri + 10 * delta) | (rr > self.ro - 10 * delta)]
            xx_bulk = (rr - self.b) / self.a

        elif  kwargs['mode']=='line':

            eta = self.rratio

            # if the cut argument is standard, perform
            # a flat cut at z = 1/2(r_i+r_o)
            cut = kwargs.get('cut','orthogonal')
            if cut == 'standard':

                ymin = (eta + 1.) / (1. - eta) / 2.
                xmin = 0.
                theta_crit = np.arccos(np.abs(self.fN) / 2)
                h = ((1./(1.-eta))**2-ymin**2 )**.5
                xmax = xmin + h
                ymax = ymin

                # modify the self.phi_0 variable to 90 degrees
                self.phi_0 = np.pi/2

            elif cut =='45degree':
                v = (1+eta)/(1-eta)/np.sqrt(2)
                ymin = v
                xmin = 0
                ymax = 0
                xmax = v

            else:
                ymin = eta / (1 - eta)
                xmin = 0.
                theta_crit = np.pi/2. - np.arccos(np.abs(self.fN) / 2)
                h = ((1 + eta) / (1 - eta)) ** .5
                xmax = xmin + np.cos(theta_crit) * h
                ymax = ymin + np.sin(theta_crit) * h




            xx = np.linspace(xmin, xmax, 8*self.nN)
            yy = np.linspace(ymin, ymax, 8*self.nN)

            rr = (xx**2+yy**2)**.5
            cottheta = np.arctan2(yy,xx)

            xx2 = xx-xmin
            yy2 = yy-ymin

            xe = (xx2**2 + yy2**2)**.5

            # select the interior of the flow
            idx = (rr >= self.ri + 5 * delta) & (rr <= self.ro - 5 * delta)
            rr = rr[idx]
            xe = xe[idx]
            cottheta = cottheta[idx]
            ttheta = np.pi/2-cottheta

            xx_bulk = (rr - self.b) / self.a

        else:

            #delta = 0.
            rr = rr[(rr >= self.ri + 10 * delta) & (rr <= self.ro - 10 * delta)]
            xx_bulk = (rr - self.b) / self.a


        # second if decision block over the secundary grid
        kwargs['pphi']=None
        if kwargs['mode']=='equatorial':

            # generate the azimuthal grid
            pphi = np.linspace(0, 2 * np.pi, 2 * self.nL + 1) + self.phi_0
            kwargs['pphi'] = pphi
        elif kwargs['mode']=='line':

            self.xx_th = np.cos(ttheta)
            pass
        else:

            # generate the meridional grid
            xx_th, ww = leg.leggauss(self.nL)
            ttheta = np.arccos(xx_th)
            ttheta = np.concatenate(([np.pi], ttheta, [0]))
            self.xx_th = np.cos(ttheta)


        # third decision block, for the x/y grid generation
        if kwargs['mode']=='equatorial':

            RR, PP = np.meshgrid(rr, pphi)
            XX = np.cos(PP) * RR
            YY = np.sin(PP) * RR

        elif kwargs['mode']=='line':
            XX=rr
            pass
        elif kwargs['mode'] =='boundaries':

            RR, TT = np.meshgrid(rr, ttheta)
            XX=TT
            YY=RR
        else:

            RR, TT = np.meshgrid(rr, ttheta)
            YY = np.cos(TT) * RR
            XX = np.sin(TT) * RR

        # produce the mapping for chebyshev polynomials
        self.Tn_eval = shell.proj_radial(self.Nmax, self.a, self.b, xx_bulk) # evaluate chebyshev simple
        self.dTndr_eval = shell.proj_dradial_dr(self.Nmax, self.a, self.b, xx_bulk)  # evaluate 1/r d/dr(r Tn)
        self.Tn_r_eval = shell.proj_radial_r(self.Nmax, self.a, self.b, xx_bulk)  # evaluate 1/r Tn # this is good

        # produce the mapping for the tri-curl part
        self.Tn_r2_eval = shell.proj_radial_r2(self.Nmax, self.a, self.b, xx_bulk) # evaluate 1/r**2 Tn
        self.d2Tndr2_eval = shell.proj_lapl(self.Nmax, self.a, self.b, xx_bulk) # evaluate 1/r**2dr r**2 dr

        # decision block on how to handle the subplots
        if kwargs['mode']=='boundaries':

            # prepare the subplot
            fig, ax = pp.subplots(2, 3, sharey=False, sharex=True, figsize=(10, 2))
            ax1 = ax[:,0]
            ax2 = ax[:,1]
            ax3 = ax[:,2]
        elif kwargs['mode']=='line':
            # prepare the subplot
            fig, (ax1, ax2, ax3) = pp.subplots(1, 3, sharey=False, sharex=True, figsize=(10, 4))
        elif kwargs['mode']=='equatorial':
            # prepare the subplot
            fig, (ax1, ax2, ax3) = pp.subplots(1, 3, sharey=True, sharex=True, figsize=(10, 4))
        else:
            fig, (ax1, ax2, ax3) = pp.subplots(1, 3, sharey=True, sharex=True, figsize=(10, 8))

        if kwargs['mode'] == 'line':
            x_arg = [xe]
            if kwargs.get('geometry', False)==True:
                x_arg = [xe, xx, yy]
        else:
            x_arg = [XX, YY]

        if kwargs['mode'] != 'boundaries' and kwargs['mode'] != 'line':
            ax1.set_aspect('equal')
            ax2.set_aspect('equal')
            ax3.set_aspect('equal')

        rarg = None
        if kwargs['type']=='simple' or kwargs['type']=='curl':

            prefix = self.title_dict[kwargs['type']]

            # initialize the fields
            U_phi = np.zeros_like(XX, dtype=complex)
            U_r = np.zeros_like(XX, dtype=complex)
            U_theta = np.zeros_like(XX, dtype=complex)

            # compute (either velocity or vorticity
            self.loop_over(U_r, U_theta, U_phi, **kwargs)

            if kwargs.get('cut','orthogonal') == 'standard':

                U_s = np.sin(ttheta) *  U_r + np.cos(ttheta)* U_theta
                U_z = np.cos(ttheta) * U_r - np.sin(ttheta) * U_theta

                U_r = U_s
                U_theta = U_z

            if kwargs.get('coordinates','spherical') == 'cylindrical':

                U_s = np.sin(TT) *  U_r + np.cos(TT)* U_theta
                U_z = np.cos(TT) * U_r - np.sin(TT) * U_theta

                U_r = U_s
                U_theta = U_z

            # plot r field component
            self.plot_field(fig, ax1, *x_arg, U_r, colormap='dual', title=prefix+'{r}$', **kwargs)


            # plot theta field component
            self.plot_field(fig, ax2, *x_arg, U_theta, colormap='dual', title=prefix+'{\Theta}$', **kwargs)

            # plot phi field component
            self.plot_field(fig, ax3, *x_arg, U_phi, colormap='dual', title=prefix+'{\phi}$', **kwargs)

            rarg = (*x_arg, np.real(U_r), np.real(U_theta), np.real(U_phi))

        else:

            # initialize the fields
            U_phi = np.zeros_like(XX, dtype=complex)
            U_r = np.zeros_like(XX, dtype=complex)
            U_theta = np.zeros_like(XX, dtype=complex)

            # compute velocity
            kwargs['type']='simple'
            self.loop_over(U_r, U_theta, U_phi,  **kwargs)

            # initialize the fields
            Omega_phi = np.zeros_like(XX, dtype=complex)
            Omega_r = np.zeros_like(XX, dtype=complex)
            Omega_theta = np.zeros_like(XX, dtype=complex)

            # compute vorticity
            kwargs['type']='curl'
            self.loop_over(Omega_r, Omega_theta, Omega_phi,  **kwargs)

            # take only real parts
            U_r = np.real(U_r)
            U_theta = np.real(U_theta)
            U_phi = np.real(U_phi)
            Omega_r = np.real(Omega_r)
            Omega_theta = np.real(Omega_theta)
            Omega_phi = np.real(Omega_phi)

            Energy = U_r*U_r + U_theta*U_theta + U_phi*U_phi
            Energy *= 0.5

            Enstrophy = Omega_r**2 +  Omega_theta**2 + Omega_phi**2
            Enstrophy *= 0.5

            Helicity = U_r * Omega_r + U_theta * Omega_theta + U_phi * Omega_phi

            # plot Energy
            self.plot_field(fig, ax1, *x_arg, Energy, colormap = 'log', title=r'$\frac{1}{2}\left|\bm{u}\right|^2$', **kwargs)
            if kwargs['mode']!='boundaries' and kwargs['mode']!='line':
                ax1.set_aspect('equal')
            # plot Enstrophy
            self.plot_field(fig, ax2, *x_arg, Enstrophy, colormap='log', title=r'$\frac{1}{2}\left|\bm{\nabla}\times\bm{u}\right|^2$', **kwargs)

            # plot Helicity
            self.plot_field(fig, ax3, *x_arg, Helicity, colormap='dual', title=r'$\bm{u}\cdot\bm{\nabla}\times\bm{u}$', **kwargs)

            rarg = (*x_arg, Energy, Enstrophy, Helicity)

        return rarg

    def plot_field(self, fig, ax, *args, **kwargs):

        if kwargs['mode']=='line':
            rr = args[0]
            ff = args[1]

            ff = np.real(ff)
            min = np.nanmin(ff)
            max = np.nanmax(ff)
        else:
            XX = args[0]
            YY = args[1]
            ZZ = args[2]
            ZZ = np.real(ZZ)
            min = np.nanmin(ZZ)
            max = np.nanmax(ZZ)

        if kwargs.get('colormap','dual') == 'log':
            map = pp.get_cmap('hot')
            norm = Normalize(min, max, clip=True)

        else:
            map = pp.get_cmap('coolwarm')
            #norm = Normalize(min, max, clip=True)
            norm = MidpointNormalize(min, max, 0, clip=True)

        if kwargs['mode']=='boundaries':
            idx_boundary = int(XX.shape[1]/2)

            ax[0].contourf(XX[:, 0:idx_boundary], YY[:, 0:idx_boundary], ZZ[:, 0:idx_boundary], 50, cmap=map, norm = norm)
            ax[0].set_aspect(15.0)
            im = ax[1].contourf(XX[:, idx_boundary + 1:], YY[:, idx_boundary + 1:], ZZ[:, idx_boundary + 1:], 50, cmap=map, norm = norm)
            ax[1].set_aspect(15.0)
            ax[0].set_title(kwargs['title'])
            cb1 = fig.colorbar(im, orientation='horizontal', ax=[ax[0], ax[1]], ticks=[min, max], shrink=0.8, format='%.2E')
            #cb1 = fig.colorbar(im, orientation='horizontal', ax=[ax[0], ax[1]], shrink=0.8)
        elif kwargs['mode']=='line':
            ax.plot(rr, ff )
        else:
            im = ax.contourf(XX, YY, ZZ, 50, cmap=map, norm = norm)
            ax.set_title(kwargs['title'])
            if kwargs.get('colormap', 'dual') == 'log':
                fig.colorbar(im, orientation='horizontal', ax=ax, ticks=[min, max], shrink=0.8, format='%.2E')
                #fig.colorbar(im, orientation='horizontal', ax=ax, shrink=0.8)
                pass
            else:
                fig.colorbar(im, orientation='horizontal', ax=ax, ticks=[min, max], shrink=0.8, format='%.2E')
                #fig.colorbar(im, orientation='horizontal', ax=ax, shrink=0.8)
                pass


    def evaluate_mode(self, l, m, idx, *args, **kwargs):

        Field_r = args[0]
        Field_theta = args[1]
        Field_phi = args[2]

        # retrieve the scalar fields
        #modeT = self.fopen['velocity/velocity_tor'].value[idx, :, 0] + self.fopen['velocity/velocity_tor'].value[idx, :, 1] * 1j
        # new way of handling data
        modeT = self.toroidalField[idx,:]
        #modeP = self.fopen['velocity/velocity_pol'].value[idx, :, 0] + self.fopen['velocity/velocity_pol'].value[idx, :, 1] * 1j
        modeP = self.poloidalField[idx,:]

        modeT = modeT[:self.Nmax]
        modeP = modeP[:self.Nmax]

        if kwargs.get('symmetry', None) == 'asym':
            if (l%2) == 0:
                modeP = np.zeros_like(modeP)
            else:
                modeT = np.zeros_like(modeT)

        if kwargs.get('symmetry', None) == 'sym':
            if (l%2) == 1:
                modeP = np.zeros_like(modeP)
            else:
                modeT = np.zeros_like(modeT)

        # initialize radial parts
        # the radial parts can be interchanged if we evaluate the vorticity field or the velocity field
        # we need to invert toroidal with poloidal however
        if kwargs['type'] == 'simple':
            # procedure for the velocity field
            rad_part_ur = l * (l + 1) * np.matmul(self.Tn_r_eval, modeP)
            rad_part_pol = np.matmul(self.dTndr_eval, modeP)
            rad_part_tor = np.matmul(self.Tn_eval, modeT)

        elif kwargs['type'] == 'curl':
            # procedure for the vorticity field
            rad_part_ur = l * (l + 1) * np.matmul(self.Tn_r_eval, modeT)
            rad_part_pol = np.matmul(self.dTndr_eval, modeT)
            rad_part_tor = -( np.matmul(self.d2Tndr2_eval, modeP) - l*(l+1)*np.matmul(self.Tn_r2_eval, modeP) )
            #rad_part_tor = -(np.matmul(self.d2Tndr2_eval, modeP))


        else:
            raise RuntimeError('Unknown vector field type '+self.vector_field_type)
            pass


        if m==0:
            factor=2.
        else:
            factor=1.

        #factor=1.
        #factor=(2*l+1.)**.5


        if kwargs['mode']=='meridional' or kwargs['mode']=='boundaries':

            # prepare arrays
            eimp = np.exp(1j * m * self.phi_0)

            # initialize the theta tranforms
            Ylm = spherical.lplm(self.Lmax, l, m, self.xx_th)
            dYlm = spherical.dplm(self.Lmax, l, m, self.xx_th)
            Ylm_sin = spherical.lplm_sin(self.Lmax, l, m, self.xx_th) * m * 1j

            # update the fields poloidal parts
            Field_r += rank_1_matrix(Ylm, rad_part_ur) * eimp
            Field_theta += rank_1_matrix(dYlm, rad_part_pol) * eimp*factor
            Field_phi += rank_1_matrix(Ylm_sin, rad_part_pol) * eimp*factor

            # update the fields toroidal parts
            Field_theta += rank_1_matrix(Ylm_sin, rad_part_tor) * eimp*factor
            Field_phi -= rank_1_matrix(dYlm, rad_part_tor) * eimp*factor

            if m!=0:
                m=-m
                # prepare arrays
                eimp = np.exp(1j * m * self.phi_0)

                # initialize the theta tranforms
                #Ylm = spherical.lplm(self.nL, l, m, self.xx_th)
                #dYlm = spherical.dplm(self.nL, l, m, self.xx_th)
                Ylm_sin = -1*Ylm_sin

                # update the fields poloidal parts
                Field_r += rank_1_matrix(Ylm, np.conj(rad_part_ur)) * eimp
                Field_theta += rank_1_matrix(dYlm, np.conj(rad_part_pol)) * eimp*factor
                Field_phi += rank_1_matrix(Ylm_sin, np.conj(rad_part_pol)) * eimp*factor

                # update the fields toroidal parts
                Field_theta += rank_1_matrix(Ylm_sin, np.conj(rad_part_tor)) * eimp*factor
                Field_phi -= rank_1_matrix(dYlm, np.conj(rad_part_tor)) * eimp*factor

        elif kwargs['mode']=='line':

            # prepare arrays
            eimp = np.exp(1j * m * self.phi_0)

            # initialize the theta tranforms
            Ylm = spherical.lplm(self.nL, l, m, self.xx_th)
            dYlm = spherical.dplm(self.nL, l, m, self.xx_th)
            Ylm_sin = spherical.lplm_sin(self.nL, l, m, self.xx_th) * m * 1j

            # update the fields poloidal parts
            Field_r += Ylm* rad_part_ur * eimp
            Field_theta += dYlm* rad_part_pol * eimp *factor
            Field_phi += Ylm_sin* rad_part_pol * eimp *factor

            # update the fields toroidal parts
            Field_theta += Ylm_sin* rad_part_tor * eimp*factor
            Field_phi -= dYlm* rad_part_tor * eimp *factor

            if m != 0:

                # prepare arrays
                eimp = np.exp(1j * -1* m * self.phi_0)

                # initialize the theta tranforms
                #Ylm = spherical.lplm(self.nL, l, m, self.xx_th)
                #dYlm = spherical.dplm(self.nL, l, m, self.xx_th)
                Ylm_sin = Ylm_sin*-1

                # update the fields poloidal parts
                Field_r += Ylm * np.conj(rad_part_ur) * eimp
                Field_theta += dYlm * np.conj(rad_part_pol) * eimp *factor
                Field_phi += Ylm_sin * np.conj(rad_part_pol) * eimp *factor

                # update the fields toroidal parts
                Field_theta += Ylm_sin * np.conj(rad_part_tor) * eimp *factor
                Field_phi -= dYlm * np.conj(rad_part_tor) * eimp *factor

        else:

            # prepare arrays
            eimp = np.exp(1j * m * kwargs['pphi'])

            # initialize the theta tranforms
            Ylm = spherical.lplm(self.Lmax, l, m, np.array([0.]))
            dYlm = spherical.dplm(self.Lmax, l, m,  np.array([0.]))
            Ylm_sin = spherical.lplm_sin(self.Lmax, l, m,  np.array([0.])) * m * 1j

            # update the fields poloidal parts
            Field_r += rank_1_matrix(eimp, rad_part_ur) * Ylm[0]
            Field_theta += rank_1_matrix(eimp, rad_part_pol) * dYlm[0]*factor
            Field_phi += rank_1_matrix(eimp, rad_part_pol) * Ylm_sin[0]*factor

            # update the fields toroidal parts
            Field_theta += rank_1_matrix(eimp, rad_part_tor) * Ylm_sin[0]*factor
            Field_phi -= rank_1_matrix(eimp, rad_part_tor) * dYlm[0]*factor

            if m!=0:
                m=-1*m
                # prepare arrays
                eimp = np.exp(1j * m * kwargs['pphi'])

                # initialize the theta tranforms
                #Ylm = spherical.lplm(self.nL, l, m, np.array([0.]))
                #dYlm = spherical.dplm(self.nL, l, m, np.array([0.]))
                #Ylm_sin = spherical.lplm_sin(self.nL, l, m, np.array([0.])) * m * 1j
                Ylm_sin = -1*Ylm_sin

                # update the fields poloidal parts
                Field_r += rank_1_matrix(eimp, np.conj(rad_part_ur)) * Ylm[0]
                Field_theta += rank_1_matrix(eimp, np.conj(rad_part_pol)) * dYlm[0] *factor
                Field_phi += rank_1_matrix(eimp, np.conj(rad_part_pol)) * Ylm_sin[0] *factor

                # update the fields toroidal parts
                Field_theta += rank_1_matrix(eimp, np.conj(rad_part_tor)) * Ylm_sin[0] *factor
                Field_phi -= rank_1_matrix(eimp, np.conj(rad_part_tor)) * dYlm[0] *factor

class MidpointNormalize(Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))

"""
    def prepare_rotation(self):

"""
