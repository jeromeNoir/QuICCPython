# -*- coding: utf-8 -*-
# Nicolò Lardelli 24th November 2016

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import sys
sys.path.append('/Users/leo/quicc-github/QuICC/Scripts/Python/pybind11/')

import QuICC as quicc_pybind

class SingletonPlm(object):

    __dict = None

    instance = None
    def __init__(self, key):
        if not SingletonPlm.instance:
            SingletonPlm.instance = key
            SingletonPlm.__dict = dict()
        else:
            SingletonPlm.instance = key
            SingletonPlm.__dict = dict()

    def get_dict(self, key):

        if (SingletonPlm.instance == key).all():
            return SingletonPlm.__dict

        else:
            SingletonPlm.instance = key
            SingletonPlm.__dict = dict()
            return SingletonPlm.__dict





SingletonPlm(0)


def plm(maxl, l, m, x):

    if l < m or m < 0 or l < 0:
        return np.zeros_like(x)

    elif m in SingletonPlm.get_dict(SingletonPlm, x):
        return SingletonPlm.get_dict(SingletonPlm, x)[m][:,l-m]
    else:
        # Compute the normalized associated legendre polynomial projection matrix
        mat = np.zeros((maxl-m+1,len(x))).T

        # call the c++ function
        quicc_pybind.plm(mat, m, x)

        SingletonPlm.get_dict(SingletonPlm, x)[m] = mat

        return SingletonPlm.get_dict(SingletonPlm, x)[m][:, l-m]




def plm1(l, m, x):

    if (l,m) in SingletonPlm.get_dict(SingletonPlm,x):
        return SingletonPlm.get_dict(SingletonPlm,x)[(l,m)]
    else:
        #Compute the normalized associated legendre polynomial projection matrix
        if l<m or m<0 or l<0:
            return np.zeros_like(x)
        maxl = l
        #print(x)
        

        if (m,m) not in SingletonPlm.get_dict(SingletonPlm,x):
            SingletonPlm.get_dict(SingletonPlm, x)[(m, m)] = pmm(maxl, m, x)[:,0]

        if (m + 1, m) not in SingletonPlm.get_dict(SingletonPlm,x):
            SingletonPlm.get_dict(SingletonPlm, x)[(m + 1, m)] = np.sqrt(2.0 * m + 3.0) * x * SingletonPlm.get_dict(SingletonPlm,x)[(m,m)]




        for i, l in enumerate(range(m+2, maxl + 1)):
            #mat[:,i+2] = np.sqrt((2.0*l + 1)/(l - m))*np.sqrt((2.0*l - 1.0)/(l + m))*x*mat[:,i+1] - np.sqrt((2.0*l + 1)/(2.0*l - 3.0))*np.sqrt((l + m - 1.0)/(l + m))*np.sqrt((l - m - 1.0)/(l - m))*mat[:,i]
            if (l,m) not in SingletonPlm.get_dict(SingletonPlm,x):
                temp = np.sqrt((2.0 * l + 1) / (l - m)) * np.sqrt((2.0 * l - 1.0) / (l + m)) * x *  SingletonPlm.get_dict(SingletonPlm,x)[(l-1,m)] - np.sqrt(
                    (2.0 * l + 1) / (2.0 * l - 3.0)) * np.sqrt((l + m - 1.0) / (l + m)) * np.sqrt(
                    (l - m - 1.0) / (l - m)) *  SingletonPlm.get_dict(SingletonPlm,x)[(l-2,m)]
                SingletonPlm.get_dict(SingletonPlm,x)[(l,m)] = temp
                #print('('+str(l)+','+str(m)+')')
        return SingletonPlm.get_dict(SingletonPlm,x)[(maxl,m)]

class DictPlm(object):

    __dict = None

    instance = None
    def __init__(self, key):
        if not SingletonPlm.instance:
            SingletonPlm.instance = key
            SingletonPlm.__dict = dict()
        else:
            SingletonPlm.instance = key
            SingletonPlm.__dict = dict()

    def get_dict(self, key):

        if (SingletonPlm.instance == key).all():
            return SingletonPlm.__dict

        else:
            SingletonPlm.instance = key
            SingletonPlm.__dict = dict()
            return SingletonPlm.__dict


def plm2(l, m, x):
    if (l, m) in SingletonPlm.get_dict(SingletonPlm, x):
        return SingletonPlm.get_dict(SingletonPlm, x)[(l, m)]
    else:
        # Compute the normalized associated legendre polynomial projection matrix
        if l < m or m < 0 or l < 0:
            return np.zeros_like(x)
        maxl = l
        # print(x)

        if l==m:
            SingletonPlm.get_dict(SingletonPlm, x)[(m, m)] = pmm(maxl, m, x)[:, 0]
            return SingletonPlm.get_dict(SingletonPlm, x)[(m, m)]

        elif l==m+1:
            SingletonPlm.get_dict(SingletonPlm, x)[(m + 1, m)] = np.sqrt(2.0 * m + 3.0) * x * plm(m, m, x)
            return SingletonPlm.get_dict(SingletonPlm, x)[(m + 1, m)]

        else:
            temp = np.sqrt((2.0 * l + 1) / (l - m)) * np.sqrt((2.0 * l - 1.0) / (l + m)) * x * plm(l-1, m, x) - np.sqrt(
                (2.0 * l + 1) / (2.0 * l - 3.0)) * np.sqrt((l + m - 1.0) / (l + m)) * np.sqrt(
                (l - m - 1.0) / (l - m)) * plm(l - 2, m, x)
            SingletonPlm.get_dict(SingletonPlm, x)[(l, m)] = temp
                # print('('+str(l)+','+str(m)+')')
            return SingletonPlm.get_dict(SingletonPlm, x)[(l, m)]


"""
def plm(l, m, x):


        #Compute the normalized associated legendre polynomial projection matrix
        if l<m or m<0 or l<0:
            return np.zeros_like(x)
        maxl = l
        mat = np.zeros((len(x), maxl - m + 1))
        mat[:,0] = pmm(maxl, m, x)[:,0]
        if maxl == m:
            return mat[:,-1]

        mat[:,1] = np.sqrt(2.0*m + 3.0)*x*mat[:,0]

        for i, l in enumerate(range(m+2, maxl + 1)):
            mat[:,i+2] = np.sqrt((2.0*l + 1)/(l - m))*np.sqrt((2.0*l - 1.0)/(l + m))*x*mat[:,i+1] - np.sqrt((2.0*l + 1)/(2.0*l - 3.0))*np.sqrt((l + m - 1.0)/(l + m))*np.sqrt((l - m - 1.0)/(l - m))*mat[:,i]

        #SingletonPlm.get_dict(SingletonPlm,x)[(l,m)]=mat[:,-1]
        return mat[:,-1]

"""
def pmm(maxl, m, x):
    """Compute the normalized associated legendre polynomial of order and degree m"""

    mat = np.zeros((len(x), 1))

    # orthogonality as Schäffer 2013
    mat[:,0] = 1.0/np.sqrt(4*np.pi)
    sx = np.sqrt(1.0 - x**2)
    for i in range(1, m+1):
        mat[:,0] = -np.sqrt((2.0*i + 1.0)/(2.0*i))*sx*mat[:,0]

    return mat

def lplm(maxl, l, m, x = None):

    # return an orthonormal assoc legendre function
    # wapper for plm
    if x is None:
        xx, ww = leg.leggauss(maxl)
    else:
        xx = np.array(x)

    y = plm(maxl, l, m, xx)
    return y

"""
def dplm0(l, m, x):
    # return the deriavative of an orthonormal assoc legendre func
    # implementation Hollerbach style
    sin_the = (1 - x ** 2) ** .5

    y = spe.lpmv(m,l+1,x)*l*(l+1-m) -spe.lpmv(m,l-1,x)*(l+1)*(l+m)/(2*l+1)

    return y*(spe.gamma(l-m+1)/spe.gamma(l+m+1)*(2*l+1)/4/np.pi)**.5/sin_the/(2*l+1)

def dplm1(l, m, x):

    sin_the = (1 - x ** 2) ** .5

    y = l* x * spe.lpmv(m,l,x) - (l+m)*spe.lpmv(m,l-1,x)

    return y * (spe.gamma(l - m + 1) / spe.gamma(l + m + 1) * (2 * l + 1) / 4 / np.pi) ** .5 /sin_the

def dplm2(l, m, x):

    sin_the = (1 - x ** 2) ** .5

    y = -(l+1) * x * spe.lpmv(m,l,x) +(l-m+1)*spe.lpmv(m,l+1,x)

    return y * (spe.gamma(l - m + 1) / spe.gamma(l + m + 1) * (2 * l + 1) / 4 / np.pi) ** .5 /sin_the

def dplm3(l, m, x):

    sin_the = (1 - x ** 2) ** .5

    y = sin_the* spe.lpmv(m+1,l,x) +m*x*spe.lpmv(m,l,x)

    return y * (spe.gamma(l - m + 1) / spe.gamma(l + m + 1) * (2 * l + 1) / 4 / np.pi) ** .5 /sin_the

def dplm4(l, m, x):

    sin_the = (1 - x ** 2) ** .5

    y = -(l+m) * (l-m+1)*sin_the * spe.lpmv(m-1,l,x) -m *x*spe.lpmv(m,l,x)

    return y * (spe.gamma(l - m + 1) / spe.gamma(l + m + 1) * (2 * l + 1) / 4 / np.pi) ** .5 /sin_the

def dplm_1(l, m, x):
    # derivative associated legendre function
    # implementation Hollerbach style, stable on the poles
    # fully normalized

    x = np.array(x)
    if(l==0 and m==0):
        return np.zeros_like(x)
    y = -1./2*((l+m)*(l-m+1)*spe.lpmv(m-1,l,x)-spe.lpmv(m+1,l,x))

    return y * (spe.gamma(l - m + 1) / spe.gamma(l + m + 1) * (2 * l + 1) / 4 / np.pi) ** .5

def dplm_1(l, m, x):

    x = np.array(x)
    y = (l+1)*(l+m)*(spe.lpmv(m+1,l-2,x)+(l+m-2)*(l+m-1)*spe.lpmv(m-1,l-2,x))
    y -= l*(l-m+1)*(spe.lpmv(m+1,l,x)+(l+m)*(l+m+1)*spe.lpmv(m-1,l,x))

    return y * (spe.gamma(l - m + 1) / spe.gamma(l + m + 1) * (2 * l + 1) / 4 / np.pi) ** .5 /((2*l+1)*2*m)

def lplm_sin_1(l, m, x):


    x = np.array(x)
    if m!=0:
        y = -1/2/m*(spe.lpmv(m+1, l-1, x) + (l+m-1)*(l+m)*spe.lpmv(m-1, l-1,x))
    else:
        y = spe.lpmv(m, l, x)/(1-x**2)**.5

    return y*(spe.gamma(l-m+1)/spe.gamma(l+m+1)*(2*l+1)/4/np.pi)**.5


def deipm(l, m, phi):
    phi = np.array(phi)
    # compute the derivative  wrt phi of the azimuthal part of  Y_l^m
    return 1j*m*np.exp(m*phi*1j)

"""

def dplm(maxl, l, m, x = None):

    # returns the derivative associated legendre function
    # implementation Hollerbach style, stable on the poles
    # fully normalized
    #m= m-1
    if x is None:
        xx, ww = leg.leggauss(maxl)
    else:
        xx = np.array(x)

    y = -1./2*(((l+m)*(l-m+1))**0.5*plm(maxl, l,m-1,xx)-((l-m)*(l+m+1))**.5*plm(maxl, l, m+1, xx))

    return y


def lplm_sin(maxl, l, m, x = None):

    # return associated legendre function /sin_theta
    # implemented with the recurrence relation for P_l^m(x)/(1-x**2)**.5 when possible (aka m!=0)
    # fully normalized

    if x is None:
        xx, ww = leg.leggauss(maxl)
    else:
        xx = np.array(x)

    if m!=0:
        y = -1/2/m*(((l-m)*(l-m-1))**.5 * plm(maxl, l-1, m+1, xx) + ((l+m)*(l+m-1))**.5 *plm(maxl, l-1, m-1, xx))*((2*l+1)/(2*l-1))**.5
    else:
        y = plm(maxl, l, m, xx)/(1-xx**2)**.5

    return y

def eipm(l, m, phi):

    # returns the azimuthal part of a Ylm spherical harmonic
    # the spherical harmonic is fully normalized, but the normalization
    # is hidden in the associated legendre function part
    phi = np.array(phi)

    # compute the azimuthal part of spherical harmonics e^{imphi}
    return np.exp(m*phi*1j)


if __name__=="__main__":
    # main function, used for testing the routines

    theta=np.linspace(0,np.pi,100)
    x = np.cos(theta)
    phi = np.linspace(0, 2*np.pi,10)







