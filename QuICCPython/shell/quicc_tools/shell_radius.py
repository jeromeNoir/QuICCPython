"""Module provides functions to generate sparse operators for the radial direction in a spherical shell."""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp

import QuICCPython.shell.quicc_tools.utils as utils
import QuICCPython.shell.quicc_tools.shell_radius_boundary as radbc


def zblk(nr, bc):
    """Create a block of zeros"""

    mat = spsp.lil_matrix((nr,nr))
    return radbc.constrain(mat,bc)

def r1(nr, a, b, bc, coeff = 1.0, zr = 0):
    """Create operator for r multiplication"""

    ns = np.arange(0, nr)
    offsets = np.arange(-1,2)
    nzrow = -1

    # Generate 1st subdiagonal
    def d_1(n):
        return np.full(n.shape, a/2.0)

    # Generate diagonal
    def d0(n):
        return np.full(n.shape, b)

    # Generate 1st superdiagonal
    def d1(n):
        return d_1(n)

    ds = [d_1, d0, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    if zr > 0:
        mat = mat.tolil()
        mat[-zr:,:] = 0
        mat = mat.tocoo()
    return radbc.constrain(mat, bc)

def r2(nr, a, b, bc, coeff = 1.0, zr = 0):
    """Create operator for r^2 multiplication"""

    ns = np.arange(0, nr)
    offsets = np.arange(-2,3)
    nzrow = -1

    # Generate 2nd subdiagonal
    def d_2(n):
        return np.full(n.shape, a**2/4.0)

    # Generate 1st subdiagonal
    def d_1(n):
        return np.full(n.shape, a*b)

    # Generate diagonal
    def d0(n):
        return np.full(n.shape, (a**2 + 2.0*b**2)/2.0)

    # Generate 1st superdiagonal
    def d1(n):
        return d_1(n)

    # Generate 2nd superdiagonal
    def d2(n):
        return d_2(n)

    ds = [d_2, d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    if zr > 0:
        mat = mat.tolil()
        mat[-zr:,:] = 0
        mat = mat.tocoo()
    return radbc.constrain(mat, bc)

def r4(nr, a, b, bc, coeff = 1.0, zr = 0):
    """Create operator for r^4 multiplication"""

    ns = np.arange(0, nr)
    offsets = np.arange(-4,5)
    nzrow = -1

    # Generate 4th subdiagonal
    def d_4(n):
        return np.full(n.shape, a**4/16.0)

    # Generate 3rd subdiagonal
    def d_3(n):
        return np.full(n.shape, a**3*b/2.0)

    # Generate 2nd subdiagonal
    def d_2(n):
        return np.full(n.shape, a**2*(a**2 + 6.0*b**2)/4.0)

    # Generate 1st subdiagonal
    def d_1(n):
        return np.full(n.shape, a*b*(3.0*a**2 + 4.0*b**2)/2.0)

    # Generate diagonal
    def d0(n):
        return np.full(n.shape, (3.0*a**4 + 24.0*a**2*b**2 + 8.0*b**4)/8.0)

    # Generate 1st superdiagonal
    def d1(n):
        return  d_1(n)

    # Generate 2nd superdiagonal
    def d2(n):
        return d_2(n)

    # Generate 3rd superdiagonal
    def d3(n):
        return d_3(n)

    # Generate 4th superdiagonal
    def d4(n):
        return d_4(n)

    ds = [d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    if zr > 0:
        mat = mat.tolil()
        mat[-zr:,:] = 0
        mat = mat.tocoo()
    return radbc.constrain(mat, bc)

def d1(nr, a, b, bc, coeff = 1.0, zr = 1):
    """Create operator for 1st derivative"""

    row = [2*j for j in range(0,nr)]
    mat = spsp.lil_matrix((nr,nr))
    for i in range(0,nr-1):
        mat[i,i+1:nr:2] = row[i+1:nr:2]
    mat[-zr:,:] = 0

    mat = coeff*(1.0/a)*mat
    return radbc.constrain(mat, bc, location = 'b')

def d2(nr, a, b, bc, coeff = 1.0, zr = 2):
    """Create operator for 2nd derivative"""

    mat = spsp.lil_matrix((nr,nr))
    for i in range(0,nr-2):
        mat[i,i+2:nr:2] = [j*(j**2 - i**2) for j in range(0,nr)][i+2:nr:2]
    mat[-zr:,:] = 0

    mat = coeff*(1.0/a**2)*mat
    return radbc.constrain(mat, bc, location = 'b')

def i1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 1st integral T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-1,2,2)
    nzrow = 0

    # Generate 1st subdiagonal
    def d_1(n):
        return a/(2.0*n)

    # Generate 1st superdiagonal
    def d1(n):
        return -d_1(n)

    ds = [d_1, d1]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i1r1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 1st integral r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-2,3)
    nzrow = 0

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2/(4.0*n)

    # Generate 1st subdiagonal
    def d_1(n):
        return a*b/(2.0*n)

    # Generate diagonal
    def d0(n):
        return 0

    # Generate 1st superdiagonal
    def d1(n):
        return -d_1(n)

    # Generate 2nd superdiagonal
    def d2(n):
        return -d_2(n)

    ds = [d_2, d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i2(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-2,3,2)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2/(4.0*n*(n - 1.0))

    # Generate diagonal
    def d0(n):
        return -a**2/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return d_2(n+1.0)

    ds = [d_2, d0, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i2r2(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r^2 T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-4,5)
    nzrow = 1

    # Generate 4th subdiagonal
    def d_4(n):
        return a**4/(16.0*n*(n - 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return (a**3*b)/(4.0*n*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return (a**2*(2.0*b**2*n + a**2 + 2.0*b**2))/(8.0*n*(n - 1.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -(a**3*b)/(4.0*n*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return -(a**2*(a**2 + 4.0*b**2))/(8.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return d_1(n - 1.0)

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*(a**2 - 2.0*b**2*n + 2.0*b**2)/(8.0*n*(n - 1.0)*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return d_3(n + 1.0)

    # Generate 4th superdiagonal
    def d4(n):
        return d_4(n + 1.0)

    ds = [d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i2r3(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r^3 T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-5,6)
    nzrow = 1

    # Generate 5th subdiagonal
    def d_5(n):
        return a**5/(32.0*n*(n - 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return 3.0*a**4*b/(16.0*n*(n - 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return a**3*(a**2*n + 3.0*a**2 + 12.0*b**2*n + 12.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2*b*(3.0*a**2 + 2.0*b**2*n + 2.0*b**2)/(8.0*n*(n - 1.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -a**3*(a**2 + 6.0*b**2)/(16.0*n*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return -a**2*b*(3.0*a**2 + 4.0*b**2)/(8.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -a**3*(a**2 + 6.0*b**2)/(16.0*n*(n - 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*b*(3.0*a**2 - 2.0*b**2*n + 2.0*b**2)/(8.0*n*(n - 1.0)*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return a**3*(a**2*n - 3.0*a**2 + 12.0*b**2*n - 12.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0))

    # Generate 4th superdiagonal
    def d4(n):
        return 3.0*a**4*b/(16.0*n*(n + 1.0))

    # Generate 5th superdiagonal
    def d5(n):
        return a**5/(32.0*n*(n + 1.0))

    ds = [d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i2r2lapl(nr, l, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r^2 Laplacian T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-2,3)
    nzrow = 1

    # Generate 2nd subdiagonal
    def d_2(n):
        return -(a**2*(l - n + 2.0)*(l + n - 1.0))/(4.0*n*(n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return (a*b*(n - 1.0))/n

    # Generate main diagonal
    def d0(n):
        return (a**2*l**2 + a**2*l + a**2*n**2 - a**2 + 2.0*b**2*n**2 - 2.0*b**2)/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return a*b*(n + 1.0)/n

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*(l - n - 1.0)*(l + n + 2.0)/(4.0*n*(n + 1.0))

    ds = [d_2, d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i2r3lapl(nr, l, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r^3 Laplacian T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-3,4)
    nzrow = 1

    # Generate 3rd subdiagonal
    def d_3(n):
        return -a**3*(l - n + 3.0)*(l + n - 2.0)/(8.0*n*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -a**2*b*(l**2 + l - 3.0*n**2 + 11.0*n - 10.0)/(4.0*n*(n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return a*(a**2*l**2 + a**2*l + 3.0*a**2*n**2 - a**2*n - 6.0*a**2 + 12.0*b**2*n**2 - 4.0*b**2*n - 16.0*b**2)/(8.0*n*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return b*(a**2*l**2 + a**2*l + 3.0*a**2*n**2 - 5.0*a**2 + 2.0*b**2*n**2 - 2.0*b**2)/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return a*(a**2*l**2 + a**2*l + 3.0*a**2*n**2 + a**2*n - 6.0*a**2 + 12.0*b**2*n**2 + 4.0*b**2*n - 16.0*b**2)/(8.0*n*(n - 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*b*(l**2 + l - 3.0*n**2 - 11.0*n - 10.0)/(4.0*n*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -a**3*(l - n - 2.0)*(l + n + 3.0)/(8.0*n*(n + 1.0))

    ds = [d_3, d_2, d_1, d0, d1, d2, d3]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-4,5,2)
    nzrow = 3

    # Generate 4th subdiagonal
    def d_4(n):
        return a**4/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -a**4/(4.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0))

    # Generate diagonal
    def d0(n):
        return 3.0*a**4/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**4/(4.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 4th superdiagonal
    def d4(n):
        return a**4/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_4, d_2, d0, d2, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-5,6)
    nzrow = 3

    # Generate 5th subdiagonal
    def d_5(n):
        return a**5/(32.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return a**4*b/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -3.0*a**5/(32.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -a**4*b/(4.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return a**5*(n - 8.0)/(16.0*n*(n - 3.0)*(n - 2.0)*(n + 1.0)*(n + 2.0))

    # Generate main diagonal
    def d0(n):
        return 3.0*a**4*b/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0)) 

    # Generate 1st superdiagonal
    def d1(n):
        return a**5*(n + 8.0)/(16.0*n*(n - 2.0)*(n - 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**4*b/(4.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -3.0*a**5/(32.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 4th superdiagonal
    def d4(n):
        return a**4*b/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 5th superdiagonal
    def d5(n):
        return a**5/(32.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r1d1r1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r D r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-5,6)
    nzrow = 3

    # Generate 5th subdiagonal
    def d_5(n):
        return a**5*(n - 4.0)/(32.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return a**4*b*(2.0*n - 7.0)/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -a**3*(a**2*n - 8.0*a**2 - 4.0*b**2*n - 4.0*b**2)/(32.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -a**4*b*(n - 4.0)/(4.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -a**3*(a**2*n**2 + 2.0*a**2*n - 20.0*a**2 + 6.0*b**2*n**2 - 6.0*b**2*n - 36.0*b**2)/(16.0*n*(n - 3.0)*(n - 2.0)*(n + 1.0)*(n + 2.0))

    # Generate main diagonal
    def d0(n):
        return -9.0*a**4*b/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 1st superdiagonal
    def d1(n):
        return a**3*(a**2*n**2 - 2.0*a**2*n - 20.0*a**2 + 6.0*b**2*n**2 + 6.0*b**2*n - 36.0*b**2)/(16.0*n*(n - 2.0)*(n - 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return a**4*b*(n + 4.0)/(4.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return a**3*(a**2*n + 8.0*a**2 - 4.0*b**2*n + 4.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 4th superdiagonal
    def d4(n):
        return -a**4*b*(2.0*n + 7.0)/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 5th superdiagonal
    def d5(n):
        return -a**5*(n + 4.0)/(32.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r3d2(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^3 D^2 T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-5,6)
    nzrow = 3

    # Generate 5th subdiagonal
    def d_5(n):
        return a**5*(n - 6.0)*(n - 5.0)/(32.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return 3.0*a**4*b*(n - 5.0)*(n - 4.0)/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return a**3*(a**2*n**2 + 7.0*a**2*n - 54.0*a**2 + 12.0*b**2*n**2 - 36.0*b**2*n - 48.0*b**2)/(32.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2*b*(15.0*a**2*n - 57.0*a**2 + 2.0*b**2*n**2 - 4.0*b**2*n - 6.0*b**2)/(8.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -a**3*(a**2*n**3 - 9.0*a**2*n**2 - 16.0*a**2*n + 132.0*a**2 + 6.0*b**2*n**3 - 54.0*b**2*n**2 + 12.0*b**2*n + 288.0*b**2)/(16.0*n*(n - 3.0)*(n - 2.0)*(n + 1.0)*(n + 2.0))

    # Generate main diagonal
    def d0(n):
        return -a**2*b*(3.0*a**2*n**2 - 66.0*a**2 + 4.0*b**2*n**2 - 16.0*b**2)/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -a**3*(a**2*n**3 + 9.0*a**2*n**2 - 16.0*a**2*n - 132.0*a**2 + 6.0*b**2*n**3 + 54.0*b**2*n**2 + 12.0*b**2*n - 288.0*b**2)/(16.0*n*(n - 2.0)*(n - 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*b*(15.0*a**2*n + 57.0*a**2 - 2.0*b**2*n**2 - 4.0*b**2*n + 6.0*b**2)/(8.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return a**3*(a**2*n**2 - 7.0*a**2*n - 54.0*a**2 + 12.0*b**2*n**2 + 36.0*b**2*n - 48.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 4th superdiagonal
    def d4(n):
        return 3.0*a**4*b*(n + 4.0)*(n + 5.0)/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 5th superdiagonal
    def d5(n):
        return a**5*(n + 5.0)*(n + 6.0)/(32.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r4(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^4 T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-8,9)
    nzrow = 3

    # Generate 8th subdiagonal
    def d_8(n):
        return coeff*a**8/(256.0*n*(n - 1.0)*(n - 2.0)*(n - 3.0))

    # Generate 7th subdiagonal
    def d_7(n):
        return coeff*(a**7*b)/(32.0*n*(n - 1.0)*(n - 2.0)*(n - 3.0))

    # Generate 6th subdiagonal
    def d_6(n):
        return coeff*(3.0*a**6*(2.0*b**2*n + a**2 + 2.0*b**2))/(64.0*n*(n - 1.0)*(n - 2.0)*(n - 3.0)*(n + 1.0))

    # Generate 5th subdiagonal
    def d_5(n):
        return -coeff*(a**5*b*(a**2*n - 4.0*b**2*n - 11.0*a**2 - 4.0*b**2))/(32.0*n*(n - 1.0)*(n - 2.0)*(n - 3.0)*(n + 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return -coeff*(a**4*(a**4*n**2 - 19.0*a**4 + 12.0*a**2*b**2*n**2 - 36.0*a**2*b**2*n - 120.0*a**2*b**2 - 4.0*b**4*n**2 - 12.0*b**4*n - 8.0*b**4))/(64.0*n*(n - 1.0)*(n - 2.0)*(n - 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -coeff*(3.0*a**5*b*(a**2*n + 4.0*b**2*n + 6.0*a**2 + 8.0*b**2))/(32.0*n*(n - 1.0)*(n - 2.0)*(n + 2.0)*(n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -coeff*(a**4*(9.0*a**4*n + 33.0*a**4 + 6.0*a**2*b**2*n**2 + 120*a**2*b**2*n + 306.0*a**2*b**2 + 16.0*b**4*n**2 + 80.0*b**4*n + 96.0*b**4))/(64.0*n*(n - 1.0)*(n - 3.0)*(n + 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return coeff*(a**5*b*(3.0*a**2*n**2 - 15.0*a**2*n - 102.0*a**2 + 8.0*b**2*n**2 - 40.0*b**2*n - 192.0*b**2))/(32.0*n*(n - 2.0)*(n - 3.0)*(n + 3.0)*(n + 2.0)*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return coeff*(3.0*a**4*(a**4*n**2 - 29.0*a**4 + 16.0*a**2*b**2*n**2 - 304.0*a**2*b**2 + 16.0*b**4*n**2 - 144.0*b**4))/(128.0*(n - 1.0)*(n - 2.0)*(n - 3.0)*(n + 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return coeff*(a**5*b*(3.0*a**2*n**2 + 15.0*a**2*n - 102.0*a**2 + 8.0*b**2*n**2 + 40.0*b**2*n - 192.0*b**2))/(32.0*n*(n - 1.0)*(n - 2.0)*(n - 3.0)*(n + 3.0)*(n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return coeff*(a**4*(9.0*a**4*n - 33.0*a**4 - 6.0*a**2*b**2*n**2 + 120.0*a**2*b**2*n - 306.0*a**2*b**2 - 16.0*b**4*n**2 + 80.0*b**4*n - 96.0*b**4))/(64.0*n*(n - 1.0)*(n - 2.0)*(n - 3.0)*(n + 3.0)*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -coeff*(3.0*a**5*b*(a**2*n + 4.0*b**2*n - 6.0*a**2 - 8.0*b**2))/(32.0*n*(n - 1.0)*(n - 2.0)*(n + 2.0)*(n + 1.0))

    # Generate 4th superdiagonal
    def d4(n):
        return -coeff*(a**4*(a**4*n**2 - 19.0*a**4 + 12.0*a**2*b**2*n**2 + 36.0*a**2*b**2*n - 120.0*a**2*b**2 - 4.0*b**4*n**2 + 12.0*b**4*n - 8.0*b**4))/(64.0*n*(n - 1.0)*(n - 2.0)*(n + 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 5th superdiagonal
    def d5(n):
        return -coeff*(a**5*b*(a**2*n - 4.0*b**2*n + 11.0*a**2 + 4.0*b**2))/(32.0*n*(n - 1.0)*(n + 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 6th superdiagonal
    def d6(n):
        return -coeff*(3.0*a**6*(a**2 - 2.0*b**2*n + 2.0*b**2))/(64.0*n*(n - 1.0)*(n + 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 7th superdiagonal
    def d7(n):
        return d_7(n + 3.0)

    # Generate 8th superdiagonal
    def d8(n):
        return d_8(n + 3.0)

    ds = [d_8, d_7, d_6, d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5, d6, d7, d8]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r4laplrd1r1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^4 laplacian 1/r D r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-5,6)
    nzrow = 3

    # Generate 5th subdiagonal
    def d_5(n):
        return a**5*(n - 6.0)*(n - 5.0)*(n - 4.0)/(32.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return a**4*b*(n - 5.0)*(n - 4.0)*(4.0*n - 15.0)/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return 3.0*a**3*(n - 4.0)*(a**2*n**2 - a**2*n - 14.0*a**2 + 8.0*b**2*n**2 - 20.0*b**2*n - 28.0*b**2)/(32.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2*b*(4.0*a**2*n**3 - 12.0*a**2*n**2 - 67.0*a**2*n + 213.0*a**2 + 8.0*b**2*n**3 - 42.0*b**2*n**2 + 28.0*b**2*n + 78.0*b**2)/(8.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return a*(a**4*n**4 + 7.0*a**4*n**3 - 52.0*a**4*n**2 - 52.0*a**4*n + 384.0*a**4 + 12.0*a**2*b**2*n**4 + 30.0*a**2*b**2*n**3 - 354.0*a**2*b**2*n**2 - 12.0*a**2*b**2*n + 1440.0*a**2*b**2 + 8.0*b**4*n**4 - 16.0*b**4*n**3 - 56.0*b**4*n**2 + 64.0*b**4*n + 96.0*b**4)/(16.0*n*(n - 3.0)*(n - 2.0)*(n + 1.0)*(n + 2.0))

    # Generate main diagonal
    def d0(n):
        return 9.0*a**2*b*(3.0*a**2*n**2 - 26.0*a**2 + 4.0*b**2*n**2 - 16.0*b**2)/(8.0*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -a*(a**4*n**4 - 7.0*a**4*n**3 - 52.0*a**4*n**2 + 52.0*a**4*n + 384.0*a**4 + 12.0*a**2*b**2*n**4 - 30.0*a**2*b**2*n**3 - 354.0*a**2*b**2*n**2 + 12.0*a**2*b**2*n + 1440.0*a**2*b**2 + 8.0*b**4*n**4 + 16.0*b**4*n**3 - 56.0*b**4*n**2 - 64.0*b**4*n + 96.0*b**4)/(16.0*n*(n - 2.0)*(n - 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*b*(4.0*a**2*n**3 + 12.0*a**2*n**2 - 67.0*a**2*n - 213.0*a**2 + 8.0*b**2*n**3 + 42.0*b**2*n**2 + 28.0*b**2*n - 78.0*b**2)/(8.0*n*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -3.0*a**3*(n + 4.0)*(a**2*n**2 + a**2*n - 14.0*a**2 + 8.0*b**2*n**2 + 20.0*b**2*n - 28.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 4th superdiagonal
    def d4(n):
        return -a**4*b*(n + 4.0)*(n + 5.0)*(4.0*n + 15.0)/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 5th superdiagonal
    def d5(n):
        return -a**5*(n + 4.0)*(n + 5.0)*(n + 6.0)/(32.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r4lapl(nr, l, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^4 Laplacian T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-6,7)
    nzrow = 3

    # Generate 6th subdiagonal
    def d_6(n):
        return -(a**6*(l - n + 6.0)*(l + n - 5.0))/(64.0*n*(n - 3.0)*(n - 1.0)*(n - 2.0))

    # Generate 5th subdiagonal
    def d_5(n):
        return -(a**5*b*(l**2 + l - 2.0*n**2 + 19.0*n - 45.0))/(16.0*n*(n - 3.0)*(n - 1.0)*(n - 2.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return (a**4*(a**2*l**2*n - 5.0*a**2*l**2 + a**2*l*n - 5.0*a**2*l + a**2*n**3 - 3.0*a**2*n**2 - 28.0*a**2*n + 96.0*a**2 - 2.0*b**2*l**2*n - 2.0*b**2*l**2 - 2.0*b**2*l*n - 2.0*b**2*l + 12.0*b**2*n**3 - 84.0*b**2*n**2 + 96.0*b**2*n + 192.0*b**2))/(32.0*n*(n - 1.0)*(n - 2.0)*(n - 3.0)*(n + 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return (a**3*b*(3.0*a**2*l**2 + 3.0*a**2*l + 2.0*a**2*n**2 + 11.0*a**2*n - 75.0*a**2 + 8.0*b**2*n**2 - 20.0*b**2*n - 28.0*b**2))/(16.0*n*(n - 1.0)*(n - 2.0)*(n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return (a**2*(a**4*l**2*n + 17.0*a**4*l**2 + a**4*l*n + 17.0*a**4*l - a**4*n**3 + 24.0*a**4*n**2 - 5.0*a**4*n - 294.0*a**4 + 16.0*a**2*b**2*l**2*n + 32.0*a**2*b**2*l**2 + 16.0*a**2*b**2*l*n + 32.0*a**2*b**2*l + 192.0*a**2*b**2*n**2 - 288.0*a**2*b**2*n - 1344.0*a**2*b**2 + 16.0*b**4*n**3 - 112.0*b**4*n - 96.0*b**4))/(64.0*n*(n - 1.0)*(n - 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -(a**3*b*(a**2*l**2*n - 8.0*a**2*l**2 + a**2*l*n - 8.0*a**2*l + 2.0*a**2*n**3 - 15.0*a**2*n**2 - 23.0*a**2*n + 180.0*a**2 + 4.0*b**2*n**3 - 30.0*b**2*n**2 + 2.0*b**2*n + 156.0*b**2))/(8.0*n*(n - 2.0)*(n - 3.0)*(n + 2.0)*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return -(a**2*(a**4*l**2*n**2 - 19.0*a**4*l**2 + a**4*l*n**2 - 19.0*a**4*l + a**4*n**4 - 37.0*a**4*n**2 + 312.0*a**4 + 6.0*a**2*b**2*l**2*n**2 - 54.0*a**2*b**2*l**2 + 6.0*a**2*b**2*l*n**2 - 54.0*a**2*b**2*l + 12.0*a**2*b**2*n**4 - 300.0*a**2*b**2*n**2 + 1728.0*a**2*b**2 + 8.0*b**4*n**4 - 104.0*b**4*n**2 + 288.0*b**4))/(16.0*(n - 1.0)*(n - 2.0)*(n - 3.0)*(n + 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -(a**3*b*(a**2*l**2*n + 8.0*a**2*l**2 + a**2*l*n + 8.0*a**2*l + 2.0*a**2*n**3 + 15.0*a**2*n**2 - 23.0*a**2*n - 180.0*a**2 + 4.0*b**2*n**3 + 30.0*b**2*n**2 + 2.0*b**2*n - 156.0*b**2))/(8.0*n*(n - 1.0)*(n - 2.0)*(n + 3.0)*(n + 2.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return (a**2*(a**4*l**2*n - 17.0*a**4*l**2 + a**4*l*n - 17.0*a**4*l - a**4*n**3 - 24.0*a**4*n**2 - 5.0*a**4*n + 294.0*a**4 + 16.0*a**2*b**2*l**2*n - 32.0*a**2*b**2*l**2 + 16.0*a**2*b**2*l*n - 32.0*a**2*b**2*l - 192.0*a**2*b**2*n**2 - 288.0*a**2*b**2*n + 1344.0*a**2*b**2 + 16.0*b**4*n**3 - 112.0*b**4*n + 96.0*b**4))/(64.0*n*(n - 1.0)*(n - 2.0)*(n + 3.0)*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return (a**3*b*(3.0*a**2*l**2 + 3.0*a**2*l + 2.0*a**2*n**2 - 11.0*a**2*n - 75.0*a**2 + 8.0*b**2*n**2 + 20.0*b**2*n - 28.0*b**2))/(16.0*n*(n - 1.0)*(n + 2.0)*(n + 1.0))

    # Generate 4th superdiagonal
    def d4(n):
        return (a**4*(a**2*l**2*n + 5.0*a**2*l**2 + a**2*l*n + 5.0*a**2*l + a**2*n**3 + 3.0*a**2*n**2 - 28.0*a**2*n - 96.0*a**2 - 2.0*b**2*l**2*n + 2.0*b**2*l**2 - 2.0*b**2*l*n + 2.0*b**2*l + 12.0*b**2*n**3 + 84.0*b**2*n**2 + 96.0*b**2*n - 192.0*b**2))/(32.0*n*(n - 1.0)*(n + 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 5th superdiagonal
    def d5(n):
        return -(a**5*b*(l**2 + l - 2.0*n**2 - 19.0*n - 45.0))/(16.0*n*(n + 3.0)*(n + 2.0)*(n + 1.0))

    # Generate 6th superdiagonal
    def d6(n):
        return -(a**6*(l - n - 5.0)*(l + n + 6.0))/(64.0*n*(n + 3.0)*(n + 2.0)*(n + 1.0))

    ds = [d_6, d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5, d6]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r2lapl2_l1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^2 Laplacian^2 T_n(x) for l = 1."""

    ns = np.arange(0, nr)
    offsets = np.arange(-2,3)
    nzrow = 3

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2*(n - 5.0)/(4.0*(n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return a*b*(n - 2.0)/n

    # Generate main diagonal
    def d0(n):
        return (a**2*n**2 + 3.0*a**2 + 2.0*b**2*n**2 - 2.0*b**2)/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return a*b*(n + 2.0)/n

    # Generate 2nd superdiagonal
    def d2(n):
        return a**2*(n + 5.0)/(4.0*(n + 1.0))

    ds = [d_2, d_1, d0, d1, d2]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r4lapl2(nr, l, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^4 Laplacian^2 T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-4,5)
    nzrow = 3

    # Generate 4th subdiagonal
    def d_4(n):
        return (a**4*(l - n + 6.0)*(l + n - 5.0)*(l - n + 4.0)*(l + n - 3.0))/(16.0*n*(n - 3.0)*(n - 1.0)*(n - 2.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -(a**3*b*(n - 4.0)*(l**2 + l - n**2 + 8.0*n - 15.0))/(2.0*n*(n - 1.0)*(n - 2.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -(a**2*(a**2*l**4 + 2.0*a**2*l**3 + 5.0*a**2*l**2*n - 20.0*a**2*l**2 + 5.0*a**2*l*n - 21.0*a**2*l - a**2*n**4 + 9.0*a**2*n**3 - 17.0*a**2*n**2 - 39.0*a**2*n + 108.0*a**2 + 2.0*b**2*l**2*n**2 - 4.0*b**2*l**2*n - 6.0*b**2*l**2 + 2.0*b**2*l*n**2 - 4.0*b**2*l*n - 6.0*b**2*l - 6.0*b**2*n**4 + 54.0*b**2*n**3 - 138.0*b**2*n**2 + 18.0*b**2*n + 216.0*b**2))/(4.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return (a*b*(a**2*l**2*n - 8.0*a**2*l**2 + a**2*l*n - 8.0*a**2*l + 3.0*a**2*n**3 - 12.0*a**2*n**2 - 15.0*a**2*n + 72.0*a**2 + 4.0*b**2*n**3 - 16.0*b**2*n**2 + 4.0*b**2*n + 24.0*b**2))/(2.0*n*(n + 1.0)*(n - 2.0))

    # Generate main diagonal
    def d0(n):
        return (3.0*a**4*l**4 + 6.0*a**4*l**3 + 2.0*a**4*l**2*n**2 - 47.0*a**4*l**2 + 2.0*a**4*l*n**2 - 50.0*a**4*l + 3.0*a**4*n**4 - 51.0*a**4*n**2 + 228.0*a**4 + 8.0*a**2*b**2*l**2*n**2 - 32.0*a**2*b**2*l**2 + 8.0*a**2*b**2*l*n**2 - 32.0*a**2*b**2*l + 24.0*a**2*b**2*n**4 - 264.0*a**2*b**2*n**2 + 672.0*a**2*b**2 + 8.0*b**4*n**4 - 40.0*b**4*n**2 + 32.0*b**4)/(8.0*(n - 1.0)*(n - 2.0)*(n + 2.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return (a*b*(a**2*l**2*n + 8.0*a**2*l**2 + a**2*l*n + 8.0*a**2*l + 3.0*a**2*n**3 + 12.0*a**2*n**2 - 15.0*a**2*n - 72.0*a**2 + 4.0*b**2*n**3 + 16.0*b**2*n**2 + 4.0*b**2*n - 24.0*b**2))/(2.0*n*(n + 2.0)*(n - 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -(a**2*(a**2*l**4 + 2.0*a**2*l**3 - 5.0*a**2*l**2*n - 20.0*a**2*l**2 - 5.0*a**2*l*n - 21.0*a**2*l - a**2*n**4 - 9.0*a**2*n**3 - 17.0*a**2*n**2 + 39.0*a**2*n + 108.0*a**2 + 2.0*b**2*l**2*n**2 + 4.0*b**2*l**2*n - 6.0*b**2*l**2 + 2.0*b**2*l*n**2 + 4.0*b**2*l*n - 6.0*b**2*l - 6.0*b**2*n**4 - 54.0*b**2*n**3 - 138.0*b**2*n**2 - 18.0*b**2*n + 216.0*b**2))/(4.0*n*(n + 3.0)*(n - 1.0)*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -(a**3*b*(n + 4.0)*(l**2 + l - n**2 - 8.0*n - 15.0))/(2.0*n*(n + 2.0)*(n + 1.0))

    # Generate 4th superdiagonal
    def d4(n):
        return (a**4*(l + n + 6.0)*(l - n - 3.0)*(l + n + 4.0)*(l - n - 5.0))/(16.0*n*(n + 3.0)*(n + 2.0)*(n + 1.0))

    ds = [d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i2r1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-3,4)
    nzrow = 1

    # Generate 3rd subdiagonal
    def d_3(n):
        return a**3/(8.0*n*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2*b/(4.0*n*(n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -d_3(n + 1.0)

    # Generate main diagonal
    def d0(n):
        return -a**2*b/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -d_3(n)

    # Generate 2nd superdiagonal
    def d2(n):
        return d_2(n + 1.0)

    # Generate 3rd superdiagonal
    def d3(n):
        return d_3(n + 1.0)

    ds = [d_3, d_2, d_1, d0, d1, d2, d3]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i2r1d1r1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r D_r r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-3,4)
    nzrow = 1

    # Generate 3rd subdiagonal
    def d_3(n):
        return a**3*(n - 2.0)/(8.0*n*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2*b*(2.0*n - 3.0)/(4.0*n*(n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return a*(a**2*n + 2.0*a**2 + 4.0*b**2*n + 4.0*b**2)/(8.0*n*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return a**2*b/(2.0*(n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -a*(a**2*n - 2.0*a**2 + 4.0*b**2*n - 4.0*b**2)/(8.0*n*(n - 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*b*(2.0*n + 3.0)/(4.0*n*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -a**3*(n + 2.0)/(8.0*n*(n + 1.0))

    ds = [d_3, d_2, d_1, d0, d1, d2, d3]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i2r2d1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 2nd integral of r^2 D_r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-3,4)
    nzrow = 1

    # Generate 3rd subdiagonal
    def d_3(n):
        return a**3*(n - 3.0)/(8.0*n*(n - 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return a**2*b*(n - 2.0)/(2.0*n*(n - 1.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return a*(a**2*n + 3.0*a**2 + 4.0*b**2*n + 4.0*b**2)/(8.0*n*(n + 1.0))

    # Generate main diagonal
    def d0(n):
        return a**2*b/((n - 1.0)*(n + 1.0))

    # Generate 1st superdiagonal
    def d1(n):
        return -a*(a**2*n - 3.0*a**2 + 4.0*b**2*n - 4.0*b**2)/(8.0*n*(n - 1.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**2*b*(n + 2.0)/(2.0*n*(n + 1.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -a**3*(n + 3.0)/(8.0*n*(n + 1.0))

    ds = [d_3, d_2, d_1, d0, d1, d2, d3]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r2(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^2 T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-6,7)
    nzrow = 3

    # Generate 6th subdiagonal
    def d_6(n):
        return a**6/(64.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 5th subdiagonal
    def d_5(n):
        return a**5*b/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return -a**4*(a**2*n - 5.0*a**2 - 2.0*b**2*n - 2.0*b**2)/(32.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -3.0*a**5*b/(16.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -a**4*(a**2*n + 17.0*a**2 + 16.0*b**2*n + 32.0*b**2)/(64.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return a**5*b*(n - 8.0)/(8.0*n*(n - 3.0)*(n - 2.0)*(n + 1.0)*(n + 2.0))

    # Generate main diagonal
    def d0(n):
        return a**4*(a**2*n**2 - 19.0*a**2 + 6.0*b**2*n**2 - 54.0*b**2)/(16.0*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 1st superdiagonal
    def d1(n):
        return a**5*b*(n + 8.0)/(8.0*n*(n - 2.0)*(n - 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**4*(a**2*n - 17.0*a**2 + 16.0*b**2*n - 32.0*b**2)/(64.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -3*a**5*b/(16.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 4th superdiagonal
    def d4(n):
        return -a**4*(a**2*n + 5.0*a**2 - 2.0*b**2*n + 2.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 5th superdiagonal
    def d5(n):
        return a**5*b/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 6th superdiagonal
    def d6(n):
        return a**6/(64.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_6, d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5, d6]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r3(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^3 T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-7,8)
    nzrow = 3

    # Generate 7th subdiagonal
    def d_7(n):
        return a**7/(128.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 6th subdiagonal
    def d_6(n):
        return 3.0*a**6*b/(64.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 5th subdiagonal
    def d_5(n):
        return -a**5*(a**2*n - 11.0*a**2 - 12.0*b**2*n - 12.0*b**2)/(128.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return -a**4*b*(3.0*a**2*n - 15.0*a**2 - 2.0*b**2*n - 2.0*b**2)/(32.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -3.0*a**5*(a**2*n + 6.0*a**2 + 12.0*b**2*n + 24.0*b**2)/(128.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -a**4*b*(3.0*a**2*n + 51.0*a**2 + 16.0*b**2*n + 32.0*b**2)/(64.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return 3.0*a**5*(a**2*n**2 - 5.0*a**2*n - 34.0*a**2 + 8.0*b**2*n**2 - 40.0*b**2*n - 192.0*b**2)/(128.0*n*(n - 3.0)*(n - 2.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate main diagonal
    def d0(n):
        return 3.0*a**4*b*(a**2*n**2 - 19.0*a**2 + 2.0*b**2*n**2 - 18.0*b**2)/(16.0*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 1st superdiagonal
    def d1(n):
        return 3.0*a**5*(a**2*n**2 + 5.0*a**2*n - 34.0*a**2 + 8.0*b**2*n**2 + 40.0*b**2*n - 192.0*b**2)/(128.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return -a**4*b*(3.0*a**2*n - 51.0*a**2 + 16.0*b**2*n - 32.0*b**2)/(64.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return -3.0*a**5*(a**2*n - 6.0*a**2 + 12.0*b**2*n - 24.0*b**2)/(128.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 4th superdiagonal
    def d4(n):
        return -a**4*b*(3.0*a**2*n + 15.0*a**2 - 2.0*b**2*n + 2.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 5th superdiagonal
    def d5(n):
        return -a**5*(a**2*n + 11.0*a**2 - 12.0*b**2*n + 12.0*b**2)/(128.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 6th superdiagonal
    def d6(n):
        return 3.0*a**6*b/(64.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 7th superdiagonal
    def d7(n):
        return a**7/(128.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_7, d_6, d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5, d6, d7]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r3d1r1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^3 D r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-7,8)
    nzrow = 3

    # Generate 7th subdiagonal
    def d_7(n):
        return a**7*(n - 6.0)/(128.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 6th subdiagonal
    def d_6(n):
        return a**6*b*(4.0*n - 21.0)/(64.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 5th subdiagonal
    def d_5(n):
        return a**5*(a**2*n**2 + 7.0*a**2*n - 54.0*a**2 + 24.0*b**2*n**2 - 84.0*b**2*n - 108.0*b**2)/(128.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return a**4*b*(21.0*a**2*n - 81.0*a**2 + 8.0*b**2*n**2 - 22.0*b**2*n - 30.0*b**2)/(32.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -a**3*(3.0*a**4*n**2 - 12.0*a**4*n - 84.0*a**4 + 24.0*a**2*b**2*n**2 - 180.0*a**2*b**2*n - 456.0*a**2*b**2 - 16.0*b**4*n**2 - 48.0*b**4*n - 32.0*b**4)/(128.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -a**4*b*(12.0*a**2*n**2 - 9.0*a**2*n - 261.0*a**2 + 32.0*b**2*n**2 - 80.0*b**2*n - 288.0*b**2)/(64.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -3.0*a**3*(a**4*n**3 + 9.0*a**4*n**2 - 24.0*a**4*n - 156.0*a**4 + 16.0*a**2*b**2*n**3 + 88.0*a**2*b**2*n**2 - 264.0*a**2*b**2*n - 1152.0*a**2*b**2 + 16.0*b**4*n**3 + 32.0*b**4*n**2 - 144.0*b**4*n - 288.0*b**4)/(128.0*n*(n - 3.0)*(n - 2.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate main diagonal
    def d0(n):
        return -3.0*a**4*b*(7.0*a**2*n**2 - 93.0*a**2 + 14.0*b**2*n**2 - 126.0*b**2)/(16.0*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 1st superdiagonal
    def d1(n):
        return 3.0*a**3*(a**4*n**3 - 9.0*a**4*n**2 - 24.0*a**4*n + 156.0*a**4 + 16.0*a**2*b**2*n**3 - 88.0*a**2*b**2*n**2 - 264.0*a**2*b**2*n + 1152.0*a**2*b**2 + 16.0*b**4*n**3 - 32.0*b**4*n**2 - 144.0*b**4*n + 288.0*b**4)/(128.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return a**4*b*(12.0*a**2*n**2 + 9.0*a**2*n - 261.0*a**2 + 32.0*b**2*n**2 + 80.0*b**2*n - 288.0*b**2)/(64.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return a**3*(3.0*a**4*n**2 + 12.0*a**4*n - 84.0*a**4 + 24.0*a**2*b**2*n**2 + 180.0*a**2*b**2*n - 456.0*a**2*b**2 - 16.0*b**4*n**2 + 48.0*b**4*n - 32.0*b**4)/(128.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 4th superdiagonal
    def d4(n):
        return a**4*b*(21.0*a**2*n + 81.0*a**2 - 8.0*b**2*n**2 - 22.0*b**2*n + 30.0*b**2)/(32.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 5th superdiagonal
    def d5(n):
        return -a**5*(a**2*n**2 - 7.0*a**2*n - 54.0*a**2 + 24.0*b**2*n**2 + 84.0*b**2*n - 108.0*b**2)/(128.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 6th superdiagonal
    def d6(n):
        return -a**6*b*(4.0*n + 21.0)/(64.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 7th superdiagonal
    def d7(n):
        return -a**7*(n + 6.0)/(128.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_7, d_6, d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5, d6, d7]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def i4r4d1(nr, a, b, bc, coeff = 1.0):
    """Create operator for 4th integral of r^4 D_r T_n(x)."""

    ns = np.arange(0, nr)
    offsets = np.arange(-7,8)
    nzrow = 3

    # Generate 7th subdiagonal
    def d_7(n):
        return a**7*(n - 7.0)/(128.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 6th subdiagonal
    def d_6(n):
        return a**6*b*(n - 6.0)/(16.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0))

    # Generate 5th subdiagonal
    def d_5(n):
        return a**5*(n - 5.0)*(a**2*n + 13.0*a**2 + 24.0*b**2*n + 24.0*b**2)/(128.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 4th subdiagonal
    def d_4(n):
        return a**4*b*(n - 4.0)*(3.0*a**2 + b**2*n + b**2)/(4.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0))

    # Generate 3rd subdiagonal
    def d_3(n):
        return -a**3*(3.0*a**4*n**2 - 15.0*a**4*n - 102.0*a**4 + 24.0*a**2*b**2*n**2 - 216.0*a**2*b**2*n - 528.0*a**2*b**2 - 16.0*b**4*n**2 - 48.0*b**4*n - 32.0*b**4)/(128.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 2nd subdiagonal
    def d_2(n):
        return -a**4*b*(3.0*a**2*n**2 - 3.0*a**2*n - 78.0*a**2 + 8.0*b**2*n**2 - 24.0*b**2*n - 80.0*b**2)/(16.0*n*(n - 3.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 1st subdiagonal
    def d_1(n):
        return -3.0*a**3*(a**4*n**3 + 10.0*a**4*n**2 - 29.0*a**4*n - 190.0*a**4 + 16.0*a**2*b**2*n**3 + 96.0*a**2*b**2*n**2 - 304.0*a**2*b**2*n - 1344.0*a**2*b**2 + 16.0*b**4*n**3 + 32.0*b**4*n**2 - 144.0*b**4*n - 288.0*b**4)/(128.0*n*(n - 3.0)*(n - 2.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate main diagonal
    def d0(n):
        return -3.0*a**4*b*(a**2*n**2 - 14.0*a**2 + 2.0*b**2*n**2 - 18.0*b**2)/(2*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 1st superdiagonal
    def d1(n):
        return 3.0*a**3*(a**4*n**3 - 10.0*a**4*n**2 - 29.0*a**4*n + 190.0*a**4 + 16.0*a**2*b**2*n**3 - 96.0*a**2*b**2*n**2 - 304.0*a**2*b**2*n + 1344.0*a**2*b**2 + 16.0*b**4*n**3 - 32.0*b**4*n**2 - 144.0*b**4*n + 288.0*b**4)/(128.0*n*(n - 3.0)*(n - 2.0)*(n - 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 2nd superdiagonal
    def d2(n):
        return a**4*b*(3.0*a**2*n**2 + 3.0*a**2*n - 78.0*a**2 + 8.0*b**2*n**2 + 24.0*b**2*n - 80.0*b**2)/(16.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 3.0))

    # Generate 3rd superdiagonal
    def d3(n):
        return a**3*(3.0*a**4*n**2 + 15.0*a**4*n - 102.0*a**4 + 24.0*a**2*b**2*n**2 + 216.0*a**2*b**2*n - 528.0*a**2*b**2 - 16.0*b**4*n**2 + 48.0*b**4*n - 32.0*b**4)/(128.0*n*(n - 2.0)*(n - 1.0)*(n + 1.0)*(n + 2.0))

    # Generate 4th superdiagonal
    def d4(n):
        return a**4*b*(n + 4.0)*(3.0*a**2 - b**2*n + b**2)/(4.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 5th superdiagonal
    def d5(n):
        return -a**5*(n + 5.0)*(a**2*n - 13.0*a**2 + 24.0*b**2*n - 24.0*b**2)/(128.0*n*(n - 1.0)*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 6th superdiagonal
    def d6(n):
        return -a**6*b*(n + 6.0)/(16.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    # Generate 7th superdiagonal
    def d7(n):
        return -a**7*(n + 7.0)/(128.0*n*(n + 1.0)*(n + 2.0)*(n + 3.0))

    ds = [d_7, d_6, d_5, d_4, d_3, d_2, d_1, d0, d1, d2, d3, d4, d5, d6, d7]
    diags = utils.build_diagonals(ns, nzrow, ds, offsets)

    mat = coeff*spsp.diags(diags, offsets, format = 'coo')
    return radbc.constrain(mat, bc)

def qid(nr, q, bc, coeff = 1.0):
    """Create a quasi identity block of order q"""

    mat = spsp.coo_matrix((nr,nr))
    if coeff != 1.0:
        mat.data = coeff*np.ones((nr-q))
    else:
        mat.data = np.ones((nr-q))
    mat.row = np.arange(q,nr)
    mat.col = mat.row
    return radbc.constrain(mat, bc)

def linear_r2x(ro, rratio):
    """Calculat a and b for linear map r = a*x + b"""

    b = (ro*rratio + ro)/2.0;
    a = ro - b;

    return (a, b)

def stencil(nr, bc, make_square):
    """Create a galerkin stencil matrix"""

    mat = qid(nr, 0, radbc.no_bc())

    if not make_square:
        bc['rt'] = 0

    return radbc.constrain(mat, bc)

def integral(nr, a, b):
    """Compute the definite integral of the expansion"""

    mat = spsp.lil_matrix((1,nr))
    mat[0,::2] = [4.0*a*(n/(n**2 - 1.0) - 1.0/(n - 1.0)) for n in np.arange(0,nr,2)]
    mat[0,0] = mat[0,0]/2.0

    return mat

def avg(nr, a, b):
    """Compute the average of the expansion"""

    mat = integral(nr,a,b)/2.0

    return mat
