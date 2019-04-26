"""Module provides functions to generate the radial boundary conditions in a spherical shell"""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp
import itertools

import QuICCPython.shell.quicc_tools.utils as utils


use_parity_bc = False

def no_bc():
    """Get a no boundary condition flag"""

    return {0:0}

def constrain(mat, bc, location = 't'):
    """Contrain the matrix with the (Tau or Galerkin) boundary condition"""

    if bc[0] > 0:
        bc_mat = apply_tau(mat, bc, location = location)
    elif bc[0] < 0:
        bc_mat = apply_galerkin(mat, bc)
    else:
        bc_mat = mat

    # top row(s) restriction if required
    if bc.get('rt', 0) > 0:
        bc_mat = restrict_eye(bc_mat.shape[0], 'rt', bc['rt'])*bc_mat

    # bottom row(s) restriction if required
    if bc.get('rb', 0) > 0:
        bc_mat = restrict_eye(bc_mat.shape[0], 'rb', bc['rb'])*bc_mat

    # left columns restriction if required
    if bc.get('cl', 0) > 0:
        bc_mat = bc_mat*restrict_eye(bc_mat.shape[1], 'cl', bc['cl'])

    # right columns restriction if required
    if bc.get('cr', 0) > 0:
        bc_mat = bc_mat*restrict_eye(bc_mat.shape[1], 'cr', bc['cr'])

    # top row(s) zeroing if required
    if bc.get('zt', 0) > 0:
        bc_mat = bc_mat.tolil()
        bc_mat[0:bc['zt'],:] = 0
        bc_mat = bc_mat.tocoo()

    # bottom row(s) zeroing if required
    if bc.get('zb', 0) > 0:
        bc_mat = bc_mat.tolil()
        bc_mat[-bc['zb']:,:] = 0
        bc_mat = bc_mat.tocoo()

    # left columns zeroing if required
    if bc.get('zl', 0) > 0:
        bc_mat = bc_mat.tolil()
        bc_mat[:, 0:bc['zt']] = 0
        bc_mat = bc_mat.tocoo()

    # right columns zeroing if required
    if bc.get('zr', 0) > 0:
        bc_mat = bc_mat.tolil()
        bc_mat[:, -bc['zr']:] = 0
        bc_mat = bc_mat.tocoo()

    return bc_mat

def apply_tau(mat, bc, location = 't'):
    """Add Tau lines to the matrix"""

    nbc = bc[0]//10

    if bc[0] == 10:
        cond = tau_value(mat.shape[1], 1, bc.get('c',None))
    elif bc[0] == 11:
        cond = tau_value(mat.shape[1], -1, bc.get('c',None))
    elif bc[0] == 12:
        cond = tau_diff(mat.shape[1], 1, bc.get('c',None))
    elif bc[0] == 13:
        cond = tau_diff(mat.shape[1], -1, bc.get('c',None))
    elif bc[0] == 20:
        cond = tau_value(mat.shape[1], 0, bc.get('c',None))
    elif bc[0] == 21:
        cond = tau_diff(mat.shape[1], 0, bc.get('c',None))
    elif bc[0] == 22:
        cond = tau_rdiffdivr(mat.shape[1], 0, bc.get('c',None))
    elif bc[0] == 23:
        cond = tau_insulating(mat.shape[1], 0, bc.get('c',None))
    elif bc[0] == 24:
        cond = tau_couette(mat.shape[1], 0, bc.get('c',None))
    elif bc[0] == 25:
        cond1 = tau_diff(mat.shape[1], -1, bc.get('c',None))
        cond2 = tau_value(mat.shape[1], 1, bc.get('cc', None))
        cond = np.vstack([cond1, cond2])
    elif bc[0] == 26:
        cond1 = tau_value(mat.shape[1], -1, bc.get('cc',None))
        cond2 = tau_diff(mat.shape[1], 1, bc.get('c',None))
        cond = np.vstack([cond1, cond2])
    elif bc[0] == 40:
        cond = tau_value_diff(mat.shape[1], 0, bc.get('c',None))
    elif bc[0] == 41:
        cond = tau_value_diff2(mat.shape[1], 0, bc.get('c',None))
    elif bc[0] == 42:
        cond1 = tau_value_diff2(mat.shape[1], -1, bc.get('c', None))
        cond2 = tau_value_diff(mat.shape[1], 1, bc.get('c', None))
        cond = np.vstack([cond1, cond2])
    elif bc[0] == 43:
        cond1 = tau_value_diff2(mat.shape[1], 1, bc.get('c', None))
        cond2 = tau_value_diff(mat.shape[1], -1, bc.get('c', None))
        cond = np.vstack([cond1, cond2])
    # Set last modes to zero
    elif bc[0] > 990 and bc[0] < 1000:
        cond = tau_last(mat.shape[1], bc[0]-990)
        nbc = bc[0]-990

    if not spsp.isspmatrix_coo(mat):
        mat = mat.tocoo()
    if location == 't':
        s = 0
    elif location == 'b':
        s = mat.shape[0]-nbc

    conc = np.concatenate
    for i,c in enumerate(cond):
        mat.data = conc((mat.data, c))
        mat.row = conc((mat.row, [s+i]*mat.shape[1]))
        mat.col = conc((mat.col, np.arange(0,mat.shape[1])))

    return mat

def tau_value(nr, pos, coeffs = None):
    """Create the boundary value tau line(s)"""

    it = coeff_iterator(coeffs, pos)

    cond = []
    c = next(it)
    if pos >= 0:
        cnst = c*tau_c()
        cond.append(cnst*np.ones(nr))
        cond[-1][0] /= tau_c()
        c = next(it)

    if pos <= 0:
        cnst = c*tau_c()
        cond.append(cnst*alt_ones(nr, 1))
        cond[-1][0] /= tau_c()

    if use_parity_bc and pos == 0:
        t = cond[0].copy()
        cond[0] = (cond[0] + cond[1])/2.0
        cond[1] = (t - cond[1])/2.0

    return np.array(cond)

def tau_diff(nr, pos, coeffs = None):
    """Create the first derivative tau line(s)"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)

    if coeffs is None:
        raise RuntimeError
    else:
        it = coeff_iterator(coeffs.get('c', None), pos)

    a = coeffs['a']
    b = coeffs['b']

    cond = []
    c = next(it)
    ns = np.arange(0,nr)
    if pos >= 0:
        cond.append(c*(2.0/a)*ns**2)
        c = next(it)

    if pos <= 0:
        cond.append(c*(2.0/a)*ns**2*alt_ones(nr, 0))

    if use_parity_bc and pos == 0:
        t = cond[0].copy()
        cond[0] = (cond[0] + cond[1])/2.0
        cond[1] = (t - cond[1])/2.0

    return np.array(cond)

def tau_diff2(nr, pos, coeffs = None):
    """Create the second deriviative tau line(s)"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)

    if coeffs is None:
        raise RuntimeError
    else:
        it = coeff_iterator(coeffs.get('c', None), pos)

    a = coeffs['a']
    b = coeffs['b']

    cond = []
    c = next(it)
    ns = np.arange(0,nr)
    if pos >= 0:
        cond.append(c*(2.0/(3.0*a**2))*(ns**4 - ns**2))
        c = next(it)

    if pos <= 0:
        cond.append(c*(2.0/(3.0*a**2))*(ns**4 - ns**2)*alt_ones(nr, 1))

    if use_parity_bc and pos == 0:
        t = cond[0].copy()
        cond[0] = (cond[0] + cond[1])/2.0
        cond[1] = (t - cond[1])/2.0

    return np.array(cond)

def tau_rdiffdivr(nr, pos, coeffs = None):
    """Create the r D 1/r tau line(s)"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)

    if coeffs is None:
        raise RuntimeError
    else:
        it = coeff_iterator(coeffs.get('c', None), pos)

    a = coeffs['a']
    b = coeffs['b']

    cond = []
    c = next(it) 
    ns = np.arange(0,nr)
    if pos >= 0:
        cond.append(c*((1.0/a)*ns**2 - (1.0/(a+b)))*tau_c())
        cond[-1][0] /= tau_c()
        c = next(it)

    if pos <= 0:
        cond.append(c*((1.0/a)*ns**2 + (1.0/(-a+b)))*tau_c()*alt_ones(nr, 0))
        cond[-1][0] /= tau_c()

    if use_parity_bc and pos == 0:
        t = cond[0].copy()
        cond[0] = (cond[0] + cond[1])/2.0
        cond[1] = (t - cond[1])/2.0

    return np.array(cond)

def tau_insulating(nr, pos, coeffs = None):
    """Create the insulating boundray tau line(s)"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)
    assert(coeffs.get('l', None) is not None)

    if coeffs is None:
        raise RuntimeError
    else:
        it = coeff_iterator(coeffs.get('c', None), pos)

    a = coeffs['a']
    b = coeffs['b']
    l = coeffs['l']

    cond = []
    c = next(it)
    ns = np.arange(0,nr)
    if pos >= 0:
        cond.append(c*((2.0/a)*ns**2 + ((l+1.0)/(a+b))*tau_c()))
        cond[-1][0] /= tau_c()
        c = next(it)

    if pos <= 0:
        cond.append(c*((2.0/a)*ns**2 + (l/(-a+b))*tau_c())*alt_ones(nr, 0))
        cond[-1][0] /= tau_c()

    if use_parity_bc and pos == 0:
        t = cond[0].copy()
        cond[0] = (cond[0] + cond[1])/2.0
        cond[1] = (t - cond[1])/2.0

    return np.array(cond)

def tau_couette(nr, pos, coeffs = None):
    """Create the toroidal Couette boundray tau line(s)"""

    assert(coeffs.get('c', None) is not None)
    #TODO: think of the ordering
    #assert(coeffs.get('l', None) is not None)
    assert(pos == 0)

    return tau_value(nr, 0, None)

def tau_value_diff(nr, pos, coeffs = None):
    """Create the no penetration and no-slip tau line(s)"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)

    cond = []
    if pos >= 0:
        cond.append(tau_value(nr,1,coeffs.get('c',None))[0])
        cond.append(tau_diff(nr,1,coeffs)[0])

    if pos <= 0:
        cond.append(tau_value(nr,-1,coeffs.get('c',None))[0])
        cond.append(tau_diff(nr,-1,coeffs)[0])

    if use_parity_bc and pos == 0:
        tv = cond[0].copy()
        td = cond[1].copy()
        cond[0] = (cond[0] + cond[2])/2.0
        cond[1] = (cond[1] + cond[3])/2.0
        cond[2] = (tv - cond[2])/2.0
        cond[3] = (td - cond[3])/2.0

    return np.array(cond)

def tau_value_diff2(nr, pos, coeffs = None):
    """Create the no penetration and stress-free tau line(s)"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)

    cond = []
    if pos >= 0:
        cond.append(tau_value(nr,1,coeffs.get('c',None))[0])
        cond.append(tau_diff2(nr,1,coeffs)[0])

    if pos <= 0:
        cond.append(tau_value(nr,-1,coeffs.get('c',None))[0])
        cond.append(tau_diff2(nr,-1,coeffs)[0])

    if use_parity_bc and pos == 0:
        tv = cond[0].copy()
        td = cond[1].copy()
        cond[0] = (cond[0] + cond[2])/2.0
        cond[1] = (cond[1] + cond[3])/2.0
        cond[2] = (tv - cond[2])/2.0
        cond[3] = (td - cond[3])/2.0

    return np.array(cond)

def tau_last(nr, nrow):
    """Create the last modes to zero tau line(s)"""

    cond = np.zeros((nrow, nr))
    for j in range(0, nrow):
        cond[j,nr-nrow+j] = tau_c()

    return cond

def stencil(nr, bc):
    """Create a Galerkin stencil matrix"""

    if bc[0] == -10:
        mat = stencil_value(nr, 1, bc.get('c',None))
    elif bc[0] == -11:
        mat = stencil_value(nr, -1, bc.get('c',None))
    elif bc[0] == -12:
        mat = stencil_value(nr, 1, bc.get('c',None))
    elif bc[0] == -13:
        mat = stencil_value(nr, -1, bc.get('c',None))
    elif bc[0] == -20:
        mat = stencil_value(nr, 0, bc.get('c',None))
    elif bc[0] == -21:
        mat = stencil_diff(nr, 0, bc.get('c',None))
    elif bc[0] == -22:
        mat = stencil_rdiffdivr(nr, 0, bc.get('c',None))
    elif bc[0] == -23:
        mat = stencil_insulating(nr, 0, bc.get('c',None))
    elif bc[0] == -40:
        mat = stencil_value_diff(nr, 0, bc.get('c',None))
    elif bc[0] == -41:
        mat = stencil_value_diff2(nr, 0, bc.get('c',None))
    elif bc[0] < -1 and bc[0] > -5:
        mat = restrict_eye(nr, 'cr', -bc[0])

    return mat

def apply_galerkin(mat, bc):
    """Apply a Galerkin stencil on the matrix"""
    
    nr = mat.shape[0]
    mat = mat*stencil(nr, bc)
    return mat

def restrict_eye(nr, t, q):
    """Create the non-square identity to restrict matrix"""

    if t == 'rt':
        offsets = [q]
        diags = [[1]*(nr-q)]
        nrows = nr - q
        ncols = nr
    elif t == 'rb':
        offsets = [0]
        diags = [[1]*(nr-q)]
        nrows = nr - q
        ncols = nr
    elif t == 'cl':
        offsets = [-q]
        diags = [[1]*(nr-q)]
        nrows = nr
        ncols = nr - q
    elif t == 'cr':
        offsets = [0]
        diags = [[1]*(nr-q)]
        nrows = nr
        ncols = nr - q

    return spsp.diags(diags, offsets, (nrows, ncols))

def stencil_value(nr, pos, coeffs = None):
    """Create stencil matrix for a zero boundary value"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)
    assert(coeffs.get('c', None) is None)

    ns = np.arange(0,nr,1)
    if pos == 0:
        offsets = [-2, 0]
        sgn = -1.0
    else:
        offsets = [-1, 0]
        sgn = -pos 

    # Generate subdiagonal
    def d_1(n):
        return galerkin_c(n+offsets[0])*sgn

    # Generate diagonal
    def d0(n):
        return np.ones(n.shape)

    ds = [d_1, d0]
    diags = utils.build_diagonals(ns, -1, ds, offsets, None, False)
    diags[-1] = diags[-1][0:nr+offsets[0]]

    return spsp.diags(diags, offsets, (nr,nr+offsets[0]))

def stencil_diff(nr, pos, coeffs = None):
    """Create stencil matrix for a zero 1st derivative"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)
    assert(coeffs.get('c', None) is None)

    ns = np.arange(0,nr,1)
    if pos == 0:
        offsets = [-2, 0]
        sgn = -1.0
    else:
        offsets = [-1, 0]
        sgn = -pos 

    # Generate subdiagonal
    def d_1(n):
        return sgn*(n+offsets[0])**2/n**2

    # Generate diagonal
    def d0(n):
        return np.ones(n.shape)

    ds = [d_1, d0]
    diags = utils.build_diagonals(ns, -1, ds, offsets, None, False)
    diags[-1] = diags[-1][0:nr+offsets[0]]

    return spsp.diags(diags, offsets, (nr,nr+offsets[0]))

def stencil_rdiffdivr(nr, pos, coeffs = None):
    """Create stencil matrix for a zero r D 1/r derivative"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)
    assert(coeffs.get('c', None) is None)
    assert(pos == 0)

    a = coeffs['a']
    b = coeffs['b']

    ns = np.arange(0,nr,1)
    offsets = [-2, -1, 0]

    # Generate 2nd subdiagonal
    def d_2(n):
        #val_num = a**2*((n - 3.0)**2*n**2 - 2.0) - b**2*(n**2 - 3.0*n + 2.0)**2
        #val_den = -a**2*((n - 2.0)*n - 1.0)*(n**2 - 2.0) + b**2*(n - 1.0)**2*n**2
        val_num = (n - 2.0)*(b**2*(n - 2.0)*(n - 1.0) - a**2*(n - 3.0)*n)
        val_den = n*(a**2*(n - 2.0)*(n + 1.0) - b**2*(n - 1.0)*n)
        val = val_num/val_den
        for i,j in enumerate(n):
            if j == 2:
                #val[i] = a**2/(2.0*a**2 + 4.0*b**2)
                val[i] = 0
            if j > 2:
                break

        return val

    # Generate 1st subdiagonal
    def d_1(n):
        #val_num = -8.0*a*b*n
        #val_den = a**2*(n**2 - 2.0)*(n*(n + 2.0) - 1.0) - b**2*n**2*(n + 1.0)**2
        val_num = -4.0*a*b
        val_den = (n + 1.0)*(a**2*(n**2 + n - 2.0) - b**2*n*(n + 1.0))
        val = val_num/val_den
        for i,j in enumerate(n):
            if j == 1:
                #val[i] = 2.0*a*b/(a**2 + 2.0*b**2)
                val[i] = a/(2.0*b)
            if j > 1:
                break

        return val

    # Generate diagonal
    def d0(n):
        return np.ones(n.shape)

    ds = [d_2, d_1, d0]
    diags = utils.build_diagonals(ns, -1, ds, offsets, None, False)
    diags[-1] = diags[-1][0:nr+offsets[0]]

    return spsp.diags(diags, offsets, (nr,nr+offsets[0]))

def stencil_insulating(nr, pos, coeffs = None):
    """Create stencil matrix for an insulating boundary"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)
    assert(coeffs.get('c', None) is None)
    assert(coeffs.get('l', None) is not None)
    assert(pos == 0)

    a = coeffs['a']
    b = coeffs['b']
    l = coeffs['l']

    ns = np.arange(0,nr,1)
    offsets = [-2, -1, 0]

    # Generate 2nd subdiagonal
    def d_2(n):
        val_num = a**2*(2.0*l*(l+1.0)-2.0*(n-3.0)*n*((n-3.0)*n+5.0)-13.0)+a*b*(2.0*l+1.0)*(2.0*(n-3.0)*n+5.0)+2.0*b**2*(n**2-3.0*n+2.0)**2 
        val_den = a**2*(2.0*l*(l+1.0)-2.0*(n-1.0)*n*((n-1.0)*n+1.0)-1.0)+a*b*(2.0*l+1.0)*(2.0*(n-1.0)*n+1.0)+2.0*b**2*(n-1.0)**2.0*n**2
        val = -val_num/val_den
        for i,j in enumerate(n):
            if j == 2:
                corr_num = a*(a*(2.0*l**2+2.0*l-1.0)+2.0*b*l+b)
                corr_den = 2.0*(a**2*(2.0*l**2+2.0*l-13.0)+5.0*a*(2.0*b*l+b)+8.0*b**2)
                val[i] = -corr_num/corr_den
            if j > 2:
                break

        return val

    # Generate 1st subdiagonal
    def d_1(n):
        val_num = 4.0*a*n*((2.0*l+1.0)*a - b)
        val_den = a**2*(2.0*l*(l+1.0)-2.0*n*(n+1.0)*(n**2+n+1.0)-1.0)+a*b*(2.0*l+1.0)*(2.0*n*(n+1.0)+1.0)+2.0*b**2*n**2*(n+1.0)**2
        val = val_num/val_den
        for i,j in enumerate(n):
            if j == 1:
                corr_num = 2.0*a*(2.0*a*l+a-b)
                corr_den = a**2*(2.0*l**2+2.0*l-13.0)+5.0*a*(2.0*b*l+b)+8.0*b**2
                val[i] = corr_num/corr_den
            if j > 1:
                break

        return val

    # Generate diagonal
    def d0(n):
        return np.ones(n.shape)

    ds = [d_2, d_1, d0]
    diags = utils.build_diagonals(ns, -1, ds, offsets, None, False)
    diags[-1] = diags[-1][0:nr+offsets[0]]

    return spsp.diags(diags, offsets, (nr,nr+offsets[0]))

def stencil_diff2(nr, pos, coeffs = None):
    """Create stencil matrix for a zero 2nd derivative"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)
    assert(coeffs.get('c', None) is None)

    ns = np.arange(0,nr,1)
    if pos == 0:
        offsets = [-2, 0]

        # Generate subdiagonal
        def d_1(n):
            return -(n - 3.0)*(n - 2.0)**2/(n**2*(n + 1.0))

    else:
        offsets = [-1, 0]

        # Generate subdiagonal
        def d_1(n):
            return -pos*(n - 2.0)*(n - 1.0)/(n*(n + 1.0))

    # Generate diagonal
    def d0(n):
        return np.ones(n.shape)

    ds = [d_1, d0]
    diags = utils.build_diagonals(ns, -1, ds, offsets, None, False)
    diags[-1] = diags[-1][0:nr+offsets[0]]

    return spsp.diags(diags, offsets, (nr,nr+offsets[0]))

def stencil_value_diff(nr, pos, coeffs = None):
    """Create stencil matrix for a zero boundary value and a zero 1st derivative"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)
    assert(coeffs.get('c', None) is None)
    assert(pos == 0)

    ns = np.arange(0,nr,1)
    offsets = [-4, -2, 0]

    # Generate 2nd subdiagonal
    def d_2(n):
        val = (n - 3.0)/(n - 1.0)
        for i,j in enumerate(n):
            if j == 4:
                val[i] = 1.0/6.0 
            if j > 4:
                break

        return val

    # Generate 1st subdiagonal
    def d_1(n):
        val = -2.0*n/(n + 1.0)
        for i,j in enumerate(n):
            if j == 2:
                val[i] = -2.0/3.0
            if j > 2:
                break

        return val

    # Generate diagonal
    def d0(n):
        return np.ones(n.shape)

    ds = [d_2, d_1, d0]
    diags = utils.build_diagonals(ns, -1, ds, offsets, None, False)
    diags[-1] = diags[-1][0:nr+offsets[0]]

    return spsp.diags(diags, offsets, (nr,nr+offsets[0]))

def stencil_value_diff2(nr, pos, coeffs = None):
    """Create stencil matrix for a zero boundary value and a zero 2nd derivative"""

    assert(coeffs.get('a', None) is not None)
    assert(coeffs.get('b', None) is not None)
    assert(coeffs.get('c', None) is None)
    assert(pos == 0)

    ns = np.arange(0,nr,1)
    offsets = [-4, -2, 0]

    # Generate 2nd subdiagonal
    def d_2(n):
        val_num = (n - 3.0)*(2.0*n**2 - 12.0*n + 19.0)
        val_den = (n - 1.0)*(2.0*n**2 - 4.0*n + 3.0)
        val = val_num/val_den
        for i,j in enumerate(n):
            if j == 4:
                val[i] = 1.0/38.0
            if j > 4:
                break

        return val

    # Generate 1st subdiagonal
    def d_1(n):
        val_num = -2.0*n*(2.0*n**2 + 7.0)
        val_den = (n + 1.0)*(2.0*n**2 + 4.0*n + 3.0)
        val = val_num/val_den
        for i,j in enumerate(n):
            if j == 2:
                val[i] = -10.0/19.0
            if j > 2:
                break

        return val

    # Generate diagonal
    def d0(n):
        return np.ones(n.shape)

    ds = [d_2, d_1, d0]
    diags = utils.build_diagonals(ns, -1, ds, offsets, None, False)
    diags[-1] = diags[-1][0:nr+offsets[0]]

    return spsp.diags(diags, offsets, (nr,nr+offsets[0]))

def tau_c():
    """Compute the chebyshev normalisation c factor"""

    return 2.0

def galerkin_c(n):
    """Compute the chebyshev normalisation c factor for galerkin boundary"""

    val = np.ones(n.shape)

    for i, j in enumerate(n):
        if j == 0:
            val[i] = 0.5
        if j > 0:
            break

    return val

def coeff_iterator(coeffs, pos):
    """Return an iterator over the constants"""

    if coeffs is None:
        it = itertools.cycle([1.0])
    else:
        try:
            if len(coeffs) == (1 + (pos == 0)):
                it = iter(coeffs)
            elif len(coeffs) == 1:
                it = itertools.cycle(coeffs)
            else:
                raise RuntimeError
        except:
            it = itertools.cycle([coeffs])

    return it

def alt_ones(nr, parity):
    """Get array of alternating 1 and -1. Parity is the parity of the -1"""

    if parity == 0:
        return np.cumprod(-np.ones(nr))
    else:
        return -np.cumprod(-np.ones(nr))

def apply_inhomogeneous(mat, modes, bc, ordering = 'SLFl', location = 't', nr = None):
    """Add inhomogeneous conditions to the matrix"""


    mat = mat.tolil()
    if location == 't':
        s = 0
    elif location == 'b':
        s = mat.shape[0]-nbc

    if bc[0] == 24:
        mat = inh_couette(mat, s, modes, bc.get('c',None), ordering, nr)

    if not spsp.isspmatrix_coo(mat):
        mat = mat.tocoo()

    return mat

def inh_couette(mat, s, modes, coeffs, ordering = 'SLFl', nr= None):
    """Set inhomogeneous constrain for toroidal Couette"""

    assert(coeffs.get('c', None) is not None)

    if ordering=='SLFm':
        assert(coeffs.get('m',None) is not None)

        if coeffs.get('axis',None) is None:
            if coeffs['m'] == 0:
                for i, l in enumerate(modes):
                    if l==1:
                        norm = np.sqrt(3.0/(4.0*np.pi))

                        mat[nr*i+s+1,0] += coeffs['c']/norm

        elif coeffs.get('axis', None) == 'x':
            if coeffs['m'] == 1:
                for i, l in enumerate(modes):
                    if l == 1:
                        norm = -np.sqrt(3 / (8.0 * np.pi))
                        factor = 2.
                        mat[nr*i + s + 1, 0] += coeffs['c'] / norm / factor

    else: # i.e. SLFl
        assert(coeffs.get('l', None) is not None)
        #assert(coeffs.get('axis',None) is not None)

        if coeffs['l'] == 1:
            for i, m in enumerate(modes):
                if coeffs.get('axis', None) is None:
                    if m == 0:
                        norm = np.sqrt(3.0/(4.0*np.pi))
                        mat[s+1,i] += coeffs['c']/norm
                elif coeffs.get('axis', None) == 'x':
                    if m==1:
                        norm = -np.sqrt(3/(8.0*np.pi))
                        factor = 2.
                        mat[s+1,i] += coeffs['c']/norm/factor

    return mat

