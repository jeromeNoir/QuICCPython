"""Module provides generic functions for the sparse chebyshev representation."""

from __future__ import division
from __future__ import unicode_literals

import numpy as np
import scipy.sparse as spsp


def build_diagonals(ns, nzrow, ds, offsets, cross_parity = None, has_wrap = True):
    """Build diagonals from function list and offsets"""

    # Build wrapped around values
    if has_wrap:
        wrap = [[0]*abs(offsets[0]) for i in range(len(ds))]
        for d,f in enumerate(ds):
            lb = max(0, -offsets[d])
            for i in range(0, lb):
                if ns[i] > nzrow:
                    lbw = sum(offsets < 0)
                    step = (ns[1]-ns[0])
                    if cross_parity == None:
                        shift = 0
                    else:
                        shift = (-1)**cross_parity 
                    col = sum(((ns + shift) - (-(ns[i] + shift) - step*offsets[d])) < 0)
                    col = lbw + col - i
                    row = i - max(0, lbw - col)
                    wrap[col][row] = f(ns[i])

    # Compute the end index of zero rows
    izrow = 0
    for i,n in enumerate(ns):
        if n <= nzrow:
            izrow = i+1
        else:
            break

    # Build diagonals
    diags = [0]*len(ds)
    for d,f in enumerate(ds):
        lb = max(0, -offsets[d])
        ub = min(len(ns),len(ns)-offsets[d])
        if izrow > lb:
            diags[d] = np.concatenate((np.zeros(izrow-lb), f(ns[izrow:ub])))
        else:
            diags[d] = f(ns[lb:ub])
        if has_wrap:
            if len(wrap[d]) > 0:
                diags[d][0:len(wrap[d])] = [x + y for x, y in zip(diags[d][0:len(wrap[d])], wrap[d][:])]

    return diags

def build_block_matrix(fields, func, func_args, restriction = None):

    if restriction is None:
        restrict = [None]*len(fields)
    else:
        try:
            n = len(restriction[0])
            if len(restriction) == len(fields):
                restrict = restriction
            else:
                raise RuntimeError('Restriction size does not match number of fields')
        except TypeError:
            restrict = [restriction]*len(fields)

    tmp = []
    for field_row in fields:
        row = []
        for j, field_col in enumerate(fields):
            args = func_args + (field_row,field_col)
            row.append(func(*args, restriction = restrict[j]))
        tmp.append(row)
    return spsp.bmat(tmp, format='coo')

def build_diag_matrix(fields, func, func_args, restriction = None):

    if restriction is None:
        restrict = [None]*len(fields)
    else:
        try:
            n = len(restriction[0])
            if len(restriction) == len(fields):
                restrict = restriction
            else:
                raise RuntimeError('Restriction size does not match number of fields')
        except TypeError:
            restrict = [restriction]*len(fields)

    tmp = []
    for j, field_row in enumerate(fields):
        args = func_args + (field_row,)
        tmp.append(func(*args, restriction = restrict[j]))
   
    return spsp.block_diag(tmp, format='coo')

def build_block_vector(fields, func, func_args, restriction = None):

    if restriction is None:
        restrict = [None] * len(fields)
    else:
        try:
            n = len(restriction[0])
            if len(restriction) == len(fields):
                restrict = restriction
            else:
                raise RuntimeError('Restriction size does not match number of fields')
        except TypeError:
            restrict = [restriction]*len(fields)

    tmp = []
    for j, field_row in enumerate(fields):
        args = func_args + (field_row,field_row)
        tmp.append(func(*args, restriction = restrict[j]))

    return spsp.vstack(tmp, format='coo')

def triplets(mat):
    if not spsp.isspmatrix_coo(mat):
        mat = mat.tocoo();

    return list(zip(mat.row,mat.col,mat.data))

def rows_kron_2d(A, B, restriction):
    """Compute a row restriction of a 2D kronecker product"""

    diag = spsp.lil_matrix((1,A.shape[0]))
    diag[0,restriction] = 1.0
    S = spsp.diags(diag.todense(), [0], shape = A.shape)

    mat = spsp.kron(S*A, B)

    return mat

def rows_kron_3d(A, B, C, restriction):
    """Compute a row restriction of a 3D kronecker product"""
    
    out_rows = B.shape[0]*C.shape[0]
    out_cols = A.shape[1]*B.shape[1]*C.shape[1]

    A = spsp.csr_matrix(A)
    itSlow = iter(restriction[0])
    itFast = iter(restriction[1])
    row = next(itSlow)
    lines = next(itFast)
    if row == 0:
        mat = spsp.kron(A[0,:], rows_kron_2d(B, C, lines))
        row = next(itSlow)
        lines = next(itFast)
    else:
        mat = spsp.coo_matrix((out_rows, out_cols))

    try:
        for i in range(1, A.shape[0]):
            if i == row:
                mat = spsp.vstack([mat, spsp.kron(A[i,:], rows_kron_2d(B, C, lines))])
                row = next(itSlow)
                lines = next(itFast)
            else:
                mat = spsp.vstack([mat, spsp.coo_matrix((out_rows, out_cols))])
    except:
        pass

    if mat.shape[0] < A.shape[0]*B.shape[0]*C.shape[0]:
        zrows = (A.shape[0]*B.shape[0]*C.shape[0] - mat.shape[0])
        mat = spsp.vstack([mat, spsp.coo_matrix((zrows, out_cols))])

    return mat

def cols_kron_3d(A, B, C, restriction):
    """Compute a column restriction of a 3D kronecker product"""
    
    out_rows = A.shape[0]*B.shape[0]*C.shape[0]
    out_cols = B.shape[1]*C.shape[1]

    A = spsp.csc_matrix(A)
    itSlow = iter(restriction[0])
    itFast = iter(restriction[1])
    col = next(itSlow)
    lines = next(itFast)
    if col == 0:
        mat = spsp.kron(A[:,0], cols_kron_2d(B, C, lines))
        col = next(itSlow)
        lines = next(itFast)
    else:
        mat = spsp.coo_matrix((out_rows, out_cols))

    try:
        for i in range(1, A.shape[1]):
            if i == col:
                mat = spsp.hstack([mat, spsp.kron(A[:,i], cols_kron_2d(B, C, lines))])
                col = next(itSlow)
                lines = next(itFast)
            else:
                mat = spsp.hstack([mat, spsp.coo_matrix((out_rows, out_cols))])
    except:
        pass

    if mat.shape[1] < A.shape[1]*B.shape[1]*C.shape[1]:
        zcols = (A.shape[1]*B.shape[1]*C.shape[1] - mat.shape[1])
        mat = spsp.hstack([mat, spsp.coo_matrix((out_rows, zcols))])

    return mat

def cols_kron_2d(A, B, restriction):
    """Compute a column restriction of a 2D kronecker product"""

    diag = np.zeros((1,A.shape[0]))
    diag[0,restriction] = 1.0
    S = spsp.diags(diag, [0], shape = A.shape)

    mat = spsp.kron(A*S, B)

    return mat

def restricted_kron_2d(A, B, restriction = None):
    """Compute a double Kronecker product with possible restrictions"""

    if restriction == None or A.nnz == 0 or B.nnz == 0:
        mat = spsp.kron(A, B)

    else:
        mat = cols_kron_2d(A, B, restriction)

    return mat

def restricted_kron_3d(A, B, C, restriction = None):
    """Compute a triple Kronecker product with possible restrictions"""

    if restriction == None or A.nnz == 0 or B.nnz == 0 or C.nnz == 0:
        mat = spsp.kron(A, spsp.kron(B, C))

    else:
        mat = cols_kron_3d(A, B, C, restriction)

    return mat

def id_from_idx_1d(idx, n):
    """Create a sparse identity from indexes for 1D matrix"""

    d = np.zeros((1,n))
    if idx.size > 0:
        d[0,idx] = 1.0

    return spsp.diags(d, [0], 2*(n,))

def id_from_idx_2d(idx, nA, nB, restriction = None):
    """Create a sparse identity from indexes for 2D matrix"""

    if restriction != None:
        tmp = idx.tolist()
        ridx = []
        for i,s in enumerate(restriction):
                r = s*nB
                for k in range(0, nB):
                    if tmp.count(r + k) > 0:
                        ridx.append(r + k)
        idx = np.array(ridx)

    n = nA*nB
    d = np.zeros((1,n))
    if idx.size > 0:
        d[0,idx] = 1.0

    return spsp.diags(d, [0], 2*(n,))

def id_from_idx_3d(idx, nA, nB, nC, restriction = None):
    """Create a sparse identity from indexes for 3D matrix"""

    if restriction != None:
        tmp = idx.tolist()
        ridx = []
        for i,s in enumerate(restriction[0]):
            for m in restriction[1][i]:
                r = s*nB*nC + m*nC
                for k in range(0, nC):
                    if tmp.count(r + k) > 0:
                        ridx.append(r + k)
        idx = np.array(ridx)

    n = nA*nB*nC
    d = np.zeros((1,n))
    if idx.size > 0:
        d[0,idx] = 1.0

    return spsp.diags(d, [0], 2*(n,))

def qid_from_idx(idx, n):
    """Create a sparse identity from zero indexes indexes"""

    d = np.ones((1,n))
    if idx.size > 0:
        d[0,idx] = 0.0

    return spsp.diags(d, [0], 2*(n,))

def idx_kron_2d(nA, nB, idxA, idxB):
    """Compute nonzero diagonal indexex for a 2D sparse kronecker identity"""
    
    idx = idxA.repeat(len(idxB))
    idx *= nB
    idx = idx.reshape(-1,len(idxB))
    idx += idxB
    idx = idx.reshape(-1)

    return idx

def idx_kron_3d(nA, nB, nC, idxA, idxB, idxC):
    """Compute nonzero diagonal indexex for a 3D sparse kronecker identity"""

    idxD = idx_kron_2d(nB, nC, idxB, idxC)
    idx = idx_kron_2d(nA, nB*nC, idxA, idxD)

    return idx

def qidx(nx, q):
    """Create index of nonzero for sparse array with q zeros at the top"""

    return np.arange(q, nx)

def sidx(nx, s):
    """Create index of nonzero for sparse array with s zeros at the bottom"""

    return np.arange(0, nx-s)