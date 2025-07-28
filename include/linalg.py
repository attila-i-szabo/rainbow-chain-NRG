"""Clone of TenPy's `eigh` using the `evd` family of LAPACK drivers.
Faster than the TenPy version, but uses more memory."""

# Copyright (C) TeNPy Developers and Attila Szabó, Apache license

import numpy as np
import scipy as sp

from tenpy.linalg.np_conserved import Array, diag, argsort
from tenpy.linalg import charges


def _eig_worker(hermitian, a, sort, UPLO="L"):
    """Worker for ``eig``, ``eigh``"""
    if a.rank != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("expect a square matrix!")
    a.legs[0].test_contractible(a.legs[1])
    if np.any(a.qtotal != a.chinfo.make_valid()):
        raise ValueError("Non-trivial qtotal -> Nilpotent. Not diagonizable!?")

    piped_axes, a = a.as_completely_blocked()  # ensure complete blocking

    dtype = np.float64 if hermitian else np.complex128
    resw = np.zeros(a.shape[0], dtype=dtype)
    resv = diag(1.0, a.legs[0], dtype=np.promote_types(dtype, a.dtype))
    if isinstance(a.legs[0], charges.LegPipe):
        resv.legs[1] = resv.legs[1].to_LegCharge()
    # w, v now default to 0 and the Identity
    for qindices, block in zip(a._qdata, a._data):  # non-zero blocks on the diagonal
        if hermitian:
            rw, rv = sp.linalg.eigh(block, lower=UPLO == "L", driver="evd")
        else:
            rw, rv = np.linalg.eig(block)
        if sort is not None:  # apply sorting options
            perm = argsort(rw, sort)
            rw = np.take(rw, perm)
            rv = np.take(rv, perm, axis=1)
        qi = qindices[0]  # both `a` and `resv` are sorted and share the same qindices
        resv._data[qi] = rv  # replace identity block
        resw[a.legs[0].get_slice(qi)] = rw  # replace eigenvalues
    if len(piped_axes) > 0:
        resv = resv.split_legs(0)  # the 'outer' facing leg is permuted back.
    return resw, resv


def eigh(a: Array, UPLO="L", sort=None) -> tuple[Array, Array]:
    r"""Calculate eigenvalues and eigenvectors for a hermitian matrix.

    ``W, V = eigh(a)`` yields :math:`a = V diag(w) V^{\dagger}`.
    **Assumes** that a is hermitian, ``a.conj().transpose() == a``.

    Parameters
    ----------
    a : :class:`Array`
        The hermitian square matrix to be diagonalized.
    UPLO : {'L', 'U'}
        Whether to take the lower ('L', default) or upper ('U') triangular part of `a`.
    sort : {'m>', 'm<', '>', '<', ``None``}
        How the eigenvalues should are sorted *within* each charge block.
        Defaults to ``None``, which is same as '<'. See :func:`argsort` for details.

    Returns
    -------
    W : 1D ndarray
        The eigenvalues, sorted within the same charge blocks according to `sort`.
    V : :class:`Array`
        Unitary matrix; ``V[:, i]`` is normalized eigenvector with eigenvalue ``W[i]``.
        The first label is inherited from `A`, the second label is ``'eig'``.

    Notes
    -----
    Faster but more memory intensive than TenPy eigh,
    as it uses the evd family of LAPACK drivers.
    Requires the legs to be contractible.
    If `a` is not blocked by charge, a blocked copy is made via a permutation ``P``,
    :math:`a' =  P a P^{-1} = V' W' (V')^{\dagger}`.
    The eigenvectors `V` are then obtained by the reverse permutation,
    :math:`V = P^{-1} V'` such that :math:`a = V W V^{\dagger}`.
    """
    w, v = _eig_worker(True, a, sort, UPLO)  # hermitian
    v.iset_leg_labels([a._labels[0], "eig"])
    return w, v
