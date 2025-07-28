"""Adaptation of TenPy's truncation mechanism to energy spectra."""

# Copyright (C) TeNPy Developers and Attila Szabó, Apache license

import numpy as np

from tenpy.tools.params import asConfig


def _combine_constraints(good1, good2, warn):
    """return logical_and(good1, good2) if there remains at least one `True` entry.

    Otherwise print a warning and return just `good1`.
    """
    res = np.logical_and(good1, good2)
    if np.any(res):
        return res
    warn("truncation: can't satisfy constraint for " + warn, stacklevel=3)
    return good1


def check_trunc_par(trunc_par, form="schmidt"):
    """Sanitise truncation parameters and set some sensible defaults.

    Args:
        trunc_par: dict of truncation parameters, see `truncate` here and in
            `tenpy.algorithms.truncation`
        form: str
            "schmidt" for compatibility with `tenpy.algorithms.truncation.truncate`
            "energy" for `.truncate`

    Returns:
        trunc_par with defaults added as needed
    """
    # check there are no invalid keys
    if form == "schmidt":
        for key in trunc_par:
            assert key in [
                "chi_max",
                "chi_min",
                "degeneracy_tol",
                "svd_min",
                "trunc_cut",
            ]
    elif form == "energy":
        for key in trunc_par:
            assert key in ["chi_max", "chi_min", "degeneracy_tol", "e_max"]
    else:
        raise ValueError(f'form must be "schmidt" or "energy", got {form}')
    # set default chi_max
    if "chi_max" not in trunc_par:
        trunc_par["chi_max"] = 500
    # set default degeneracy_tol
    if "degeneracy_tol" not in trunc_par:
        trunc_par["degeneracy_tol"] = 1e-10
    # set default svd_min for schmidt
    if form == "schmidt" and "svd_min" not in trunc_par:
        trunc_par["svd_min"] = 1e-8

    return trunc_par


def truncate(E, options):
    """Given an energy spectrum `E`, determine which values to keep.

    Options
    -------
    .. cfg:config:: truncation

        chi_max : int
            Keep at most `chi_max` eigenvalues.
        chi_min : int
            Keep at least `chi_min` eigenvalues.
        degeneracy_tol: float
            Don't cut between neighboring eigenvalues with
            differences below `degeneracy_tol`.
            In other words, keep either both `i` and `j` or none, if the
            eigenvalues are degenerate within`degeneracy_tol`, which we
            expect to happen in the case of symmetries.
        e_max : float
            Discard all energies above `e_max`.

    Parameters
    ----------
    E : 1D array
        Energies (as returned by an eigensolver), not necessarily sorted.
    options: dict-like
        Config with constraints for the truncation, see :cfg:config:`truncation`.
        If a constraint can not be fulfilled (without violating a previous one), it is ignored.
        A value ``None`` indicates that the constraint should be ignored.

    Returns
    -------
    mask : 1D bool array
        Index mask, True for indices which should be kept.
    """
    options = asConfig(options, "truncation")

    chi_max = options.get("chi_max", None, int)
    chi_min = options.get("chi_min", None, int)
    deg_tol = options.get("degeneracy_tol", None, "real")
    e_max = options.get("e_max", None, "real")

    piv = np.argsort(E)[::-1]  # sort *descending*
    ES = E[piv]

    # goal: find an index 'cut' such that we keep piv[cut:], i.e. cut between `cut-1` and `cut`.
    good = np.ones(len(piv), dtype=np.bool_)  # good[cut] = (is `cut` a good choice?)
    # we choose the smallest 'good' cut.

    if chi_max is not None:
        # keep at most chi_max values
        good2 = np.zeros(len(piv), dtype=np.bool_)
        good2[-chi_max:] = True
        good = _combine_constraints(good, good2, "chi_max")

    if chi_min is not None and chi_min > 1:
        # keep at most chi_max values
        good2 = np.ones(len(piv), dtype=np.bool_)
        good2[-chi_min + 1 :] = False
        good = _combine_constraints(good, good2, "chi_min")

    if deg_tol:
        # don't cut between values (cut-1, cut) with ``ES[cut-1] - ES[cut] < deg_tol``
        good2 = np.empty(len(piv), np.bool_)
        good2[0] = True
        good2[1:] = np.greater_equal(ES[:-1] - ES[1:], deg_tol)
        good = _combine_constraints(good, good2, "degeneracy_tol")

    if e_max is not None:
        # keep only values E[i] <= e_max
        good2 = np.less_equal(ES, e_max)
        good = _combine_constraints(good, good2, "e_max")

    cut = np.nonzero(good)[0][0]  # smallest possible cut: keep as many E as allowed
    mask = np.zeros(len(E), dtype=np.bool_)
    np.put(mask, piv[cut:], True)
    return mask
