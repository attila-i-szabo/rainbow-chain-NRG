from sys import stderr
from datetime import datetime

import tenpy.linalg.np_conserved as npc
from tenpy.linalg.truncation import svd_theta


def slice_mps_end(X: npc.Array, idx):
    """Extracts state number `idx` from the final tensor `X` of the folded MPS.

    The return tensor is given two extra legs `vL` and `vR`; the latter
    is conjugated and carries the charge of the extracted eigenstate."""
    X = X.take_slice(idx, "vO")
    X = X.add_trivial_leg(label="vL", qconj=+1)
    X = X.add_trivial_leg(label="vR", qconj=-1)
    X = X.gauge_total_charge("vR")
    return X


def qr_step(X: npc.Array, trunc_par, verbose=True):
    """Performs one step of the unzipping algorithm, using QR decomposition
    to split off isometries.

    Args
    ----
    X: input tensor, expected to have legs
        * `vI`: virtual leg to lower indices
        * `vL`: virtual leg to unzipped sites to the left
        * `vR`: virtual leg to unzipped sites to the right, conj.
        * `pL`, `pR`: physical legs

    trunc_par: truncation parameters, cf. `tenpy.algorithms.truncation.truncate`
        (normalisation does *not* match the SVD context there)

    verbose: whether to print error estimates to stderr

    Returns
    -------
    A: left canonical MPS tensor, with legs `vL`, `p`, `vR`

    X: remainder to be fused into MPS tensor to the inside,
        with legs `vL`, `vR`, `vI`

    B: right canonical MPS tensor, with legs `vL`, `p`, `vR`
    """
    X = X.combine_legs([["pR", "vR"], ["vI", "pL", "vL"]], [0, 1])
    B, X, err = npc.qr(
        X, inner_labels=["vL", "vR"], inner_qconj=-1, trunc_par=trunc_par
    )
    B = B.split_legs(0).ireplace_label("pR", "p")
    if verbose:
        print(err, file=stderr)
    X = X.split_legs(1).combine_legs([["vL", "pL"], ["vI", "vR"]], [0, 1])
    A, X, err = npc.qr(X, inner_labels=["vR", "vL"], trunc_par=trunc_par)
    A = A.split_legs(0).ireplace_label("pL", "p")
    if verbose:
        print(err, file=stderr)
    X = X.split_legs(1)
    if verbose:
        print(A.shape, X.shape, B.shape, file=stderr)
    return A, X, B


def svd_step(X: npc.Array, trunc_par, verbose=True):
    """Performs one step of the unzipping algorithm, using SVD
    to split off isometries.

    Args
    ----
    X: input tensor, expected to have legs
        * `vI`: virtual leg to lower indices
        * `vL`: virtual leg to unzipped sites to the left
        * `vR`: virtual leg to unzipped sites to the right, conj.
        * `pL`, `pR`: physical legs

    trunc_par: truncation parameters, cf. `tenpy.algorithms.truncation.truncate`

    verbose: whether to print error estimates to stderr

    Returns
    -------
    A: left canonical MPS tensor, with legs `vL`, `p`, `vR`

    X: remainder to be fused into MPS tensor to the inside,
        with legs `vL`, `vR`, `vI`

    B: right canonical MPS tensor, with legs `vL`, `p`, `vR`
    """
    # split off tensor B to the right
    X = X.combine_legs([["vI", "pL", "vL"], ["pR", "vR"]], [0, 1])
    X, S, B, err, renorm = svd_theta(X, trunc_par, inner_labels=["vR", "vL"])
    if verbose:
        print(err, renorm, file=stderr)
    B = B.split_legs(1).ireplace_label("pR", "p")
    X.iscale_axis(S, axis="vR")
    # split off tensor A to the left
    X = X.split_legs(0).combine_legs([["vL", "pL"], ["vI", "vR"]], [0, 1])
    A, S, X, err, renorm = svd_theta(X, trunc_par, inner_labels=["vR", "vL"])
    if verbose:
        print(err, renorm, file=stderr)
    A = A.split_legs(0).ireplace_label("pL", "p")
    X = X.iscale_axis(S, axis="vL").split_legs(1)
    if verbose:
        print(A.shape, X.shape, B.shape, file=stderr)
    return A, X, B


def last_svd_step(X: npc.Array, trunc_par, verbose=True):
    """Performs the last step of the unzipping algorithm:
    squeezes the `vI` leg of the centremost tensor and performs an SVD

    Args
    ----
    X: input tensor, expected to have legs
        * `vI`: virtual leg to lower indices
        * `vL`: virtual leg to unzipped sites to the left
        * `vR`: virtual leg to unzipped sites to the right, conj.
        * `pL`, `pR`: physical legs

    trunc_par: truncation parameters, cf. `tenpy.algorithms.truncation.truncate`

    verbose: whether to print error estimates to stderr

    Returns
    -------
    A: left canonical MPS tensor, with legs `vL`, `p`, `vR`

    S: Schmidt values

    B: right canonical MPS tensor, with legs `vL`, `p`, `vR`
    """
    X = X.squeeze("vI")
    X = X.combine_legs([["pL", "vL"], ["pR", "vR"]], [0, 1])
    A, S, B, err, renorm = svd_theta(X, trunc_par, inner_labels=["vR", "vL"])
    A = A.split_legs(0).ireplace_label("pL", "p")
    B = B.split_legs(1).ireplace_label("pR", "p")
    if verbose:
        print(err, renorm, file=stderr)
        print(A.shape, S.shape, B.shape, file=stderr)
    return A, S, B


def unzip(
    MPS,
    idx=None,
    trunc_par={"chi_max": 1000, "svd_min": 1e-8},
    algorithm="SVD",
    return_mps=True,
    verbose=True,
):
    """Takes a zipped mps (given as a list of Arrays) and unzips it into
    tensors of a standard MPS.

    Args
    ----
    MPS: a list of `npc.Array`s representing the zipped MPS

    idx: index of eigenstate encoded in `MPS` to be unzipped
        if `None` (default), `slice_mps_end` not called,
        and its output format is expected for `MPS[-1]`

    trunc_par: truncation parameters, cf. `tenpy.algorithms.truncation.truncate`

    algorithm: method of splitting off isometries, can be "SVD" or "QR"

    return_mps: return the unzipped MPS tensors in addition to the Schmidt values

    verbose: whether to print runtime information to stderr

    Returns
    -------
    if `return_mps`: A, B, (S, leg)
    else: S, leg

    A: list of left-canonical MPS tensors from left to right

    B: list of right-canonical MPS tensors from right top left

    S: Schmidt values in the middle

    leg: `LegCharge` corresponding to the array `S`
    """
    if return_mps:
        As = []
        Bs = []
    X = MPS[-1] if idx is None else slice_mps_end(MPS[-1], idx)
    L = len(MPS)

    for i in range(L - 1, 0, -1):
        if verbose:
            print("Rung", i, file=stderr)
            print(datetime.now(), file=stderr)

        if algorithm == "SVD":
            A, X, B = svd_step(X, trunc_par, verbose)
        elif algorithm == "QR":
            A, X, B = qr_step(X, trunc_par, verbose)
        else:
            raise ValueError(f"Invalid algorithm {algorithm!r}, must be 'SVD' or 'QR'")

        X = npc.tensordot(MPS[i - 1], X, ["vO", "vI"])

        if return_mps:
            As.append(A)
            Bs.append(B)

    # i = 0
    if verbose:
        print("Rung 0", file=stderr)
        print(datetime.now(), file=stderr)

    A, S, B = last_svd_step(X, trunc_par, verbose)

    if return_mps:
        As.append(A)
        Bs.append(B)

    # Output
    if return_mps:
        return As, Bs, (S, B.get_leg("vL"))
    else:
        return S, B.get_leg("vL")  # leg needed to assign S to charge sectors
