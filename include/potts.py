"""Functions to build MPO tensors for NRG of the three-state Potts model."""

# Copyright (C) Attila Szabó, Apache license

import numpy as np

from tenpy.networks.site import ClockSite
from tenpy.linalg.charges import LegCharge
import tenpy.linalg.np_conserved as npc

# "only identities to the left/right" indices for both implementations
IdL = 0
IdR = 1


def MPO_tensor(
    scale_L, scale_R, scale_Z, scale_X1, scale_X2, symmetry="Z3", dtype=None
):
    """Builds an MPO tensor for three-state Potts model.

    Args
    ----
    scale_L: scale factor of the identity on the "IdL" leg
    scale_R: scale factor of the identity on the "IdR" leg
    scale_Z: scale factor of the transverse field
    scale_X1: scale factor of the first X operator in the XX term
    scale_X2: scale factor of the second X operator in the XX term
    symmetry: "Z3" or "Z2", subgroup of S3 to be enforced
    dtype: of the output, default: least common dtype of the scale factors
    """

    if dtype is None:
        dtype = np.asarray([scale_L, scale_R, scale_Z, scale_X1, scale_X2]).dtype

    W = np.zeros((4, 4, 3, 3), dtype=dtype)
    I = np.eye(3)

    if symmetry == "Z3":
        pleg = ClockSite(3).leg
        Wleg = LegCharge(pleg.chinfo, [0, 2, 3, 4], [[0], [1], [2]])

        X = np.roll(np.eye(3), 1, axis=0)
        Xhc = X.T
        ZZhc = np.diag(2 * np.cos(2 * np.pi * np.arange(3) / 3))

        W[0, 0] = scale_L * I
        W[0, 1] = scale_Z * ZZhc
        W[1, 1] = scale_R * I
        W[0, 2] = scale_X1 * X
        W[2, 1] = scale_X2 * Xhc
        W[0, 3] = scale_X1 * Xhc
        W[3, 1] = scale_X2 * X
    elif symmetry == "Z2":
        chinfo = npc.ChargeInfo([2], ["parity"])
        pleg = LegCharge(chinfo, [0, 2, 3], [[0], [1]])
        Wleg = npc.LegCharge(chinfo, [0, 3, 4], [[0], [1]])

        # real and imaginary (parity-preserving and -flipping) parts of the X operator
        XR = np.diag([2, -1, -1]) / 2**0.5
        XI = np.asarray([[0, 0, 0], [0, 0, 1.5**0.5], [0, 1.5**0.5, 0]])
        ZZhc = np.asarray([[0, 2**0.5, 0], [2**0.5, 1, 0], [0, 0, -1]])

        W[0, 0] = scale_L * I
        W[0, 1] = scale_Z * ZZhc
        W[1, 1] = scale_R * I
        W[0, 2] = scale_X1 * XR
        W[2, 1] = scale_X2 * XR
        W[0, 3] = scale_X1 * XI
        W[3, 1] = scale_X2 * XI
    else:
        raise ValueError(f'Unknown symmetry {symmetry}, expected "Z3" or "Z2"')

    W = npc.Array.from_ndarray(
        W, [Wleg, Wleg.conj(), pleg, pleg.conj()], labels=["wL", "wR", "p", "p*"]
    )
    return W


def rainbow_tensors(Δ, FM=-1.0, g=1.0, symmetry="Z3"):
    """Builds MPO tensors needed for NRG of the three-state Potts model.

    Args
    ----
    Δ: scale factor of NRG
    FM: whether the interaction is ferromagnetic (-1, default)
        or antiferromagnetic (+1)
    g: renormalised field h/J (default: 1.0)

    Returns
    -------
    MPO: (2,) tuple of MPO tensors for the left and right halves of the chain
    H0: dummy environment tensor for the first NRG step
    """
    f = lambda x: np.exp(x * Δ)
    MPO = (
        MPO_tensor(
            f(1 / 2), f(-1 / 2), FM * g * f(-1 / 2), FM * f(1 / 8), f(-5 / 8), symmetry
        ),
        MPO_tensor(
            f(-1 / 2), f(1 / 2), FM * g * f(-1 / 2), FM * f(-5 / 8), f(1 / 8), symmetry
        ),
    )
    H = npc.eye_like(MPO[0], "wL", ["wL", "wR"])
    H = H.add_trivial_leg(label="vO", qconj=-1).add_trivial_leg(label="vO*")
    return MPO, H
