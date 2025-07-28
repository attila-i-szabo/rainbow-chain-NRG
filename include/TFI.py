import numpy as np

from tenpy.networks.site import SpinHalfSite
from tenpy.linalg.charges import LegCharge
import tenpy.linalg.np_conserved as npc

# "only identities to the left/right" indices for both implementations
IdL = 0
IdR = 1


def MPO_tensor(scale_L, scale_R, scale_Z, scale_X1, scale_X2, dtype=None):
    """Builds an MPO tensor for the transverse field Ising model.

    Args
    ----
    scale_L: scale factor of the identity on the "IdL" leg
    scale_R: scale factor of the identity on the "IdR" leg
    scale_Z: scale factor of the transverse field
    scale_X1: scale factor of the first X operator in the XX term
    scale_X2: scale factor of the second X operator in the XX term
    dtype: of the output, default: least common dtype of the scale factors
    """

    if dtype is None:
        dtype = np.asarray([scale_L, scale_R, scale_Z, scale_X1, scale_X2]).dtype

    W = np.zeros((3, 3, 2, 2), dtype=dtype)
    I = np.eye(2)

    site = SpinHalfSite(conserve="parity")
    pleg = site.leg
    Wleg = LegCharge(pleg.chinfo, [0, 2, 3], [[0], [1]])

    W[0, 0] = scale_L * I
    W[0, 1] = scale_Z * site.get_op("Sigmaz").to_ndarray()
    W[1, 1] = scale_R * I
    W[0, 2] = scale_X1 * site.get_op("Sigmax").to_ndarray()
    W[2, 1] = scale_X2 * site.get_op("Sigmax").to_ndarray()

    W = npc.Array.from_ndarray(
        W, [Wleg, Wleg.conj(), pleg, pleg.conj()], labels=["wL", "wR", "p", "p*"]
    )
    return W


def rainbow_tensors(Δ, FM=-1.0, g=1.0):
    """Builds MPO tensors needed for NRG of the transverse field Ising model.

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
        MPO_tensor(f(1 / 2), f(-1 / 2), FM * g * f(-1 / 2), FM * f(1 / 8), f(-5 / 8)),
        MPO_tensor(f(-1 / 2), f(1 / 2), FM * g * f(-1 / 2), FM * f(-5 / 8), f(1 / 8)),
    )
    H = npc.eye_like(MPO[0], "wL", ["wL", "wR"])
    H = H.add_trivial_leg(label="vO", qconj=-1).add_trivial_leg(label="vO*")
    return MPO, H
