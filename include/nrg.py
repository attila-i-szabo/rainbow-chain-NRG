import numpy as np
import tenpy.linalg.np_conserved as npc

from .linalg import eigh as eigh_fast
from .truncate import truncate


def nrg_step(
    H: npc.Array, W: tuple[npc.Array, npc.Array], trunc_par=None, Id=(0, 1), fast=True
) -> tuple[npc.Array, npc.Array, np.ndarray]:
    """Performs one step of the NRG protocol.

    Args
    ----
    H: effective Hamiltonian, equivalent to:

            V -- ... -- V --
            |           |
            W -- ... -- W --
            |           |
            V -- ... -- V --
    with legs

        * `wL` (MPO virtual leg to the left)
        * `wR` (MPO virtual leg to the left, conj.)
        * `vO` (MPS ket virtual leg to higher indices, conj.)
        * `vO*` (MPS bra virtual leg to higher indices)

    W: MPO tensors for the left and right halves of the chain, with legs
        * `wL` (MPO virtual leg to the left)
        * `wR` (MPO virtual leg to the left, conj.)
        * `p` (physical leg connected to bra side)
        * `p*` (physical leg connected to ket side, conj.)

    Id: (2,) tuple, "only identities to the left/right" indices in `wL`/`wR`

    trunc_par: truncation parameters, cf. :function:`truncate.truncate`
        if `None` (default), the full eigenspace is returned

    fast: use "evd" family of eigensolvers (faster but takes more memory)

    Returns
    -------
    v: next tensor of the zipped MPS, with legs
        * `vI` (to lower indices)
        * `vO` (to higher indices/eigenvector index, conj.)
        * `pL` (physical leg on negative site)
        * `pR` (physical leg on positive site)

    H: effective Hamiltonian for a chain one site longer

    e: energy eigenvalues
        `e[i]` is the eigenvalue corresponding to the eigenvector `v[:,i]`
    """
    WL, WR = W
    IdL, IdR = Id
    ## build effective Hamiltonian
    # fuse slice IdL of WL, rename p to pL
    Heff = npc.tensordot(H, WL.take_slice(IdL, "wL"), axes=("wL", "wR"))
    Heff.ireplace_labels(["p", "p*"], ["pL", "pL*"])
    # fuse slice IdR of WR, rename p to pR
    Heff = npc.tensordot(Heff, WR.take_slice(IdR, "wR"), axes=("wR", "wL"))
    Heff.ireplace_labels(["p", "p*"], ["pR", "pR*"])
    ## diagonalise
    Heff = Heff.combine_legs([["vO", "pL*", "pR*"], ["vO*", "pL", "pR"]], [1, 0])
    if fast:
        e, v = eigh_fast(Heff)
    else:
        e, v = npc.eigh(Heff)
    # truncate
    if trunc_par is not None:
        mask = truncate(e, trunc_par)
        e = e[mask]
        v = v[:, mask]
    # result tensor with unpacked legs
    v = v.split_legs(0).ireplace_labels(["eig", "vO*"], ["vO", "vI"])
    ## build new effective Hamiltonian
    # fuse v, WL, WR, v* in order
    H = npc.tensordot(H, v, ["vO", "vI"])
    H = npc.tensordot(WR, H, [["wL", "p*"], ["wR", "pR"]]).ireplace_label("p", "pR")
    H = npc.tensordot(WL, H, [["wR", "p*"], ["wL", "pL"]]).ireplace_label("p", "pL")
    H = npc.tensordot(H, v.conj(), [["vO*", "pL", "pR"], ["vI*", "pL*", "pR*"]])
    assert H._labels == ["wL", "wR", "vO", "vO*"]
    # subtract GS energy and scale energies
    H[IdL, IdR] -= e.min() * npc.eye_like(H, "vO", ["vO", "vO*"])
    return v, H, e
