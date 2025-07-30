"""Unzips a folded MPS wave function from NRG.

Usage: potts_unzip.py [HDF5 file] [parameter JSON file]

Output is written into an HDF5 (results) and a JSON (parameters) file."""

# Copyright (C) Attila Szabó, Apache license

from sys import argv, stderr

import numpy as np
import tenpy.linalg.np_conserved as npc

import h5py
from tenpy.tools import hdf5_io
import json

from include import unzip
from include.truncate import check_trunc_par

with h5py.File(argv[1], "r") as f:
    MPS = hdf5_io.load_from_hdf5(f, "/mps")
fname = argv[1].removesuffix("_zipped.h5") + "_unzip"

with open(argv[2]) as f:
    param = json.load(f)
param_clean = {}

# exploit that NRG tensor trains can be truncated to get a shorter rainbow chain
if "L" in param:
    L = param["L"]
    assert L % 2 == 0
    L = L // 2
    assert len(MPS) >= L, f"Input MPS too short ({len(MPS)} instead of requested {L})"
    fname += f"_L{2 * L}"
    MPS = MPS[:L]
else:
    L = len(MPS)
param_clean["L"] = 2 * L

# Parse "charge"
if param["charge"] in ["+", "-"]:
    symmetry = 2
    charge = {"+": 0, "-": 1}[param["charge"]]
    print("Total parity sector", param["charge"], file=stderr)
    param_clean["charge"] = param["charge"]
elif isinstance(param["charge"], int):
    symmetry = 3
    charge = param["charge"] % 3
    param_clean["charge"] = charge
    print("Total Z3 charge sector", charge, file=stderr)
else:
    raise ValueError(
        f"Invalid charge {param['charge']!r}, expected '+', '-', or an integer"
    )
fname += f"_Q{param['charge']}"

# Check MPS has the correct charge throughout
for i, v in enumerate(MPS):
    q = v.chinfo.mod.item()
    assert q == symmetry, f"Unexpected charge type {q} != {symmetry} on tensor {i}"

# Find lowest energy state in charge sector
outleg = MPS[-1].get_leg("vO")
q_slice = outleg.to_qdict()[(charge,)]
idx = q_slice.start

# Z2 parity if Q = 0
compute_z2 = charge == 0 and symmetry == 3
if compute_z2:
    leg = MPS[0].get_leg("pL")
    adapter = np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]])  # connects q to -q
    adapter = npc.Array.from_ndarray(
        adapter, [leg.conj(), leg.conj()], dtype=float, labels=["a", "b"]
    )

    def callback(A, B, M):
        M = npc.tensordot(M, A, ["vR", "vL"])
        M = npc.tensordot(M, adapter, ["p", "a"])
        M = npc.tensordot(
            M, A.complex_conj().ireplace_label("vR", "vR*"), [["vR*", "b"], ["vL", "p"]]
        )
        return M

    triv_leg = npc.LegCharge.from_qflat(leg.chinfo, [[0]])
    M = npc.Array.from_ndarray([[1.0]], labels=["vR", "vR*"], legcharges=[triv_leg] * 2)

# Read parameters of unzipping
trunc_par = check_trunc_par(param.get("trunc_par", {}), "schmidt")
param_clean["trunc_par"] = trunc_par
fname += f"_chi{trunc_par['chi_max']}_cutoff{trunc_par['svd_min']}"

algorithm = param.get("algorithm", "SVD")
param_clean["algorithm"] = algorithm

save_mps = param.get("save_mps", True)
param_clean["save_mps"] = save_mps

with open(f"{fname}.json", "w") as f:
    json.dump(param_clean, f, indent=4)

if compute_z2:
    uMPS, M = unzip(
        MPS,
        idx,
        trunc_par,
        algorithm,
        return_mps=save_mps,
        callback=callback,
        callback_args=M,
    )
else:
    uMPS = unzip(MPS, idx, trunc_par, algorithm, return_mps=save_mps)

# Assemble HDF5 output
if save_mps:
    S, leg, As, Bs = uMPS
    pickle = {"S": S, "leg": leg, "A": As, "B": Bs, "parameters": param_clean}
else:
    S, leg = uMPS
    pickle = {"S": S, "leg": leg, "parameters": param_clean}

if compute_z2:
    pickle["Z2"] = M

with h5py.File(f"{fname}.h5", "w") as f:
    hdf5_io.save_to_hdf5(f, pickle)

print("Parameters saved to", f"{fname}.json", file=stderr)
print("Data output saved to", f"{fname}.h5", file=stderr)

# Save entanglement spectrum in human readable form
q = leg.to_qflat()
ES = -2 * np.log(S)
if compute_z2:
    np.savetxt(
        f"{fname}_ES_parity.txt",
        np.column_stack((q, ES, np.diag(M.to_ndarray().real))),
        fmt="%d %.8f %7.4f",
    )
    print("Entanglement spectrum saved to", f"{fname}_ES_parity.txt", file=stderr)
else:
    np.savetxt(f"{fname}_ES.txt", np.column_stack((q, ES)), fmt="%d %.8f")
    print("Entanglement spectrum saved to", f"{fname}_ES.txt", file=stderr)
