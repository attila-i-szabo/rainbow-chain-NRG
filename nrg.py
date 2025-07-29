"""Performs a number of NRG iterations with free boundary conditions.

Usage: nrg.py [parameter JSON file]

Output is written into an HDF5 (results) and a JSON (parameters) file."""

# Copyright (C) Attila Szabó, Apache license

from sys import argv, stderr

import numpy as np

import h5py
from tenpy.tools import hdf5_io
import json

from include import nrg_step, potts_rainbow_tensors, TFI_rainbow_tensors
from include.truncate import check_trunc_par

with open(argv[1]) as f:
    param = json.load(f)
param_clean = {}

model = param["model"]
param_clean["model"] = model
fname = param.get("fnroot", f"nrg_{model}_")

FM = param.get("coupling", "FM")
param_clean["coupling"] = FM
fname += FM
FM = {"FM": -1.0, "AFM": 1.0}[FM]

g = param.get("g", 1.0)
param_clean["g"] = g
if "g_name" in param:
    fname += "_" + param["g_name"]
    param_clean["g_name"] = param["g_name"]
elif not np.isclose(g, 1.0):
    fname += f"_{g:.3f}"

L = param["L"]
param_clean["L"] = L

Δ = param["Delta"]
param_clean["Delta"] = Δ
if Δ <= 0:
    raise ValueError(f"NRG step size must be positive, got {Δ}")
fname += f"_{L}x{Δ:.3f}"

if model == "Potts3":
    symmetry = param.get("symmetry", "Z3")
    param_clean["symmetry"] = symmetry
    if symmetry not in ["Z3", "Z2"]:
        raise ValueError(f"Unknown symmetry {symmetry!r}, expected 'Z3' or 'Z2'")
    if symmetry == "Z2":
        fname += "_Z2"

    # NRG tensors and trivial starting Hamiltonian
    MPO, H = potts_rainbow_tensors(Δ, FM, g, symmetry)
elif model == "TFI":
    # NRG tensors and trivial starting Hamiltonian
    MPO, H = TFI_rainbow_tensors(Δ, FM, g)
else:
    raise ValueError(f"Unknown model {model!r}, expected 'TFI' or 'Potts3'")

# Read NRG runtime params
trunc_par = check_trunc_par(param.get("trunc_par", {}), "energy")
param_clean["trunc_par"] = trunc_par
fname += f"_chi{trunc_par['chi_max']}"

save_mps = param.get("save_mps", True)
param_clean["save_mps"] = save_mps
save_env = param.get("save_env")
param_clean["save_env"] = save_env

# Save parameters
with open(f"{fname}.json", "w") as f:
    json.dump(param_clean, f, indent=4)

with h5py.File(f"{fname}_zipped.h5", "w") as f:
    # Dump basic info, set up containers
    hdf5_io.save_to_hdf5(
        f,
        {
            "mps": [],
            "spectrum": [],
            "env": {},
            "mpo_tensor": MPO,
            "parameters": param_clean,
        },
    )

    # main NRG loop
    n_env = 0
    for i in range(L // 2):
        print("Rung", i, file=stderr)
        v, H, e = nrg_step(H, MPO, trunc_par=trunc_par)
        if e.size != trunc_par["chi_max"]:
            print(
                f"Kept {e.size} eigenvalues instead of {trunc_par['chi_max']}",
                file=stderr,
            )
        if save_mps:
            hdf5_io.save_to_hdf5(f, v, f"/mps/{i}")

        print("GS energy:", e.min(), file=stderr)

        # save spectrum
        outleg = v.get_leg("vO")
        print("Contents of spectrum:", file=stderr)
        print(outleg, file=stderr)
        sectors = outleg.to_qdict()
        hdf5_io.save_to_hdf5(f, {q: e[sectors[q]] for q in sectors}, f"/spectrum/{i}")
        if save_env is not None and (i + 1) % save_env == 0:
            hdf5_io.save_to_hdf5(f, i + 1, f"/env/keys/{n_env}")
            hdf5_io.save_to_hdf5(f, H, f"/env/values/{n_env}")
            n_env += 1

    # Fix length attributes
    f.get("/spectrum").attrs.modify("len", L // 2)
    if save_mps:
        f.get("/mps").attrs.modify("len", L // 2)
    if save_env is not None:
        f.get("/env/keys").attrs.modify("len", n_env)
        f.get("/env/values").attrs.modify("len", n_env)

print("Parameters saved to", f"{fname}.json", file=stderr)
print("Data output saved to", f"{fname}_zipped.h5", file=stderr)
