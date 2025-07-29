"""Finds the value of g=h/J corresponding to the renormalised critical point.

Usage: nrg.py [parameter JSON file]

Output (parameters for NRG) written into a JSON file."""

# Copyright (C) Attila Szabó, Apache license

from sys import argv, exit, stderr
import json
from functools import partial

import numpy as np

from include import nrg_step, potts_rainbow_tensors, TFI_rainbow_tensors
from include.truncate import check_trunc_par

with open(argv[1]) as f:
    param = json.load(f)
param_clean = {}

model = param["model"]
param_clean["model"] = model
fname = param.get("fnroot", f"nrg_{model}_")

FM = param.get("coupling", "FM")  # todo complex couplings?
param_clean["coupling"] = FM
fname += FM
FM = {"FM": -1.0, "AFM": 1.0}[FM]

Δ = param["Delta"]
param_clean["Delta"] = Δ
if Δ <= 0:
    raise ValueError("Delta must be positive")
fname += f"_D{Δ:.3f}"

if model == "Potts3":
    symmetry = param.get("symmetry", "Z3")
    param_clean["symmetry"] = symmetry
    if symmetry not in ["Z3", "Z2"]:
        raise ValueError(f"Unknown symmetry {symmetry}, expected 'Z3' or 'Z2'")
    if symmetry == "Z2":
        fname += "_Z2"

    rainbow_tensors = partial(potts_rainbow_tensors, symmetry=symmetry)
elif model == "TFI":
    rainbow_tensors = TFI_rainbow_tensors
else:
    raise ValueError(f"Unknown model {model}, expected 'TFI' or 'Potts3'")

trunc_par = check_trunc_par(param.get("trunc_par", {}), "energy")
param_clean["trunc_par"] = trunc_par
fname += f"_chi{trunc_par['chi_max']}"

fname += "_finetune"
log = open(fname + ".log", "w")

tune = param["finetune"]

L = param["L"]
max_step = tune.get("max_step")
if L is None:
    if max_step is None:
        raise KeyError("Need to specify either `L` or `max_step`.")
    else:
        L = 2 * max_step
elif max_step is None:
    max_step = L // 2
param_clean["L"] = L

# misc other parameters
if "save_mps" in param:
    param_clean["save_mps"] = param["save_mps"]
if "save_env" in param:
    param_clean["save_env"] = param["save_env"]


def make_output(g):
    """Prints the converged value of g, makes JSON output, quits the program"""
    print("# Final value of g:", g, file=log)
    param_clean["g"] = g
    param_clean["g_name"] = "finetune"
    with open(f"{fname}.json", "w") as f:
        json.dump(param_clean, f, indent=4)
    log.close()
    print("NRG parameters saved to", f"{fname}.json", file=stderr)
    exit()


def divergence_step(g):
    """Finds the NRG step where the ground state energy leaves the approved boundaries"""
    print(f"# Starting to test g = {g}\n", file=log)

    # NRG tensors and trivial starting Hamiltonian
    MPO, H = rainbow_tensors(Δ, FM, g)

    es = []
    direction = None
    for i in range(max_step):
        _, H, e = nrg_step(H, MPO, trunc_par=trunc_par)
        if e.size != trunc_par["chi_max"]:
            print(
                f"g={g:18.16f}, step {i}: kept {e.size} eigenvalues instead of {trunc_par['chi_max']}",
                file=stderr,
            )

        emin = e.min()
        es.append(emin)
        print(f"{i:3} {emin:15.12f}", file=log)
        log.flush()

        # test if NRG should be stopped
        if i > tune["burn_in"] and (emin < tune["min"] or emin > tune["max"]):
            direction = "up" if emin > tune["max"] else "down"
            print(f"# Diverges {direction}wards at step {i}", file=log)
            break

    # if direction isn't set, it converged: save result and stop the program
    if direction is None:
        print(f"# Converged up to {max_step} steps", file=log)
        make_output(g)

    return g, direction, es


guesses = [divergence_step(tune["guess"][0]), divergence_step(tune["guess"][1])]


def new_guess(guess):
    """Interpolate the energy diff on the last step that appears in both"""
    c1, _, e1 = guess[-1]
    c2, _, e2 = guess[-2]
    end = min(len(e1), len(e2))
    d1 = e1[end - 1] - e1[end - 2]
    d2 = e2[end - 1] - e2[end - 2]
    return c1 - d1 * (c1 - c2) / (d1 - d2)


max_iter = tune.get("max_iter", 20)
atol = tune.get("tolerance", np.finfo(float).eps)
for i in range(max_iter - 2):
    corr = new_guess(guesses)
    if np.abs(corr - guesses[-1][0]) < atol:
        print(f"# Estimate of g converged to {corr} +/- {atol:.2e}", file=log)
        make_output(corr)
    print(f"# Step {len(guesses)+1}", file=log)
    data = divergence_step(corr)
    guesses.append(data)

# if program didn't stop, it failed to converge
print(f"# Failed to converge in {max_iter} steps", file=log)

# find most successful estimate
n = np.argmax([len(x[2]) for x in guesses])
print(f"# Best estimate of g: {guesses[n][0]}", file=log)
print(f"# It diverges after {len(guesses[n][2])} NRG steps", file=log)
make_output(guesses[n][0])
