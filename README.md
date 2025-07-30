## Numerical RG for interacting rainbow chains

This repository implements the numerical RG algorithm used to iteratively diagonalise rainbow chains in the paper [Rainbow chains and numerical renormalisation group for accurate chiral conformal spectra](https://arxiv.org/abs/2412.09685).

### Usage guide

#### Running an NRG simulation:

```
nrg.py json/potts_nrg.json
```

Some important fields of the JSON parameter file:

* `"model"`: `"Potts3"` or `"TFI"`, controls which Hamiltonian to simulate
* `"coupling"`: `"FM"` or `"AFM"`, controls the overall sign of the Hamiltonian
* `"L"`: length of the rainbow chain (before folding)
* `"Delta"`: exponential decay rate of the rainbow chain
* `"g"`: value of $g = h/J$ for the simulation
* `"trunc_par"`: describes how to truncate the low-energy NRG spectrum
    * `"chi_max"`: maximum number of states to keep
    * `"degeneracy_tol"`: states within this energy difference are considered degenerate and are either both kept or both discarded
* `"g_name"`: string to indicate the value of $g=h/J$ in file names (default: $g$ to 3 digits)
* `"save_mps"`: whether to save the actual MPS or only the evolution of the NRG spectrum

The output is a JSON file of the valid parameters and an HDF5 file containing the evolution of the spectrum and the folded MPS tensors (if requested).

#### Fine-tuning to the critical point

```
finetune.py json/potts_finetune.json
```

For each guess of $g=h/J$, this code performs an NRG simulation up to a fixed number of steps or until the ground-state energy leaves a given range (considered consistent with the critical spectrum). A guess is accepted if the maximum number of steps is reached without leaving the given energy range.

Otherwise, a new guess is computed from the evolution of the ground state energy for the previous two guesses: It is assumed that the deviation from the critical case (specifically the energy difference between step *n* and *n*+1) is linear in $g-g_c$, and the new guess is obtained with the method of secants. The program also terminates if the update to the guess is smaller than a given threshold.

Most parameter fields for NRG are still valid (of course not `"g"`, we are looking for that). In addition, we have a `"finetune"` field, with the following subfields:

* `"guess"`: a list of two initial guesses for $g=h/J$
* `"max"`, `"min"`: the highest and lowest energies considered consistent with the ground state in the critical spectrum
* `"burn_in"`: the number of NRG steps that must be performed before the NRG may be terminated
* `"max_step"`: the maximum number of NRG steps before the guess is considered converged
* `"tolerance"`: change in guess before it is considered converged
* `"max_iter"`: maximum number of guesses before terminating

The output is a JSON file containing all input parameters relevant for NRG as well as the converged value of $g$.

### Implementation notes

### License

The code is freely usable under the Apache License v2.0. If you use it in your own projects, please cite the paper [Rainbow chains and numerical renormalisation group for accurate chiral conformal spectra](https://arxiv.org/abs/2412.09685).

Some code from [TeNPy](https://tenpy.readthedocs.io/), also licensed under the Apache License, is incorporated.