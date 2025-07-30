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
* `"g_name"`: string to indicate this value of $g$ in file names (default: $g$ to 3 digits)
* `"save_mps"`: whether to save the actual MPS or only the evolution of the NRG spectrum


### Implementation notes

### License

The code is freely usable under the Apache License v2.0. If you use it in your own projects, please cite the paper [Rainbow chains and numerical renormalisation group for accurate chiral conformal spectra](https://arxiv.org/abs/2412.09685).

Some code from [TeNPy](https://tenpy.readthedocs.io/), also licensed under the Apache License, is incorporated.