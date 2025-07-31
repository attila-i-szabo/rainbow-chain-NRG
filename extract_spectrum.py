"""Extracts one NRG spectrum from an HDF5 file.

Usage: extract_spectrum.py [HDF5 file] [size of rainbow chain]

Output: low-energy spectrum in a text file, columns: charge(s), energy."""

# Copyright (C) Attila Szabó, Apache license

import numpy as np
import h5py
from tenpy.tools import hdf5_io
from sys import argv, stderr

fname = argv[1].removesuffix("_zipped.h5")

L = int(argv[2])
fname += f"_L{L}_spectrum.txt"

with h5py.File(argv[1], "r") as f:
    specs = hdf5_io.load_from_hdf5(f, "/spectrum")

spec = specs[L // 2 - 1]
spec_out = []
for q, es in spec.items():
    n_q = len(q)
    spec_out.append(np.column_stack((np.ones((es.size, 1)) * q, es)))
spec_out = np.concatenate(spec_out)

np.savetxt(fname, spec_out, fmt="%3d " * n_q + "%11.8f")
print("Output data saved to", fname, file=stderr)
