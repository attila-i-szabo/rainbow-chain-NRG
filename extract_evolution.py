"""Extracts the evolution of the NRG spectrum from an HDF5 file.

Usage: extract_evolution.py [HDF5 file] [#eigenstates per charge sector (default: 20)]

Output: low-energy spectra in a text file, one step per line."""

# Copyright (C) Attila Szabó, Apache license

import numpy as np
import h5py
from tenpy.tools import hdf5_io
from sys import argv, stderr


def pad(x, max_length):
    if x is None:
        return np.full(max_length, np.nan)
    elif x.size >= max_length:
        return x[:max_length]
    else:
        return np.pad(x, (0, max_length - x.size), constant_values=np.nan)


with h5py.File(argv[1], "r") as f:
    specs = hdf5_io.load_from_hdf5(f, "/spectrum")

fname = argv[1].removesuffix("_zipped.h5") + "_spectrum_evolution.txt"
num = int(argv[2]) if len(argv) > 2 else 20

# Compute largest sector dimension across all saved spectra
max_lengths = {}
for spec in specs:
    for q in spec:
        max_lengths[q] = max(max_lengths.get(q, 0), len(spec[q]))
# cap them at num, write them out into lists in order
sectors = sorted(max_lengths)
max_lengths = np.minimum([max_lengths[q] for q in sectors], num)
boundaries = [0] + np.cumsum(max_lengths).tolist()

with open(fname, "w") as f:
    # Write header
    for i, x in enumerate(sectors):
        print(f"# Columns {boundaries[i]+1}-{boundaries[i+1]}: {list(x)}", file=f)
    # Build table
    table = np.column_stack(
        [
            np.asarray([pad(spec.get(q), l) for spec in specs])
            for q, l in zip(sectors, max_lengths)
        ]
    )
    np.savetxt(f, table, fmt="%16.13f")
print("Output data saved to", fname, file=stderr)
