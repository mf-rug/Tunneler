"""Standalone worker for the parallel Wavefront-OBJ writer (see _write_obj_parallel in
Tunneler_LoadMenu_tk_con.py). Formatting millions of floats to ASCII is CPU-bound and
GIL-held, so the big-tunnel sphere-mesh write is split across several of these worker
processes.

It is deliberately a *tiny, numpy-only script*, run via subprocess (never imported by the
plugin): a normal multiprocessing.Pool on macOS uses 'spawn', which re-executes the
parent's __main__ module -- here that would re-run the whole plugin (heavy imports, the
env check, the GUI loop) in every worker. Running an independent script sidesteps that
entirely.

Usage (all args positional):
    python Tunneler_meshwrite.py <npy_path> <start> <end> <fmt_b64> <out_path>

Loads rows [start:end] of the array memory-mapped from <npy_path>, formats each row with
the printf-style row format (base64-encoded to survive the shell), and writes the bytes to
<out_path>. The parent concatenates the part files in order.
"""
import sys
import base64
import numpy as np


def main():
    npy_path, start, end, fmt_b64, out_path = (
        sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5])
    rowfmt = base64.b64decode(fmt_b64).decode()
    blk = np.ascontiguousarray(np.load(npy_path, mmap_mode='r')[start:end])
    with open(out_path, 'wb') as f:
        f.write(((rowfmt * len(blk)) % tuple(blk.ravel().tolist())).encode())


if __name__ == '__main__':
    main()
