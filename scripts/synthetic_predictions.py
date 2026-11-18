"""Write the synthetic AFV prediction zarr in tests/fixtures/synthetic: smooth V fibres along z and wavy H fibres
along x, as uint8 probabilities in `semantic` (2, Z, Y, X) and their 2x mean-pooled `tracing` (2, Z/2, Y/2, X/2),
chunked and compressed like a real prediction

The fixture's reference.afv is what avf-spline-generator 0.2.0 writes for it (see tests/fixtures/synthetic/README.md).
Needs numpy, zarr 2 and numcodecs

Usage: synthetic_predictions.py OUT.zarr
"""

import sys

import numpy as np
import zarr
from numcodecs import Blosc

Z, Y, X = 128, 192, 192


def tubes() -> np.ndarray:
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float32) for n in (Z, Y, X)), indexing="ij")
    v = np.zeros((Z, Y, X), np.float32)
    h = np.zeros((Z, Y, X), np.float32)
    for cy in range(16, Y, 24):
        for cx in range(20, X, 28):
            yy = cy + 3.0 * np.sin(z / 17.0 + cx)
            v = np.maximum(v, np.exp(-((y - yy) ** 2 + (x - cx) ** 2) / 8.0))
    for cz in range(12, Z, 20):
        for cy in range(10, Y, 22):
            zz = cz + 2.5 * np.sin(x / 23.0 + cy)
            h = np.maximum(h, np.exp(-((z - zz) ** 2 + (y - cy) ** 2) / 8.0))
    return np.stack([v, h]).astype(np.float32)


def main() -> None:
    p = tubes()
    semantic = np.rint(p * 255).astype(np.uint8)
    pooled = p.reshape(2, Z // 2, 2, Y // 2, 2, X // 2, 2).mean(axis=(2, 4, 6), dtype=np.float32)
    tracing = np.rint(pooled * 255).astype(np.uint8)
    group = zarr.open_group(sys.argv[1], mode="w")
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    group.create_dataset("semantic", data=semantic, chunks=(2, 64, 64, 64), compressor=compressor)
    group.create_dataset("tracing", data=tracing, chunks=(2, 64, 64, 64), compressor=compressor)


if __name__ == "__main__":
    main()
