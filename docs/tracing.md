# Tracing

## GPU

With a usable NVIDIA GPU, fibrs runs the direction field, centrality, seed tests and curve finishing on it.
Tracing itself always runs on the CPU.

You need Linux or Windows x86-64, compute capability 8.0+ and a CUDA 12 driver. No CUDA toolkit.

| Flag | Effect |
|---|---|
| `--device auto` | GPU if available, else CPU (default) |
| `--device cpu` | CPU only |
| `--device cuda` | GPU only, fail if it can't |
| `--cuda-device N` | pick a GPU |

With `auto`, fibrs falls back to the CPU if the GPU fails mid-run, for example by running out of memory.

The GPU gives the same direction field and finishing as the CPU. Centrality rounds slightly differently, the same
way the original's `--device cuda` does. A resumed run keeps the rounding it started with.

### GPU slab decoding

Off by default. To turn it on:

```sh
pip install "fibrs[gpu-zstd]"
export FIBRS_NVCOMP=on
```

Only use it on files you trust. A corrupt file can give wrong output or crash, where the CPU would report an
error.

## Threads

`--threads N` sets the tracing threads. Output is the same at any thread count.

## Tiled tracing

```sh
fibrs ... --tiles-yx 2 2 --tile-halo 256
```

Traces a window as a grid of tiles in Y and X, one at a time, so peak memory is that of the densest tile.

- Each tile also reads `--tile-halo` voxels (default 256) of its neighbours.
- Tile edges are multiples of 256 voxels for the usual prediction zarrs.
- Fibres are cut at tile edges and rejoined when the ends are within 1.5 voxels.
- Output can differ slightly from an untiled run near tile edges.
- `--resume` skips finished tiles.

## Joining Z bands

`fibrs join` cuts each band's fibres to its core planes and rejoins them across band edges, the same way tiles
are joined.

- Bands can be listed in any order, but their cores must not overlap.
- `--radius` sets how close two ends must be to join (default 1.5 voxels).
- `--stats` writes per-family counts.
- On PHerc1545's 42 bands, it takes 13 minutes and 5.6 GiB, against 26 minutes for `join.py`, with the same
  output.

## Inputs

- `.npy` files, and Zarr v2 arrays or groups
- uint8, float16, float32 or float64
- Blosc, Zstd, Zlib, GZip or LZ4 compression

Zarr filters and other compressors are not supported.

## Memory

fibrs uses more memory than the original: 9.0 GiB against 4.6 GiB on a 512-plane band of PHerc0826. A smaller
`--native-cache-mib` or `--field-cache-mib` lowers that, at some cost in speed.

On Linux, fibrs turns off transparent huge pages for the tracer. On a dense window that cut tracing from
114–134 s to about 103 s. `MIMALLOC_ALLOW_THP=1` turns them back on.

`FIBRS_DEBUG_MEM=1` logs memory use per stage.

## Differences from the original

- `--device` defaults to `auto` instead of `cpu`. Pass `--device cpu` to match the original's default output.
- fibrs and the original can't resume each other's work directories.
- An interrupted family restarts from its first seed.
- `--storage` and `--field-cache-mib` are accepted but do nothing.
- Help and error messages are worded differently.
