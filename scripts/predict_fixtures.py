"""Write tests/fixtures/predict: small CT arrays and what the Python band reference feeds the network for them

    git show <predict-python-ref>:python/fibrs/predict/{contract,pipeline,villa}.py into REF/fibrs_ref/
    python scripts/predict_fixtures.py REF tests/fixtures/predict

REF holds the reference package (``fibrs_ref``: fibrs 0.2's Python ``fibrs.predict`` modules ``contract``,
``pipeline`` and ``villa``, at PR #3's head ddf69c7). Each case runs the reference ``TileFeed`` over a fixture CT and
records, per column, the batches it hands the network (tile origins and the sha256 of their bytes, as one
``(n, 128, 128, 128)`` uint8 array) and the air count; ``fibrs predict --dry-run`` must print the same. The tile feed
needs only numpy and zarr 2: a stand-in replaces the torch staging buffer

Needs numpy, zarr 2.18 and numcodecs 0.15
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numcodecs
import numpy as np
import zarr


class _Staged:
    def __init__(self, a):
        self._a = a

    def numpy(self):
        return self._a


class _FakeTorch:
    """The two names TileFeed._stage uses"""

    uint8 = np.uint8

    @staticmethod
    def empty(shape, dtype, pin_memory=False):
        return _Staged(np.empty(shape, dtype=dtype))


def blobs(rng, shape, n, region=None):
    """Mostly zero, with ``n`` boxes of noise starting inside ``region`` (the whole array by default), so chunks
    compress and the tiles away from the region are air"""
    a = np.zeros(shape, dtype=np.uint8)
    region = region or shape
    for _ in range(n):
        lo = [int(rng.integers(0, r)) for r in region]
        size = [int(rng.integers(4, max(5, s // 5))) for s in shape]
        sl = tuple(slice(lo_, lo_ + sz) for lo_, sz in zip(lo, size))
        a[sl] = rng.integers(1, 256, size=a[sl].shape, dtype=np.uint8)
    return a


def write_ct(path: Path, data: np.ndarray, chunks, compressor, nested: bool, drop_zero_chunks: bool):
    if path.exists():
        shutil.rmtree(path)
    z = zarr.open_array(str(path), mode="w", shape=data.shape, chunks=chunks, dtype="u1", compressor=compressor,
                        fill_value=0, dimension_separator="/" if nested else ".",
                        write_empty_chunks=not drop_zero_chunks)
    z[:] = data


CASES = [
    # name, shape, chunks, compressor, nested, drop all-zero chunks, blobs, dry-run arguments
    dict(name="band", shape=(40, 620, 430), region=(40, 250, 200), chunks=(16, 64, 64), compressor=numcodecs.Zstd(level=3), nested=True,
         drop=True, blobs=6, runs=[
             dict(mode="band", z0=4, z1=36, column=192, batch=3),
             dict(mode="band", z0=0, z1=40, column=96, batch=4),
             dict(mode="band", z0=10, z1=12, column=480, batch=4, only=[0]),
         ]),
    # stored chunks that are all zero still run in villa mode
    dict(name="villa", shape=(150, 420, 330), region=(150, 200, 260), chunks=(64, 64, 64),
         compressor=numcodecs.Blosc(cname="lz4", clevel=5, shuffle=numcodecs.Blosc.SHUFFLE), nested=False,
         drop=True, blobs=4, sparse=True, runs=[
             dict(mode="villa", z0=0, z1=150, column=192, batch=1),
             dict(mode="band", z0=20, z1=120, column=288, batch=16),
         ]),
    # every axis shorter than a tile: one tile, reflected on every side, more than once in z
    dict(name="tiny", shape=(6, 50, 70), chunks=(4, 32, 32), compressor=numcodecs.Zstd(level=1), nested=False,
         drop=True, blobs=3, runs=[
             dict(mode="band", z0=0, z1=6, column=480, batch=4),
             dict(mode="villa", z0=0, z1=6, column=96, batch=1),
         ]),
]


def feed(ref, ct_path: Path, run: dict) -> list[dict]:
    contract, pipeline, villa = ref
    ct = zarr.open_array(str(ct_path), mode="r")
    z0, z1 = run["z0"], run["z1"]
    nz, ny, nx = z1 - z0, ct.shape[1], ct.shape[2]
    cols = contract.Columns(ny, nx, run["column"])
    only = run.get("only")
    pz_all = contract.positions(nz)
    todo = [(i, col, cols.tiles(col, pz_all)) for i, col in enumerate(cols.cols) if only is None or i in only]
    runs = villa.VillaSelection(ct_path, ct.shape).tile_runs if run["mode"] == "villa" else None
    f = pipeline.TileFeed(_FakeTorch, str(ct_path), z0, z1, [(i, t) for i, _, t in todo], run["batch"], 3,
                          pin=False, runs=runs)
    out = []
    try:
        for i, col, tiles in todo:
            batches = []
            while True:
                item = f.get()
                assert item.col == i
                if isinstance(item, pipeline.ColumnEnd):
                    break
                x = np.ascontiguousarray(item.x.numpy()[:, 0])
                batches.append(dict(origins=[list(t) for t in item.tiles], sha256=hashlib.sha256(x.tobytes()).hexdigest()))
            out.append(dict(index=i, col=list(col), tiles=len(tiles), tiles_air=item.air, batches=batches))
    finally:
        f.close()
    return out


def main(ref_dir: str, out_dir: str) -> None:
    sys.path.insert(0, ref_dir)
    from fibrs_ref import contract, pipeline, villa

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    expected = {}
    rng = np.random.default_rng(20261118)
    for case in CASES:
        data = blobs(rng, case["shape"], case["blobs"], case.get("region"))
        ct_path = out / f"{case['name']}.zarr"
        write_ct(ct_path, data, case["chunks"], case["compressor"], case["nested"], case["drop"])
        if case.get("sparse"):
            # stored chunks that are all zero, and absent ones among stored neighbours
            z = zarr.open_array(str(ct_path), mode="r+", write_empty_chunks=True)
            z[0:64, 0:64, 192:256] = 0
            for name in ("2.0.0", "1.2.3"):
                (ct_path / name).unlink(missing_ok=True)
        for run in case["runs"]:
            key = f"{case['name']}:{run['mode']}:{run['z0']}-{run['z1']}:c{run['column']}:b{run['batch']}"
            if run.get("only"):
                key += ":only" + ",".join(map(str, run["only"]))
            expected[key] = dict(ct=ct_path.name, **run, columns=feed((contract, pipeline, villa), ct_path, run))
    (out / "expected.json").write_text(json.dumps(expected, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:])
