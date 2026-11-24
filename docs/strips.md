# fibrs strips

Runs villa's sheet-strip inference (`fibers/sheet_strip`, checkpoint
[scrollprize/fiber_strip_small_v1](https://huggingface.co/scrollprize/fiber_strip_small_v1) @ `7bd163a`) over an
AFV's fibres, natively on an NVIDIA GPU, without PyTorch. Each fibre gets a sheet strip: a tifxyz directory (x,
y, z and mask TIFFs and `meta.json`), `trace.json`, `expected_error.tif` and one line in the run's index, in the
same layout as the Python runner that drives villa's model.

## How it works

Per fibre, the CPU plans every decision along the fibre (its frame, the CT seed heading and the frame model's
request). The GPU runs the frame model on its crops. The CPU turns that into each decision's final frame. The GPU
then crops the CT, renders the tube and runs the strip model, and the CPU stitches the decisions into a strip and
writes it.

Decisions are batched across fibres: 16 CPU workers keep fibres in flight, and the GPU takes 64 decisions per
launch. The kernels are Rust, built with [cuda-oxide](https://github.com/NVIDIA/cuda-rust) and committed as PTX
(`ptx/strips.sm80.ptx`). cuDNN runs the convolutions where it is faster.

## Speed

One RTX 5090, decisions per second end to end:

| Runner | dec/s |
|---|---|
| villa's PyTorch runner, 12 processes at batch 4 | about 254 |
| `fibrs strips run`, whole PHerc0211 scroll (495,179 fibres) | 2,095 |

That is about 8× faster. Each release since then is checked with a paired A/B on 1,152 warm PHerc0211 fibres. The
batch-invariant runner below is 1.042× (95% CI 1.030–1.054) faster than the whole-scroll build.

## Quality

The runner's strips were compared with villa's PyTorch runner (bf16, batch 4) on 200 PHerc0211 fibres and one
PHercParis4 fibre:

| Measure | PyTorch | fibrs |
|---|---|---|
| Mean decision error (voxels) | 1.66663 | 1.66659 |
| Mean strip error | 1.34960 | 1.34960 |
| p95 strip error | 4.19782 | 4.19639 |
| Kept strip nodes | 2,021,075 | 2,021,075 |
| PHercParis4 mean / p99 error | 0.011095 / 0.062578 | 0.011039 / 0.062280 |

For every error measure, the paired bootstrap 95% CI of fibrs minus PyTorch includes zero or lies below it. The
output files and fields match. As a negative control, the strip model's weights perturbed by 5% fail the same
comparison.

The strips aren't bitwise equal to PyTorch's, since the bf16 arithmetic runs in a different order. A run's own
output doesn't depend on the batch size or the number of workers, though. Every cuDNN convolution runs on fixed
groups of 16 samples, so the same fibres give byte-identical strips at `--batch 40 --workers 8` and at
`--batch 64 --workers 16`.

## Usage

```sh
fibrs strips run \
  --afv PHerc0211_afv_v2.afv \
  --out strips/PHerc0211 \
  --weights fiber_strip_small_v1.safetensors \
  --ct s3://vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr \
  --cache-dir ctcache \
  --umbilicus PHerc0211_umbilicus.json \
  --voxel-size-um 9.362
```

The run selects every fibre of at least `--min-length-mm` (default 2.5) in Morton order, and resumes from the
index: run it again with the same `--out` and it takes up the fibres not yet written. An `--out` begun under
another runner tag (the `runner` field of `--provenance`), or with a different AFV, CT, checkpoint or decision
setting, is refused, so one index never mixes strips made differently.

| Option | Meaning |
|---|---|
| `--afv` | the fibre set |
| `--out` | output directory: `strips/`, `index/`, `logs/`, `run.json` |
| `--weights` | the checkpoint's tensors as one float32 safetensors file; its source hash is checked |
| `--ct`, `--ct-level` | the CT zarr and level |
| `--cache-dir` | vc_reader's disk cache holding the CT |
| `--umbilicus` | the scroll's umbilicus JSON |
| `--ids`, `--ids-file`, `--exclude-ids-file` | run only these fibres, or skip some |
| `--shard K N` | this shard of N contiguous shards |
| `--batch` | decisions per GPU launch (default 64) |
| `--workers` | CPU workers, fibres in flight (default 16) |
| `--gpu-chunks` | CT chunks in the GPU cache, 2 MiB each (default 4096) |
| `--device` | CUDA device ordinal |
| `--limit` | at most this many fibres |
| `--provenance` | JSON recorded in `run.json`; its `runner` tags every index line |
| `--no-write` | timing only: run and stitch, write no strips and no index lines |

### Missing CT chunks

The runner reads the CT only from the disk cache, never from the network. A GPU launch that needs a chunk the
cache lacks fails before it uploads anything, and every fibre in that launch is deferred rather than written. The
run finishes the other fibres, lists the missing chunks in `<out>/logs/uncached_chunks.txt` and exits with code
75. Fetch those chunks into the cache and run again; the
rerun takes up the deferred fibres.

## Requirements

- Linux x86-64 (Windows builds, but hasn't been run on a GPU)
- NVIDIA GPU, compute capability 8.0+
- CUDA 12+ driver, cuDNN 9
- the weights export and a CT disk cache, below

## Known limits

- The weights export, a float32 safetensors file of the checkpoint with its source hashes and configs as
  metadata, is made by a script that isn't in this repository yet.
- The CT has to be fetched into the disk cache before a run, by a tool that isn't in this repository either. A
  run fetches nothing itself.
- The runner supports this one checkpoint, and refuses weights exported from any other.
- One GPU per process. Use `--shard` to split a scroll across GPUs.
- It has only been run on an RTX 5090. Other compute-capability 8.0+ GPUs should work but are untested.
