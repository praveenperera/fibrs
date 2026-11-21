# Benchmarks and parity

The measurements behind the speed and same-output claims in the [README](../README.md).

## Same output

fibrs's `.afv` files are compared with the original's table by table: every fibre row, point block,
spatial-index row and metadata value, except the per-run uuid.

- **0.4.0**, the published binary, matches the original with `--device cuda --threads 8` on two 512-plane bands:
  - PHerc1545, z 1280–1792: 82,646 fibres, 83,741 point blocks
  - PHerc0826, z 12032–12544: 149,680 fibres, 161,787 point blocks

  It also matched on both bands with `--device cpu --threads 4`, in four runs.
- **Earlier releases** were checked the same way. 0.1.0 was also checked in a fresh environment next to the
  original, on a 512³ crop, through both the drop-in and `python -m fibrs`.
- **GPU centrality.** On both bands, at `--threads 4` and `--threads 8`, the output was the same with
  `--device cuda` and `--device cpu`, and the same as the original's with either device.
- **CI** traces a small synthetic input (`tests/fixtures/synthetic`) with every build's wheel on Linux, macOS and
  Windows, through the drop-in and through `python -m fibrs`, and compares it with the original's output using
  `scripts/compare_afv.py`. The drop-in must also match with no seeds traced ahead (`FIBRS_DEBUG_WINDOW=1`) and
  with every seed traced ahead traced again (`FIBRS_DEBUG_RETRACE=1`).

## End to end, fibrs 0.4.0

Three 512-plane bands, each across the whole plane, from CT to `.afv`, on one RTX 5090 with an Intel Core Ultra 7
270K Plus (24 cores). All prediction is fp32 (float32 with TF32 convolutions, PyTorch's default and villa's
setting).

- **fibrs 0.4.0** (the release binary) runs `fibrs predict` in band mode, then traces with
  `--device cuda --threads 8`. Each time is the faster of two runs.
- **The public pipeline** is villa's `lasagna.qualzz_inference` (villa main `0bddf2757`), followed by
  `avf-spline-generator` 0.2.0 on the CPU with `--threads 4`, its fastest setting here. villa's times are estimated.

| Band | Tiles with data | villa prediction (est.) | Original tracer | Public pipeline | fibrs prediction | fibrs tracing | fibrs total | Speed-up |
|---|---|---|---|---|---|---|---|---|
| PHerc1545, z 1280–1792 | 8.4% | about 5,000 s | 1,613.8 s | about 6,614 s | 105.8 s | 11.0 s | 116.9 s | 56.6× |
| PHerc0826, z 12032–12544 | 15.2% | about 5,850 s | 4,615.7 s | about 10,466 s | 215.2 s | 25.0 s | 240.2 s | 43.6× |
| PHerc0358, z 2304–2816 | 33.2% | about 5,400 s | about 12,540 s (est.) | about 17,940 s | 399.6 s | 43.2 s | 442.9 s | 40.5× |
| All three | | | | about 35,020 s | | | 800.0 s | 43.8× |

By stage, prediction is 47.3×, 27.2× and 13.5× faster, and tracing 146×, 185× and 290× faster.

Notes:

- **Tiles with data** is the share of the band's 128³ tiles that hold CT data inside the mask. Band mode runs only
  those, while villa runs every tile.
- **villa's estimate.** On PHerc1545 villa ran all 37,446 tiles in 4,266 s, then was stopped during its final
  accumulator write, estimated at about 800 s. The other two bands are scaled from that rate (114 ms per tile),
  plus the same final write.
- **Machine load.** The original tracer's PHerc0358 run took 15,631.6 s under heavy other load (load average
  26.4). The table uses an estimate for a lighter load instead: its 32,739 s of CPU time over the 2.61 cores it got
  on the other bands. Against the measured time, that band is 47.5× faster. Other runs had load averages of 10–24
  for villa and the original, and 3–4 for fibrs. None is corrected for.
- **Same predictions.** Both tracers traced fibrs's predictions. Tracing villa's predictions could take a little
  more or less time.

**Without villa's disk writes.** villa writes its tile accumulators to disk as float32 as it runs, and has no
setting to avoid it. On this machine that took a large part of its time. Estimated from villa runs with the
accumulators in memory (53–54 ms per tile), the public pipeline would take about 3,600 s, 7,000 s and 14,700 s,
and fibrs would still be 30.9×, 29.1× and 33.2× faster (31.6× overall).

### A whole scroll (estimated)

| Scroll | Planes | fibrs 0.4.0 | villa and the original tracer |
|---|---|---|---|
| PHerc1545 | 20,961 | about 176–181 min | about 93–137 h |
| PHerc0826 | 16,920 | about 156–163 min | about 87–127 h |
| PHerc0358 | 14,744 | about 173–175 min | about 81–123 h |

That is roughly 28–49× faster per scroll.

- **fibrs.** Prediction was measured over the whole of PHerc1545 and PHerc0826 with fibrs 0.2.0, which predicts
  at about 0.4.0's speed (2.7 h and 2.4 h). PHerc0358's prediction and all three traces are scaled from the bands
  above by each scroll's tiles with data.
- **villa and the original.** villa's rate per tile (114–134 ms) times the scroll's tile count, plus the original
  tracer's time per tile with data on the three bands (0.51–0.94 s) times the scroll's tiles with data.
- Neither counts the download or joining the bands. Neither reference tool was run on a whole scroll. Tracing in
  overlapping 1024-plane windows instead of single bands would roughly double the tracing time: about 15–20 more
  minutes for fibrs and 40–80 more hours for the original.

## Tracer alone, fibrs 0.1.0 to 0.3.1

These runs time the tracer alone on the CPU, with `--threads 4` for both tools, on the same 24-core machine with
other work running.

| Run | Band | Tool | Wall | CPU time | Peak memory |
|---|---|---|---|---|---|
| 1 | PHerc0826, z 12032–12544 | original | 11,840 s (3.3 h) | about 11,940 s | 4.6 GiB |
| 1 | | fibrs, before 0.1.0 | 191 s | 516 s | 12.4 GiB |
| 2 | PHerc0826, z 12032–12544 | original | 4,686 s (1.3 h) | 12,296 s | 4.6 GiB |
| 2 | | fibrs 0.1.0 | 172 s | 472 s | 10.8 GiB |
| 3 | PHerc1545, z 1280–1792 | original | 1,621 s (27 min) | 4,149 s | 4.6 GiB |
| 3 | | fibrs 0.1.0 | 73.4 s | 191 s | 7.0 GiB |
| 4 | PHerc0358, z 2304–2816 | original | 15,632 s (4.3 h) | 32,739 s | 4.6 GiB |
| 4 | | fibrs `42c4710` | 332 s | 902 s | 14.0 GiB |
| 5 | PHerc0813, z 2304–2816 | original | about 9,176 s (2.5 h) | about 27,460 s | 4.7 GiB |
| 5 | | fibrs 0.2.0 | 202 s | 564 s | 11.8 GiB |

Over runs 2 to 4, fibrs is 22–47× faster on wall time and about 22–38× on CPU time. Runs 1 and 5 are left out of
the range because the original was interrupted and resumed, so its totals are partly estimated (about 62× and 45×).
In runs 2 to 5, both tools wrote the same fibres.

The original kept about 2.1–3.0 cores busy (about one in run 1), and fibrs about 2.6–2.8. The ratio moves with the
machine's load.

### Later tracer changes

On a dense 512 × 2048 × 2048 crop of PHerc0358 (z 1792–2304, y 2176–4224, x 2944–4992) at `--threads 8`, each
pair run back to back:

- **Decoding 8 planes at a time** (0.2.0): 104 s and a 3.5 GiB peak with 0.1.0, down to 50–52 s and 2.6 GiB, with
  the same `.afv`. Decoding whole chunks took 69–72 s.
- **Tracing seeds ahead** (0.3.1, with glibc malloc): from 47–55 s to 41–43 s, with the same `.afv` and peak.

On whole bands with `--device cpu` and glibc malloc, 0.3.1 took PHerc1545 from 0.3.0's 77 s to 58 s at
`--threads 4` (46 s to 43 s at `--threads 8`), and PHerc0826 from 174 s to 153 s (119 s to 106 s). Peak memory rose
by at most 0.3 GiB.

### Centrality on the GPU

Measured before the direction field moved to the GPU, with `--threads 8` on an RTX 5090 host. Each figure is the
mean of two runs, and all eight `.afv` files were identical.

| Band | Centrality and seeds, CPU | Same, GPU | Whole run, CPU | Whole run, GPU |
|---|---|---|---|---|
| PHerc1545, z 1280–1792 | 8.1 s | 1.8 s | 44.2 s | 38.0 s |
| PHerc0826, z 12032–12544 | 20.4 s | 5.0 s | 120 s | 105 s |

The original's `--device cuda` doesn't speed up centrality. On the same bands it took 52.6 s and 65.9 s, against
35.3 s and 59.3 s with `--device cpu`. Most of that time goes to host work and copying tiles to and from the GPU.

## Prediction speed

Against villa's `lasagna.qualzz_inference` (`0bddf2757`, PyTorch 2.14.1, CUDA 13.0, cuDNN 9.24) on one RTX 5090.
fibrs was a pre-release build of 0.2.0 (`e9adcc9`) with cuDNN 9.27.0 and cuBLAS 13.4.1.3.

The test crop is 726 tiles of PHerc0826 (z 12032–12544, y 4096–5120, x 1536–2560). Runs alternated (ABBA), with
6 timed pairs per precision after a warm-up. Times are whole-command wall times, including reads and writes.

| fibrs precision | villa, fp32 (median) | fibrs (median) | Speed-up, mean (95% CI) |
|---|---|---|---|
| fp32 (default) | 45.37 s | 26.77 s | 1.72× (1.66–1.78×) |
| bf16 (opt-in) | 45.35 s | 14.06 s | 3.19× (3.10–3.27×) |

On a 512 × 2048 × 2048 box of PHerc0358 with no all-zero tiles, one fp32 run each took 165.6 s for villa and
85.3 s for fibrs `42c4710` in band mode (1.94×), or 84.5 s with `--mode villa` (1.96×).

## Prediction parity

- **fp32**, `--mode villa` against villa on the PHerc0826 crop:
  - the 99.9th percentile of the probability differences over the foreground is 1 of 255 in each channel
  - the largest difference is 3 of 255
  - the most likely class agrees at 99.97% of voxels in `semantic` and 99.973% in `tracing`

  That is the same order as two runs of villa itself with `cudnn.benchmark` on.
- **bf16** against villa's fp32 on the same crop: the 99.9th percentile is 6 and 4 of 255, and the most likely class
  agrees at 99.55% and 99.62% of voxels.
- **bf16 against fp32**, traced with fibrs on 512-plane bands of PHerc0826 and PHerc0813: traced fibre length
  agreed to 99.2–99.4%, 97.0–97.8% of fibres matched, and the fibre counts were within 0.3%.
- **Band mode** matches the reference engine value for value on a full PHerc0826 band, at fp32 and bf16.
- **Libraries.** Switching from a system cuDNN 9.27.0 and cuBLAS 13.4 to the pip cuDNN 9.28.1 and cuBLAS 12.9 left
  the output unchanged on three band columns, at fp32 and bf16.

## Why the speed-up varies

- **Density.** How much of a region is papyrus rather than air varies between scrolls and along each one. Band mode
  skips tiles with no data, so on sparse bands it runs a fraction of villa's tiles.
- **Tracing time** depends on the band and the machine's load. The original took 4,686 s and 12,728 s in two
  uninterrupted runs on the same PHerc0826 band. Per fibre, it took about 0.027–0.035 s, against 0.6–0.75 ms for
  fibrs.
- **Estimates.** Where a reference run was interrupted or not run on the whole band, its time is estimated, and so is
  the ratio.
