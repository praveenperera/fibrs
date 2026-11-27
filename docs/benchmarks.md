# Benchmarks and parity checks

The measurements behind the README's figures, with their method, and the figures from earlier releases.

## Same output

The published 0.4.0 binary, the one in the release archive and the PyPI wheel, writes the original's `.afv` with
`--device cuda --threads 8` on a 512-plane band of PHerc1545 (z 1280–1792: 82,646 fibres, 83,741 point blocks) and
on a 512-plane band of PHerc0826 (z 12032–12544: 149,680 fibres, 161,787 point blocks). The two files were compared
table by table: every fibre row, point block and spatial-index row, and every metadata value except the per-run
uuid. With `--device cpu --threads 4`, 0.4.0 also wrote the original's `.afv` on both bands, in four runs.

Earlier releases were checked the same way. On the PHerc0826 band, the port's `.afv` is identical to the original's:
149,680 fibres, 161,787 point blocks and 161,787 spatial-index rows, plus the metadata and schema. The 0.1.0 build
was also checked on a 512³ crop of the band: the wheel and the drop-in were installed into a fresh environment
beside the original, and the crop was traced through `avf-spline-generator` and through `python -m fibrs`. Both
outputs were identical to the original's `.afv`.

Centrality on the GPU gives the same `.afv`. On the PHerc0826 band and on a 512-plane band of PHerc1545
(z 1280–1792), with `--threads 4` and `--threads 8`, the port's output was identical with `--device cuda` and
`--device cpu`, and identical to the original's with `--device cpu` and with `--device cuda`. The original's CUDA
kernels round some multiply-adds as one operation where its CPU code rounds twice; the port reproduces each, and on
these bands both roundings give the same `.afv`.

CI traces a small synthetic input (`tests/fixtures/synthetic`) with every build's wheel on Linux and macOS,
through the drop-in and through `python -m fibrs`. Both must write what the original wrote for it, compared by
`scripts/compare_afv.py`: every metadata value except the per-run uuid, the schema and every table row. The drop-in
must also write it with no seed traced ahead of the one being committed (`FIBRS_DEBUG_WINDOW=1`) and with every seed
traced ahead traced again (`FIBRS_DEBUG_RETRACE=1`).

## Speed

### End to end, fibrs 0.4.0

From CT to `.afv` on three 512-plane bands, each across the whole plane, on one RTX 5090 and an Intel Core Ultra 7
270K Plus (24 cores). All prediction is fp32: float32 tensors with TF32 tensor-core convolutions, PyTorch's default
and the precision villa's runner uses. fibrs 0.4.0, the published release binary, runs `fibrs predict` at fp32 in
band mode, then traces with `--device cuda --threads 8`. Both fibrs times are measured; each is the faster of two
runs on its band. The public pipeline is villa's `lasagna.qualzz_inference` at fp32 (villa main `0bddf2757`), run
as published, followed by `avf-spline-generator` 0.2.0 on the CPU with `--threads 4`, its fastest setting here
(`--threads 8` was slower on both bands tried). The original tracer's times are measured; villa's are estimated.

| Band | Tiles with data | villa prediction, fp32 (estimated) | Original tracer, `--threads 4` | Public pipeline | fibrs 0.4.0 prediction, fp32 | fibrs 0.4.0 tracing | fibrs 0.4.0 total | Speed-up, fp32 |
|---|---|---|---|---|---|---|---|---|
| PHerc1545, z 1280–1792 | 8.4% | about 5,000 s | 1,613.8 s | about 6,614 s | 105.8 s | 11.0 s | 116.9 s | 56.6× |
| PHerc0826, z 12032–12544 | 15.2% | about 5,850 s | 4,615.7 s | about 10,466 s | 215.2 s | 25.0 s | 240.2 s | 43.6× |
| PHerc0358, z 2304–2816 | 33.2% | about 5,400 s | about 12,540 s (estimated) | about 17,940 s | 399.6 s | 43.2 s | 442.9 s | 40.5× |
| All three | | | | about 35,020 s | | | 800.0 s | 43.8× |

- **Tiles with data** is the share of the band's 128³ tiles with CT data inside the mask, the tiles band mode runs.
  villa runs every tile.
- **The machine's load.** The original tracer's runs had a one-minute load average of 10.9 (PHerc1545) and 14.9
  (PHerc0826), against about 3–4 during fibrs's runs. It kept about 2.6 cores busy, so the load probably cost it
  little, but it is not corrected for. Its PHerc0358 run took 15,631.6 s while the machine carried heavy other work
  (load average 26.4, up to 79.9). The table uses an estimate for a lighter load instead: its 32,739 s of CPU time
  over the 2.61 cores its other two runs got. Against the measured time, that band is 47.5× faster.
- **villa's estimate.** On PHerc1545 villa ran all 37,446 tiles in 4,266 s and was stopped during its final
  accumulator write, estimated at about 800 s from the 790 s write it made mid-band. PHerc0826 (44,376 tiles) and
  PHerc0358 (about 40,300) are scaled from that rate, 114 ms per tile, plus the same final write. That PHerc1545 run
  had a one-minute load average of 13–24, which may have slowed it, and it is not corrected for either.
- Both tracers traced fibrs's predictions of each band. A trace of villa's predictions could take somewhat longer
  or shorter, as the tracer's time depends on what it reads.

By stage, prediction is 47.3×, 27.2× and 13.5× faster (fp32 for both tools; villa's side estimated), and tracing
146×, 185× and 290× faster (362× against the measured PHerc0358 time; fibrs on the GPU at `--threads 8`, the original
on the CPU at `--threads 4`).

villa writes its tile accumulators to disk as float32 arrays as it runs, and on this machine that write took a large
part of its time; its runner has no setting for a smaller accumulator type. Without the write, estimated from villa
runs on 1,536-tile boxes of the same bands with the accumulators in memory (53–54 ms per tile), the public pipeline
would take about 3,600 s, 7,000 s and 14,700 s on the three bands, and fibrs 0.4.0 would still be 30.9×, 29.1× and
33.2× faster (31.6× over all three, fp32). The older villa figures under [Predictions](predict.md) and
[Why the speed-up varies](#why-the-speed-up-varies) come from a crop and a box small enough to keep the accumulators
in memory, so they leave this write out; they agree with these figures (2,231 s for villa's PHerc0358 band
prediction there, about 2,150 s here).

A whole scroll, end to end on one RTX 5090, estimated (not measured), with prediction at fp32:

| Scroll | Planes | fibrs 0.4.0 | villa and the original tracer |
|---|---|---|---|
| PHerc1545 | 20,961 | about 176–181 min | about 93–137 h |
| PHerc0826 | 16,920 | about 156–163 min | about 87–127 h |
| PHerc0358 | 14,744 | about 173–175 min | about 81–123 h |

That is roughly 28–49× faster per scroll. For fibrs, prediction was run over the whole of PHerc1545 and PHerc0826
with fibrs 0.2.0, whose prediction runs at about 0.4.0's speed (2.7 h and 2.4 h, fp32, measured). PHerc0358's
prediction and all three traces are scaled from the bands above by each scroll's count of tiles with data, with each
scroll traced as its 512-plane bands. For villa and the original tracer, villa's rate per tile above (114–134 ms, the
upper figure including its final write) is multiplied by the scroll's tile count, and the original tracer's time
per tile with data on the three bands (0.51–0.94 s) by the scroll's tiles with data; the range spans those rates.
Neither counts the download or joining the bands' outputs into one file, and neither reference tool was run on a
whole scroll. Tracing in overlapping 1024-plane windows instead of single bands would roughly double the tracing for
both tools, adding about 15–20 minutes for fibrs and about 40–80 hours for the original.

### Tracer alone, fibrs 0.1.0 to 0.3.1

These runs, with earlier releases, time the tracer alone on the CPU.

Five runs on an Intel Core Ultra 7 270K Plus (24 cores), with `--threads 4` for both tools and other work
running on the machine each time. Runs 1 and 2 trace the same 512 × 8169 × 8169 band of PHerc0826, run 3 a
512-plane band of PHerc1545 (z 1280–1792), run 4 a 512 × 7783 × 7783 band of PHerc0358 (z 2304–2816) and run 5
a 512-plane band of PHerc0813 (z 2304–2816):

| Run | Tool | Wall | CPU time | Peak memory |
|---|---|---|---|---|
| 1 | `avf-spline-generator` 0.2.0 | 11,840 s (3.3 h) | about 11,940 s | 4.6 GiB |
| 1 | `fibrs`, built before 0.1.0 | 191 s | 516 s | 12.4 GiB |
| 2 | `avf-spline-generator` 0.2.0 | 4,686 s (1.3 h) | 12,296 s | 4.6 GiB |
| 2 | `fibrs` 0.1.0 | 172 s | 472 s | 10.8 GiB |
| 3 | `avf-spline-generator` 0.2.0 | 1,621 s (27 min) | 4,149 s | 4.6 GiB |
| 3 | `fibrs` 0.1.0 | 73.4 s | 191 s | 7.0 GiB |
| 4 | `avf-spline-generator` 0.2.0 | 15,632 s (4.3 h) | 32,739 s | 4.6 GiB |
| 4 | `fibrs` `42c4710` | 332 s | 902 s | 14.0 GiB |
| 5 | `avf-spline-generator` 0.2.0 | about 9,176 s (2.5 h) | about 27,460 s | 4.7 GiB |
| 5 | `fibrs` 0.2.0 | 202 s | 564 s | 11.8 GiB |
| Speedup, runs 2 to 4 | | **22×–47×** | **about 22×–38×** (38× with the 853 s repeat of run 4) | |

- Run 1: the one-minute load average was 17.5–18.8. The original's run was interrupted by a reboot and
  finished with `--resume`. Its wall time is the sum of the two parts, and the first part's CPU time is
  estimated from the second part's CPU-to-wall ratio (1.01). An earlier uninterrupted run of the original on
  the same band took 12,728 s. Run 1's total is estimated, so it sets neither end of the range.
- Run 2: the one-minute load average averaged 13.9 during the original's run and 8.8 during fibrs's. The two
  `.afv` files held the same 149,680 fibres (compared logically, as each holds its own run uuid).
- Run 3: the one-minute load average was 10.0 during the original's run and 7.4 during fibrs's. This band is
  sparser. The two `.afv` files held the same 82,646 fibres and 83,741 point blocks.
- Run 4: the two runs were not at the same time. This is the densest band measured. The two `.afv` files
  held the same 448,286 fibres and 460,560 point blocks.
- Run 5: the two runs were not at the same time. The original's first attempt logged progress for 5,074 s before it
  was stopped and finished with `--resume` in 4,102 s. Its wall time is the sum of the two parts, and the first
  part's CPU time is estimated from the second part's CPU-to-wall ratio (2.99). The resumed part loaded the
  direction field and the V centrality instead of recomputing them; it restored 103,340 batches 105 s after it
  started and was tracing again after 126 s. The ratios are about 45× on wall time and about 49× on CPU time; as
  the total is estimated, the run sets neither end of the range. The two `.afv` files held the same 337,735 fibres
  and 344,279 point blocks. Run 5's memory is the peak RSS of both tools.

The original kept about 2.1–3.0 cores busy in runs 2 to 5 and about one in run 1; the CPU times of runs 1 and 5 are
partly estimated. The port kept about 2.6–2.8. The ratio varies with the run and with the machine's load: the
ranges are the lowest and highest of the uninterrupted runs, 2 to 4. The CPU range also covers a second run of the
same build on the run 4 band, which used 853 s of CPU time (38×).

On dense bands, fibrs's speed also depends on how it reads the predictions and allocates memory. Take a
512 × 2048 × 2048 crop of PHerc0358's densest area (z 1792–2304, y 2176–4224, x 2944–4992) with `--threads 8` and
`--field-cache-mib 224`, each pair of builds run back to back on the same busy machine. A pre-release build of
0.2.0 that decodes the predictions 8 planes at a time took 50–52 s with a 2.6 GiB peak and wrote the same `.afv` as
0.1.0, which took 104 s with a 3.5 GiB peak. One that decoded whole chunks took 69–72 s, and one that also gave
each decoded chunk its own mapping spent 40% of its CPU time in the kernel.

Tracing seeds ahead on idle threads (0.3.1, measured with glibc malloc, before mimalloc became the allocator) takes
that crop at `--threads 8` from 0.2.0's 47–55 s to 41–43 s, in two pairs of runs pinned to the same eight cores on a busy machine: tracing V from 12.0–15.4 s to 8.5–9.9 s and H
from 8.7–10.2 s to 6.2–6.4 s. The `.afv` is the same, and the peak is 2.6 GiB for both.

On whole bands with `--device cpu` and glibc malloc, each pair of builds run back to back on the same busy machine, it took
PHerc1545 from 0.3.0's 77 s to 58 s at `--threads 4` and from 46 s to 43 s at `--threads 8`, and PHerc0826 from
174 s to 153 s and from 119 s to 106 s. The `.afv` is the same, and the peak is at most 0.3 GiB higher (5.6 GiB
for PHerc1545, 9.1–9.2 GiB for PHerc0826).

### Centrality on the GPU

Measured before the direction field moved to the GPU, with `--threads 8` on an RTX 5090 host, each band ran four
times back to back, CPU, GPU, GPU, CPU, with other work running on the machine:

| Band | Centrality and seeds, CPU | Same, GPU | Whole run, CPU | Whole run, GPU |
|---|---|---|---|---|
| PHerc1545, z 1280–1792 | 8.1 s | 1.8 s | 44.2 s | 38.0 s |
| PHerc0826, z 12032–12544 | 20.4 s | 5.0 s | 120 s | 105 s |

Each figure is the mean of its two runs, and all eight `.afv` files were identical. The direction field (20–51 s
here) and tracing ran on the CPU in these runs, so the whole run gains less, and its time moves with the machine's load
(a mean one-minute load of 8–13 across these runs).

The original's `--device cuda` doesn't make centrality faster. On the same bands with `--threads 8` its centrality
took 52.6 s and 65.9 s with `--device cuda`, against 35.3 s and 59.3 s with `--device cpu`. Its kernels ran for
2.2 s and 4.7 s of that; the rest went to host work and to copying each 128³ tile and its halo to the GPU and back.

## Why the speed-up varies

Each speed-up above depends on the data, so each is quoted as the range of the measured points where there is
more than one:

- **Density.** The share of a region that is papyrus rather than air changes from scroll to scroll and along
  each scroll. The table gives it as the share of 128³ tiles that `fibrs predict --dry-run` would run (band mode,
  tiles whose input isn't all zero), measured the same way for every row.
- **Prediction skips air.** villa main runs every tile; band mode skips tiles whose input is all zero, so on a
  sparse band it runs a fraction of villa's tiles and on a dense one about as many. `--mode villa` doesn't skip
  all-zero tiles, and on the PHerc0826 crop it ran 864 tiles where villa ran 726, since it runs a tile once for
  each of its columns that the tile's core overlaps. Every measured prediction time is the whole command's wall
  time, reading the same CT crop and writing arrays of the same shapes and chunk sizes. villa's prediction was not
  run on the whole PHerc0358 band; its time there is estimated, so that end-to-end ratio is an estimate too.
- **Tracing varies with the run.** The tracer ratio varies with the run and with the machine's load, so it is given
  as the lowest to highest of the uninterrupted runs; the two runs whose original was interrupted and resumed have
  estimated totals and set neither end. The original's time per fibre also differs between bands and between runs:
  about 0.035 s on the dense PHerc0358 band (run 4) and about 0.027 s on PHerc0813 (run 5, estimated), and its two
  uninterrupted runs on the same PHerc0826 band took 4,686 s and 12,728 s. fibrs took about 0.6–0.75 ms per fibre
  on runs 4 and 5. Pre-release builds of 0.2.0 traced dense data slower than 0.1.0; a later pre-release build is
  faster than 0.1.0 there (the last row).

| Stage | Scroll | Region (z / y / x) | Tiles run | Original | fibrs | Ratio | Precision | Measured as |
|---|---|---|---|---|---|---|---|---|
| Tracing | PHerc0826 | z 12032–12544, whole plane | 15.2% | 11,840 s | 191 s | about 62× | CPU | one run each; the original was interrupted and resumed; total estimated (run 1; fibrs built before 0.1.0) |
| Tracing | PHerc0826 | z 12032–12544, whole plane | 15.2% | 4,686 s | 172 s | 27× | CPU | one run each (run 2; fibrs 0.1.0) |
| Tracing | PHerc1545 | z 1280–1792, whole plane | 8.4% | 1,621 s | 73.4 s | 22× | CPU | one run each (run 3; fibrs 0.1.0) |
| Tracing | PHerc0358 | z 2304–2816, whole plane | 33.2% | 15,632 s | 332 s | 47× | CPU | one run each, not at the same time (run 4; fibrs `42c4710`, before 0.2.0's chunk-cache change) |
| Tracing | PHerc0813 | z 2304–2816, whole plane | 25.4% | 9,176 s | 202 s | about 45× | CPU | one run each, not at the same time; the original was interrupted and resumed; total estimated (run 5; fibrs 0.2.0) |
| Prediction, `--mode villa` | PHerc0826 | z 12032–12544 / y 4096–5120 / x 1536–2560 | 100% | 45.37 s | 26.77 s | 1.72× (1.66–1.78×) | fp32 | ABBA, 6 pairs: medians, mean ratio with 95% CI |
| Prediction, `--mode villa` | PHerc0826 | same crop | 100% | 45.35 s | 14.06 s | 3.19× (3.10–3.27×) | bf16 (opt-in) | ABBA, 6 pairs: medians, mean ratio with 95% CI |
| Prediction, band mode | PHerc0358 | z 2304–2816 / y 2176–4224 / x 2944–4992 | 100% | 165.6 s | 85.3 s | 1.94× | fp32 | one run each, not at the same time; fibrs `42c4710` |
| Prediction, `--mode villa` | PHerc0358 | same box | 100% | 165.6 s | 84.5 s | 1.96× | fp32 | one run each, not at the same time, against the same villa run; fibrs `42c4710` |
| End to end, prediction in band mode | PHerc0826 | z 12032–12544 / y 4096–5120 / x 1536–2560 | 100% | 387.5 s | 41.7 s | 9.3× | fp32 | one run of each stage |
| End to end, prediction in band mode | PHerc0358 | z 2304–2816, whole plane | 33.2% | 17,863 s, prediction estimated | 739 s | 24×, estimated | fp32 | villa's prediction time extrapolated from the box above; the rest one run of each stage, fibrs `e9adcc9` predicting and `42c4710` tracing; both tracers read fibrs's prediction |
| Tracing, fibrs 0.1.0 → a pre-release build of 0.2.0 | PHerc0358 | z 1792–2304 / y 2176–4224 / x 2944–4992 | 100% | 104 s (0.1.0) | 50–52 s (pre-release 0.2.0) | 2.0–2.1× | CPU | back to back on one machine, `--threads 8`; not in the tracing range |

"Original" is `avf-spline-generator` 0.2.0 for tracing and villa's `lasagna.qualzz_inference` (main `0bddf2757`,
PyTorch fp32) for prediction; end to end is the two stages added, on the PHerc0826 crop from a single run of each
(prediction 45.4 s and 27.0 s, `fibrs predict` in band mode, not the ABBA medians). On the PHerc0358 band villa's
prediction was not run: its 2,231 s is the box run's 5.4 s outside inference plus its 160.2 s of inference scaled
from the box's 2,904 tiles to the band's 40,344, as villa main runs every tile. fibrs's prediction (407.2 s) and
both tracers (15,632 s and 332 s) were measured. Both tracers read the same saved fibrs prediction of the band, and
each stage was timed in its own run. Tracing ran with `--threads 4` for both tools unless the row says otherwise.
