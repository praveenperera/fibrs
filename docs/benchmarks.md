# Benchmarks

## Same output

fibrs's `.afv` matches the original's in every table and metadata value, except the per-run uuid. Checked on:

| Band | Fibres | Point blocks |
|---|---|---|
| PHerc1545, z 1280–1792 | 82,646 | 83,741 |
| PHerc0826, z 12032–12544 | 149,680 | 161,787 |

Checked with the 0.4.0 release binary, on both `--device cuda` and `--device cpu`.

CI also traces a small synthetic input on Linux, macOS and Windows and compares it with the original's output.

## End to end

CT to `.afv`, one RTX 5090, Intel Core Ultra 7 270K Plus, fp32 throughout. fibrs 0.4.0 against villa's
prediction followed by the original tracer at `--threads 4`, its fastest setting.

| Band | Public pipeline | fibrs 0.4.0 | Speed-up |
|---|---|---|---|
| PHerc1545, z 1280–1792 | about 6,614 s | 116.9 s | 57× |
| PHerc0826, z 12032–12544 | about 10,466 s | 240.2 s | 44× |
| PHerc0358, z 2304–2816 | about 17,940 s | 442.9 s | 41× |
| All three | about 35,020 s | 800.0 s | 44× |

villa's times are estimated from a full run on PHerc1545. The original's PHerc0358 time is estimated because the
machine was heavily loaded during that run.

villa writes its accumulators to disk as it runs, which is slow. Without that, fibrs is about 32× faster.

### By stage

| Band | villa prediction | fibrs prediction | Original tracer | fibrs tracing |
|---|---|---|---|---|
| PHerc1545 | about 5,000 s | 105.8 s | 1,613.8 s | 11.0 s |
| PHerc0826 | about 5,850 s | 215.2 s | 4,615.7 s | 25.0 s |
| PHerc0358 | about 5,400 s | 399.6 s | about 12,540 s | 43.2 s |

### Whole scrolls (estimated)

| Scroll | Planes | fibrs 0.4.0 | villa + original |
|---|---|---|---|
| PHerc1545 | 20,961 | about 3 h | about 93–137 h |
| PHerc0826 | 16,920 | about 2.7 h | about 87–127 h |
| PHerc0358 | 14,744 | about 2.9 h | about 81–123 h |

Scaled from the bands above. fibrs's prediction time was measured on whole scrolls for PHerc1545 and PHerc0826.

## Tracer only, CPU

Earlier releases, both tools at `--threads 4`, on a busy 24-core machine:

| Band | Original | fibrs | Speed-up |
|---|---|---|---|
| PHerc0826, z 12032–12544 | 4,686 s | 172 s (0.1.0) | 27× |
| PHerc1545, z 1280–1792 | 1,621 s | 73.4 s (0.1.0) | 22× |
| PHerc0358, z 2304–2816 | 15,632 s | 332 s (`42c4710`) | 47× |
| PHerc0813, z 2304–2816 | about 9,176 s | 202 s (0.2.0) | about 45× |

Both tools wrote the same fibres. The PHerc0813 time is estimated because the original's run was interrupted and
resumed.

## Prediction

Against villa on a 726-tile crop of PHerc0826, alternating runs, 6 pairs each:

| Precision | villa | fibrs | Speed-up |
|---|---|---|---|
| fp32 | 45.37 s | 26.77 s | 1.72× |
| bf16 | 45.35 s | 14.06 s | 3.19× |

On a denser PHerc0358 box, fibrs was 1.94× faster at fp32.

### Accuracy against villa, fp32

- 99.9% of foreground voxels are within 1/255 of villa's
- the largest difference is 3/255
- the most likely class agrees at 99.97% of voxels

That's about the same as two runs of villa against each other.

### bf16 against fp32

Traced on two bands, total fibre length agreed to 99.2–99.4% and fibre counts to within 0.3%.
