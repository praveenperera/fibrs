# Synthetic tracer fixture

`predictions.zarr` is a small synthetic AFV prediction (2 × 128 × 192 × 192 `semantic`, its pooled `tracing`),
written by `scripts/synthetic_predictions.py` (a rerun writes the same arrays, though Blosc may compress
them to different chunk bytes). `reference.afv` is what the original tracer,
`avf-spline-generator` 0.2.0 ([Qualzz/AVF_SPLINE_GENERATOR](https://github.com/Qualzz/AVF_SPLINE_GENERATOR) @
`bee798f`, with numpy 2.4.6, scipy 1.18.1, numba 0.64.0, zarr 2.18.7 and Python's SQLite 3.53.1), wrote for it
(117 fibres) with:

```sh
avf-spline-generator --predictions predictions.zarr --array semantic --channel-order VH \
  --tracing-predictions predictions.zarr --tracing-array tracing --voxel-size-um 9.362 \
  --storage compressed --device cpu --threads 4 --output reference.afv --work-dir work
```

CI traces the same input with fibrs, through the drop-in and through `python -m fibrs`, and
`scripts/compare_afv.py` must find every table and every metadata value but the per-run uuid identical.
