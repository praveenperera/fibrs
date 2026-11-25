//! GPU tests of the input stage, `#[ignore]`d since they need the GPU, the 0211 CT cache and villa's reference
//! dump (written by a script outside this repository); run them with `--ignored`
//!
//! Paths default to a local layout and are overridden by `STRIPS_INPUT_REF` (the dump),
//! `STRIPS_CT_CACHE`, `STRIPS_CT_URL` and `STRIPS_WEIGHTS`

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use color_eyre::eyre::{Result, WrapErr, ensure, eyre};
use cudarc::driver::sys as cu;
use serde_json::Value;
use volkit::gpu::Buf;

use super::crops::{ChunkCache, CropAt, Crops, FRAME_CROP, Pass, STRIP_CROP, StripInput};
use super::frame::FrameModel;
use super::{Device, Weights};
use crate::strips::ct::Volume;
use crate::strips::types::{Family, FrameRequest, Mat3};

/// `strips-kernels` `crops::zscore` on the host, op for op: `x - mean` exactly as a float32 pair, then the pair
/// product with the reciprocal deviation, rounded once (Rust rounds each operation and fuses only `mul_add`)
fn zscore(x: f32, [mh, ml, ih, il]: [f32; 4]) -> f32 {
    let two_sum = |a: f32, b: f32| {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    };
    let (dh, de) = two_sum(x, -mh);
    let dl = de + -ml;
    let ph = dh * ih;
    let pe = dh.mul_add(ih, -ph);
    ph + (pe + dh.mul_add(il, dl * ih))
}

fn env_path(key: &str, default: &str) -> PathBuf {
    std::env::var_os(key).map(PathBuf::from).unwrap_or_else(|| PathBuf::from(default))
}

/// The reference dump: raw little-endian arrays named in manifest.json
struct Dump {
    dir: PathBuf,
    manifest: HashMap<String, (String, Vec<usize>)>,
}

impl Dump {
    fn open(dir: &Path) -> Result<Self> {
        let text = std::fs::read(dir.join("manifest.json")).wrap_err_with(|| format!("read {}", dir.display()))?;
        let v: Value = serde_json::from_slice(&text)?;
        let mut manifest = HashMap::new();
        for (name, entry) in v.as_object().ok_or_else(|| eyre!("manifest is not an object"))? {
            if name.starts_with('_') {
                continue;
            }
            let dtype = entry[0].as_str().unwrap_or_default().to_string();
            let shape = entry[1].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect();
            manifest.insert(name.clone(), (dtype, shape));
        }
        Ok(Self { dir: dir.to_path_buf(), manifest })
    }

    fn shape(&self, name: &str) -> &[usize] {
        &self.manifest[name].1
    }

    fn read<T: Le>(&self, name: &str, dtype: &str) -> Result<Vec<T>> {
        let (got, shape) = self.manifest.get(name).ok_or_else(|| eyre!("dump has no {name}"))?;
        ensure!(got == dtype, "{name}: {got}, expected {dtype}");
        let bytes = std::fs::read(self.dir.join(format!("{name}.bin")))?;
        let n: usize = shape.iter().product();
        ensure!(bytes.len() == n * size_of::<T>(), "{name}: {} bytes for shape {shape:?}", bytes.len());
        Ok(bytes.chunks_exact(size_of::<T>()).map(T::le).collect())
    }
}

/// A little-endian scalar of the dump
trait Le: Sized {
    fn le(b: &[u8]) -> Self;
}

macro_rules! le {
    ($($t:ty),*) => {$(
        impl Le for $t {
            fn le(b: &[u8]) -> Self {
                <$t>::from_le_bytes(b.try_into().unwrap())
            }
        }
    )*};
}
le!(f32, f64, i64, u32);

impl Dump {}

/// The dump's decisions: frame requests and strip crops
struct Decisions {
    req_at: Vec<CropAt>,
    requests: Vec<FrameRequest>,
    strip_at: Vec<CropAt>,
    tubes: Vec<Vec<[f32; 3]>>,
}

fn mat(v: &[f64]) -> Mat3 {
    [[v[0], v[1], v[2]], [v[3], v[4], v[5]], [v[6], v[7], v[8]]]
}

fn decisions(d: &Dump) -> Result<Decisions> {
    let pos: Vec<f64> = d.read("req_positions", "f8")?;
    let frames: Vec<f64> = d.read("req_frames", "f8")?;
    let paths: Vec<f32> = d.read("req_paths", "f4")?;
    let families: Vec<i64> = d.read("req_families", "i8")?;
    let k = pos.len() / 3;
    let mut req_at = Vec::with_capacity(k);
    let mut requests = Vec::with_capacity(k);
    for i in 0..k {
        let p = [pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]];
        let frame = mat(&frames[9 * i..9 * i + 9]);
        req_at.push(CropAt { pos: p, frame });
        requests.push(FrameRequest {
            pos: p,
            frame,
            paths: paths[128 * i..128 * i + 128].try_into().unwrap(),
            family: if families[i] == 0 { Family::H } else { Family::V },
        });
    }
    let spos: Vec<f64> = d.read("strip_positions", "f8")?;
    let sframes: Vec<f64> = d.read("strip_frames", "f8")?;
    let tube: Vec<f32> = d.read("tube", "f4")?;
    let lens: Vec<i64> = d.read("tube_len", "i8")?;
    ensure!(spos.len() == 3 * k && lens.len() == k, "strip records for {} of {k} decisions", spos.len() / 3);
    let mut strip_at = Vec::with_capacity(k);
    let mut tubes = Vec::with_capacity(k);
    let mut at = 0;
    for i in 0..k {
        strip_at.push(CropAt {
            pos: [spos[3 * i], spos[3 * i + 1], spos[3 * i + 2]],
            frame: mat(&sframes[9 * i..9 * i + 9]),
        });
        let n = lens[i] as usize;
        tubes.push((0..n).map(|j| [tube[3 * (at + j)], tube[3 * (at + j) + 1], tube[3 * (at + j) + 2]]).collect());
        at += n;
    }
    ensure!(at * 3 == tube.len(), "tube points: {} listed, {} stored", at, tube.len() / 3);
    Ok(Decisions { req_at, requests, strip_at, tubes })
}

struct Setup {
    dev: Device,
    cache: ChunkCache,
    crops: Crops,
    frame: FrameModel,
    dump: Dump,
    dec: Decisions,
}

const BATCH: usize = 64;

fn setup(slots: usize) -> Result<Setup> {
    let dump = Dump::open(&env_path("STRIPS_INPUT_REF", "/build/vesuvius/strips-scale/speed/native/ref/input"))?;
    let dec = decisions(&dump)?;
    let url = std::env::var("STRIPS_CT_URL").unwrap_or_else(|_| {
        "s3://vesuvius-challenge-open-data/PHerc0211/volumes/20250821151803-9.362um-1.2m-113keV-masked.zarr".into()
    });
    let vol = Volume::open(&env_path("STRIPS_CT_CACHE", "/build/vesuvius/strips-scale/ctcache"), &url, 0)?;
    let weights = Weights::open(&env_path(
        "STRIPS_WEIGHTS",
        "/build/vesuvius/strips-scale/speed/native/weights/fiber_strip_small_v1.safetensors",
    ))?;
    let dev = Device::open(0)?;
    let cache = ChunkCache::new(&dev, Arc::new(vol), slots, 64)?;
    let crops = Crops::new(&dev, BATCH, 2.0)?;
    let frame = FrameModel::new(&dev, &weights, BATCH)?;
    Ok(Setup { dev, cache, crops, frame, dump, dec })
}

/// Max |a - b|, the fraction of values whose bits differ, and how many are not finite in `a`
fn compare(a: &[f32], b: &[f32]) -> (f32, f64, usize) {
    assert_eq!(a.len(), b.len());
    let mut max = 0f32;
    let mut differ = 0usize;
    let mut bad = 0usize;
    for (x, y) in a.iter().zip(b) {
        if !x.is_finite() {
            bad += 1;
            continue;
        }
        if x.to_bits() != y.to_bits() {
            differ += 1;
            max = max.max((x - y).abs());
        }
    }
    (max, differ as f64 / a.len() as f64, bad)
}

fn load_avg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .split_whitespace()
        .take(3)
        .collect::<Vec<_>>()
        .join(" ")
}

/// Crops, tube and frame model against villa's GPU path on the same decisions and CT
#[test]
#[ignore]
fn input_parity() -> Result<()> {
    let mut s = setup(4096)?;
    let k = s.dec.requests.len();
    let gpu = &s.dev.gpu;
    println!("parity: {k} decisions, load {}", load_avg());

    // frame crops and the frame model over every decision, b64
    let crop_buf = gpu.alloc(BATCH * FRAME_CROP.samples() * 4)?;
    let ref_crops: Vec<f32> = s.dump.read("frame_crops", "f4")?;
    let ref_heading: Vec<f32> = s.dump.read("frame_heading", "f4")?;
    let ref_normal: Vec<f32> = s.dump.read("frame_normal", "f4")?;
    ensure!(s.dump.shape("frame_crops") == [k, 32, 32, 32], "frame_crops {:?}", s.dump.shape("frame_crops"));
    let n = FRAME_CROP.samples();
    let (mut crop_max, mut crop_diff, mut model_max, mut end_max) = (0f32, 0f64, 0f32, 0f32);
    let mut mine: Vec<f32> = Vec::with_capacity(6 * k);
    for start in (0..k).step_by(BATCH) {
        let end = (start + BATCH).min(k);
        let b = end - start;
        gpu.fill(&crop_buf, 0xFF)?;
        s.crops.frame(&s.dev, &mut s.cache, &s.dec.req_at[start..end], &crop_buf)?;
        let got: Vec<f32> = gpu.download_range(&crop_buf, 0, b * n)?;
        let (m, f, bad) = compare(&got, &ref_crops[start * n..end * n]);
        ensure!(bad == 0, "frame crops {start}..{end}: {bad} non-finite values");
        crop_max = crop_max.max(m);
        crop_diff += f * b as f64;

        // the model on our crops (end to end), then on the reference's (the model alone)
        let ours = s.frame.run(&s.dev, &crop_buf, &s.dec.requests[start..end])?;
        gpu.upload_into(&crop_buf, 0, &ref_crops[start * n..end * n])?;
        let theirs = s.frame.run(&s.dev, &crop_buf, &s.dec.requests[start..end])?;
        for (j, (o, t)) in ours.iter().zip(&theirs).enumerate() {
            let i = start + j;
            for c in 0..3 {
                let (rh, rn) = (ref_heading[3 * i + c], ref_normal[3 * i + c]);
                model_max = model_max.max((t.heading[c] - rh).abs()).max((t.normal[c] - rn).abs());
                end_max = end_max.max((o.heading[c] - rh).abs()).max((o.normal[c] - rn).abs());
            }
            mine.extend(o.heading.iter().chain(&o.normal));
        }
    }
    println!(
        "frame crops: max abs diff {crop_max:.3e}, fraction differing {:.4}; frame model on reference crops: max abs \
         diff {model_max:.3e}; end to end (our crops): {end_max:.3e}",
        crop_diff / k as f64
    );
    let out = env_path("STRIPS_INPUT_OUT", "/tmp/strips-input-out");
    std::fs::create_dir_all(&out)?;
    let bytes: Vec<u8> = mine.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(out.join("frame_outputs.bin"), bytes)?;
    println!("frame outputs (K, 6) f32 -> {}", out.join("frame_outputs.bin").display());

    // strip crops and the tube for the subset
    let sub = s.dump.shape("subset_strip_crops")[0];
    let ref_strip: Vec<f32> = s.dump.read("subset_strip_crops", "f4")?;
    let idx: Vec<u32> = s.dump.read("subset_tube_index", "u4")?;
    let val: Vec<f32> = s.dump.read("subset_tube_value", "f4")?;
    let nnz: Vec<i64> = s.dump.read("subset_tube_nnz", "i8")?;
    let sn = STRIP_CROP.samples();
    let strip_buf = gpu.alloc(BATCH * sn * 2 * 4)?;
    let (mut smax, mut sdiff, mut tube_bad, mut tube_nnz) = (0f32, 0f64, 0usize, 0usize);
    let mut at = 0usize;
    for start in (0..sub).step_by(BATCH) {
        let end = (start + BATCH).min(sub);
        let tubes: Vec<&[[f32; 3]]> = s.dec.tubes[start..end].iter().map(Vec::as_slice).collect();
        gpu.fill(&strip_buf, 0xFF)?;
        s.crops.strip(&s.dev, &mut s.cache, &s.dec.strip_at[start..end], &tubes, &strip_buf)?;
        // the crop's CT channel is raw; its readers z-score it on load, as this does on the host
        let StripInput::Raw { stats, .. } = s.crops.strip_input(&strip_buf) else {
            unreachable!("a strip crop is raw")
        };
        let stats: Vec<f32> = gpu.download_range(stats, 0, 4 * (end - start))?;
        for j in 0..end - start {
            let got: Vec<f32> = gpu.download_range(&strip_buf, j * sn * 2, sn * 2)?;
            let st = [stats[4 * j], stats[4 * j + 1], stats[4 * j + 2], stats[4 * j + 3]];
            let ct: Vec<f32> = got.iter().step_by(2).map(|&x| zscore(x, st)).collect();
            let i = start + j;
            let (m, f, bad) = compare(&ct, &ref_strip[i * sn..(i + 1) * sn]);
            ensure!(bad == 0, "strip crop {i}: {bad} non-finite values");
            smax = smax.max(m);
            sdiff += f;
            let mut want = vec![0f32; sn];
            for q in 0..nnz[i] as usize {
                want[idx[at + q] as usize] = val[at + q];
            }
            at += nnz[i] as usize;
            tube_nnz += nnz[i] as usize;
            tube_bad += got.iter().skip(1).step_by(2).zip(&want).filter(|(g, w)| g.to_bits() != w.to_bits()).count();
        }
    }
    println!(
        "strip crops ({sub}): max abs diff {smax:.3e}, fraction differing {:.4}; tube: {tube_bad} of {} values differ \
         ({tube_nnz} non-zero in the reference)",
        sdiff / sub as f64,
        sub * sn
    );
    println!("cache: {:?}", s.cache.counts);
    ensure!(tube_bad == 0, "the tube differs from render_tube at {tube_bad} values");
    ensure!(smax < 1e-3 && crop_max < 1e-3, "crops differ by up to {smax:.3e} (strip), {crop_max:.3e} (frame)");
    ensure!(model_max < 1e-4, "frame model differs by {model_max:.3e} on the reference crops");
    Ok(())
}

/// Seconds between two events recorded on the compute stream
fn timed(dev: &Device, f: impl FnOnce() -> Result<()>) -> Result<f64> {
    let ctx = dev.gpu.inner.context();
    let (a, b) = (
        ctx.new_event(Some(cu::CUevent_flags::CU_EVENT_DEFAULT))?,
        ctx.new_event(Some(cu::CUevent_flags::CU_EVENT_DEFAULT))?,
    );
    a.record(dev.gpu.inner.stream())?;
    f()?;
    b.record(dev.gpu.inner.stream())?;
    b.synchronize()?;
    Ok(a.elapsed_ms(&b)? as f64 / 1e3)
}

/// ms/decision at b64 of the frame crops, the frame model, the strip crops with the tube, and the strip crop's
/// write ceiling, over the dump's decisions in order from a warm cache
#[test]
#[ignore]
fn input_bench() -> Result<()> {
    let mut s = setup(4096)?;
    let gpu = &s.dev.gpu;
    let k = s.dec.requests.len() / BATCH * BATCH;
    let crop_buf = gpu.alloc(BATCH * FRAME_CROP.samples() * 4)?;
    let strip_buf: Buf = gpu.alloc(BATCH * STRIP_CROP.samples() * 2 * 4)?;
    let tubes: Vec<&[[f32; 3]]> = s.dec.tubes.iter().map(Vec::as_slice).collect();
    let no_tubes: Vec<&[[f32; 3]]> = vec![&[]; BATCH];
    println!("bench: {k} decisions in batches of {BATCH}, load {}", load_avg());

    // cold pass: fills the cache and counts what each launch needs
    let started = Instant::now();
    for start in (0..k).step_by(BATCH) {
        let r = start..start + BATCH;
        s.crops.frame(&s.dev, &mut s.cache, &s.dec.req_at[r.clone()], &crop_buf)?;
        s.crops.strip(&s.dev, &mut s.cache, &s.dec.strip_at[r.clone()], &tubes[r], &strip_buf)?;
    }
    gpu.sync()?;
    let cold = s.cache.counts;
    println!(
        "cold pass {:.2} s: {} chunks uploaded ({:.2} GB), {} fill, most per launch {}, cache {:.2} GiB",
        started.elapsed().as_secs_f64(),
        cold.uploads,
        cold.uploads as f64 * 2.097152e-3,
        cold.fills,
        cold.most,
        s.cache.device_bytes() as f64 / (1u64 << 30) as f64
    );

    for rep in 0..3 {
        let (mut fc, mut fm, mut sc, mut ceil) = (0f64, 0f64, 0f64, 0f64);
        for start in (0..k).step_by(BATCH) {
            let r = start..start + BATCH;
            fc += timed(&s.dev, || s.crops.frame(&s.dev, &mut s.cache, &s.dec.req_at[r.clone()], &crop_buf))?;
            fm += timed(&s.dev, || s.frame.launch(&s.dev, &crop_buf, &s.dec.requests[r.clone()]))?;
            sc += timed(&s.dev, || {
                s.crops.strip(&s.dev, &mut s.cache, &s.dec.strip_at[r.clone()], &tubes[r.clone()], &strip_buf)
            })?;
            // the same launch reading no CT and rendering no tube: the write ceiling
            ceil += timed(&s.dev, || {
                s.crops.sample(
                    &s.dev,
                    &mut s.cache,
                    &STRIP_CROP,
                    &s.dec.strip_at[r.clone()],
                    Some(&no_tubes),
                    &strip_buf,
                    Pass::WriteCeiling,
                )
            })?;
        }
        let per = |t: f64| t * 1e3 / k as f64;
        println!(
            "rep {rep}: ms/decision at b{BATCH}: frame crop {:.4}, frame model {:.4}, strip crop + tube {:.4}, strip write \
             ceiling {:.4}; load {}",
            per(fc),
            per(fm),
            per(sc),
            per(ceil),
            load_avg()
        );
    }
    println!("warm cache: {:?}", s.cache.counts);
    Ok(())
}

/// What one pass of [`input_prefetch`] produced: a hash of every launch's output bits, and the cache's counts
struct PrefetchPass {
    launches: Vec<u64>,
    uploads: u64,
    fills: u64,
    evictions: u64,
    prefetched: u64,
}

/// The dump's frame launches then its strip subset's, at b64 in the engine's order, through a fresh cache of
/// `slots` chunks; with `ahead`, each launch reads the next one's chunks ahead once its kernels are queued, as the
/// engine does
fn prefetch_pass(slots: usize, ahead: bool) -> Result<PrefetchPass> {
    use std::hash::{DefaultHasher, Hasher};

    let mut s = setup(slots)?;
    let gpu = &s.dev.gpu;
    let k = s.dec.requests.len();
    let sub = s.dump.shape("subset_strip_crops")[0];
    let frame_buf = gpu.alloc(BATCH * FRAME_CROP.samples() * 4)?;
    let strip_buf = gpu.alloc(BATCH * STRIP_CROP.samples() * 2 * 4)?;
    let tubes: Vec<&[[f32; 3]]> = s.dec.tubes.iter().map(Vec::as_slice).collect();
    // (frame launch?, decision range) in launch order
    let launches: Vec<(bool, std::ops::Range<usize>)> = (0..k)
        .step_by(BATCH)
        .map(|a| (true, a..(a + BATCH).min(k)))
        .chain((0..sub).step_by(BATCH).map(|a| (false, a..(a + BATCH).min(sub))))
        .collect();

    let mut hashes = Vec::with_capacity(launches.len());
    for (i, (frame, r)) in launches.iter().enumerate() {
        let (buf, n) = if *frame {
            s.crops.frame(&s.dev, &mut s.cache, &s.dec.req_at[r.clone()], &frame_buf)?;
            (&frame_buf, FRAME_CROP.samples())
        } else {
            s.crops.strip(&s.dev, &mut s.cache, &s.dec.strip_at[r.clone()], &tubes[r.clone()], &strip_buf)?;
            (&strip_buf, STRIP_CROP.samples() * 2)
        };
        if let Some((next_frame, next)) = launches.get(i + 1).filter(|_| ahead) {
            let (spec, at) = if *next_frame { (&FRAME_CROP, &s.dec.req_at) } else { (&STRIP_CROP, &s.dec.strip_at) };
            s.crops.prefetch(&mut s.cache, &[(spec, &at[next.clone()])])?;
        }

        let got: Vec<f32> = gpu.download_range(buf, 0, r.len() * n)?;
        let mut h = DefaultHasher::new();
        for v in &got {
            h.write_u32(v.to_bits());
        }
        hashes.push(h.finish());
    }

    let c = s.cache.counts;
    Ok(PrefetchPass {
        launches: hashes,
        uploads: c.uploads,
        fills: c.fills,
        evictions: c.evictions,
        prefetched: c.prefetched,
    })
}

/// Reading the next launch's chunks ahead changes no output bit and no cache decision: the dump's launches through
/// a cache small enough to evict give the same crops, uploads, fills and evictions with read-ahead as without
#[test]
#[ignore]
fn input_prefetch() -> Result<()> {
    // 256 slots against up to 112 chunks per launch and 797 in all: every pass evicts
    let slots = 256;
    println!("prefetch: {slots} slots, load {}", load_avg());
    let t = Instant::now();
    let plain = prefetch_pass(slots, false)?;
    let plain_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let ahead = prefetch_pass(slots, true)?;
    let ahead_s = t.elapsed().as_secs_f64();
    for (name, p, secs) in [("on demand", &plain, plain_s), ("read ahead", &ahead, ahead_s)] {
        println!(
            "{name}: {} launches in {secs:.2} s (cold disk reads included), {} uploads ({} read ahead), {} fills, {} \
             evictions",
            p.launches.len(),
            p.uploads,
            p.prefetched,
            p.fills,
            p.evictions
        );
    }

    ensure!(plain.launches.len() == ahead.launches.len(), "launch counts differ");
    let differ: Vec<usize> = (0..plain.launches.len()).filter(|&i| plain.launches[i] != ahead.launches[i]).collect();
    ensure!(differ.is_empty(), "launches {differ:?} differ with read-ahead");
    ensure!(plain.evictions > 0, "the pass evicted nothing: the cache is too large to test eviction");
    ensure!(
        plain.prefetched == 0 && ahead.prefetched > 0,
        "read-ahead count {} (on demand {})",
        ahead.prefetched,
        plain.prefetched
    );
    ensure!(
        (plain.uploads, plain.fills, plain.evictions) == (ahead.uploads, ahead.fills, ahead.evictions),
        "the cache's uploads, fills and evictions differ with read-ahead"
    );
    Ok(())
}
