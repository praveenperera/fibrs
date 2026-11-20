//! The `fb_walk_*` kernels: the fibre walk's per-seed tracer ([`crate::walk`]) for one chunk of seeds of a family
//!
//! The family's volumes stay resident for the whole walk as brick pools with a table over the full brick grid
//! ([`NO_BRICK`] where a brick reads as its default): presence (u8, default 0), centrality (u16, default 0) and the
//! packed direction bytes `nx << 8 | ny` (u16, default `128 << 8 | 128`). Coverage is a table of brick addresses
//! (0 where the brick reads as 0) that the host extends as commits mark new bricks; the kernels read it while the
//! host's commits change it, with volatile loads, so each u16 read is the old value or the new one and is read
//! once. Every read is recorded as a check, so a result holds for any coverage with the values its checks saw.
//!
//! A chunk runs as: [`fb_walk_seeds`] (thread per seed: `trace_seed`'s start, survivors appended to a list),
//! [`fb_walk_trace`] (thread per survivor: both halves into the survivor's path slot), [`fb_walk_spans`] (thread
//! per survivor: `valid_curve_spans` and the length filter), [`fb_walk_scan`] over the per-seed counts (checks,
//! accepted points, curves), [`fb_walk_gather`] (thread per seed: checks, points, tangents and curves copied out
//! in seed order) and [`fb_walk_results`] (thread per seed: its ranges, or `Cpu` when it has to be traced on the CPU). Outputs that
//! would overflow their buffers send the seeds that reach past them back to the CPU, which by the scans' order is
//! a suffix of the chunk. Marks are left to the host, which runs `mark_list` on a result's accepted points when it
//! uses the result: most results traced ahead are never used, and their marks would outweigh everything else

use core::ptr::{read_volatile, write_volatile};

use cuda_device::atomic::{AtomicOrdering, DeviceAtomicU32};
use cuda_device::float::{
    add_rn_f32, add_rn_f64, div_rn_f32, div_rn_f64, mul_rn_f32, mul_rn_f64, sqrt_rn_f32, sqrt_rn_f64,
};
use cuda_device::shared::SharedArray;
use cuda_device::thread::{blockDim_x, blockIdx_x, sync_threads, threadIdx_x};
use cuda_device::kernel;

use crate::walk::{self, Curve, Fields, Ops, Path, Reads, SpansFail};
use crate::walk_layout::{
    CHECK_WORDS, CU_WORDS, PREFIX_CHECKS, RESULT_WORDS, SPAN_CAP, SURV_WORDS, T_CHECKS, T_CURVES, T_PTS, T_SEEDS,
    T_SURV,
};
use crate::{BRICK, MASK, NO_BRICK, SHIFT};

/// The tracer's arithmetic in explicit round-to-nearest instructions
struct Rn;

impl Ops for Rn {
    #[inline(always)]
    fn mul(a: f64, b: f64) -> f64 {
        mul_rn_f64(a, b)
    }

    #[inline(always)]
    fn add(a: f64, b: f64) -> f64 {
        add_rn_f64(a, b)
    }

    #[inline(always)]
    fn div(a: f64, b: f64) -> f64 {
        div_rn_f64(a, b)
    }

    #[inline(always)]
    fn sqrt(a: f64) -> f64 {
        sqrt_rn_f64(a)
    }

    #[inline(always)]
    fn mul32(a: f32, b: f32) -> f32 {
        mul_rn_f32(a, b)
    }

    #[inline(always)]
    fn add32(a: f32, b: f32) -> f32 {
        add_rn_f32(a, b)
    }

    #[inline(always)]
    fn div32(a: f32, b: f32) -> f32 {
        div_rn_f32(a, b)
    }

    #[inline(always)]
    fn sqrt32(a: f32) -> f32 {
        sqrt_rn_f32(a)
    }
}

/// Per-seed state after [`fb_walk_seeds`]: exited (no curves), a survivor (its index in the low [`INDEX_BITS`]),
/// or back to the CPU, with the number of prefix checks at [`PRE_SHIFT`]
const EXITED: u32 = 0;
const SURVIVOR: u32 = 1 << 31;
const CPU: u32 = 1 << 30;
const PRE_SHIFT: u32 = 28;
const INDEX_BITS: u32 = (1 << PRE_SHIFT) - 1;

/// One resident volume: a table over the full brick grid and the pool of brick payloads it points into
#[derive(Clone, Copy)]
struct Bricks<T: Copy> {
    table: *const u32,
    pool: *const T,
    default: T,
}

/// The volumes' shape and brick grid
#[derive(Clone, Copy)]
struct Geom {
    dims: [i32; 3],
    gy: u32,
    gx: u32,
}

impl Geom {
    #[inline(always)]
    fn brick(&self, z: i32, y: i32, x: i32) -> u32 {
        let (z, y, x) = (z as u32, y as u32, x as u32);
        ((z >> SHIFT) * self.gy + (y >> SHIFT)) * self.gx + (x >> SHIFT)
    }

    #[inline(always)]
    fn offset(z: i32, y: i32, x: i32) -> u32 {
        let (z, y, x) = (z as u32, y as u32, x as u32);
        ((z & MASK) << (2 * SHIFT)) | ((y & MASK) << SHIFT) | (x & MASK)
    }
}

impl<T: Copy> Bricks<T> {
    /// The payload of brick `b`, or None for its default
    #[inline(always)]
    unsafe fn slot(&self, b: u32) -> *const T {
        let slot = unsafe { *self.table.add(b as usize) };
        if slot == NO_BRICK {
            return core::ptr::null();
        }
        unsafe { self.pool.add(slot as usize * BRICK as usize) }
    }

    #[inline(always)]
    unsafe fn get(&self, g: &Geom, z: i32, y: i32, x: i32) -> T {
        let p = unsafe { self.slot(g.brick(z, y, x)) };
        if p.is_null() {
            return self.default;
        }
        unsafe { *p.add(Geom::offset(z, y, x) as usize) }
    }

    /// The eight corners of an in-bounds cell, one table lookup when the cell lies inside one brick
    #[inline(always)]
    unsafe fn corners(&self, g: &Geom, z: i32, y: i32, x: i32) -> [T; 8] {
        let edge = MASK as i32;
        if (z & edge) == edge || (y & edge) == edge || (x & edge) == edge {
            return unsafe {
                [
                    self.get(g, z, y, x),
                    self.get(g, z, y, x + 1),
                    self.get(g, z, y + 1, x),
                    self.get(g, z, y + 1, x + 1),
                    self.get(g, z + 1, y, x),
                    self.get(g, z + 1, y, x + 1),
                    self.get(g, z + 1, y + 1, x),
                    self.get(g, z + 1, y + 1, x + 1),
                ]
            };
        }
        let p = unsafe { self.slot(g.brick(z, y, x)) };
        if p.is_null() {
            return [self.default; 8];
        }
        const ROW: usize = 1 << SHIFT;
        const PLANE: usize = 1 << (2 * SHIFT);
        let at = Geom::offset(z, y, x) as usize;
        unsafe {
            [
                *p.add(at),
                *p.add(at + 1),
                *p.add(at + ROW),
                *p.add(at + ROW + 1),
                *p.add(at + PLANE),
                *p.add(at + PLANE + 1),
                *p.add(at + PLANE + ROW),
                *p.add(at + PLANE + ROW + 1),
            ]
        }
    }
}

/// The family's resident fields as the tracer reads them
#[derive(Clone, Copy)]
struct Vols {
    g: Geom,
    pres: Bricks<u8>,
    cent: Bricks<u16>,
    dirs: Bricks<u16>,
    /// `decode_packed` of every packed direction, four f64 per entry
    decode: *const f64,
    /// `ring()`'s 13 exponentials
    ring: *const f64,
}

impl Fields for Vols {
    #[inline(always)]
    fn dims(&self) -> [i32; 3] {
        self.g.dims
    }

    #[inline(always)]
    fn pres_corners(&self, z: i32, y: i32, x: i32) -> [u8; 8] {
        unsafe { self.pres.corners(&self.g, z, y, x) }
    }

    #[inline(always)]
    fn cent_corners(&self, z: i32, y: i32, x: i32) -> [u16; 8] {
        unsafe { self.cent.corners(&self.g, z, y, x) }
    }

    #[inline(always)]
    fn dir(&self, z: i32, y: i32, x: i32) -> u16 {
        unsafe { self.dirs.get(&self.g, z, y, x) }
    }

    #[inline(always)]
    fn dir_corners(&self, z: i32, y: i32, x: i32) -> [u16; 8] {
        unsafe { self.dirs.corners(&self.g, z, y, x) }
    }

    #[inline(always)]
    fn decode(&self, d: u16) -> [f64; 3] {
        let at = d as usize * 4;
        unsafe { [*self.decode.add(at), *self.decode.add(at + 1), *self.decode.add(at + 2)] }
    }

    #[inline(always)]
    fn ring(&self, i: usize) -> f64 {
        unsafe { *self.ring.add(i) }
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
unsafe fn vols(
    d: u32,
    h: u32,
    w: u32,
    gy: u32,
    gx: u32,
    pres_table: *const u32,
    pres_pool: *const u8,
    cent_table: *const u32,
    cent_pool: *const u16,
    dir_table: *const u32,
    dir_pool: *const u16,
    decode: *const f64,
    ring: *const f64,
) -> Vols {
    Vols {
        g: Geom { dims: [d as i32, h as i32, w as i32], gy, gx },
        pres: Bricks { table: pres_table, pool: pres_pool, default: 0 },
        cent: Bricks { table: cent_table, pool: cent_pool, default: 0 },
        dirs: Bricks { table: dir_table, pool: dir_pool, default: 128 << 8 | 128 },
        decode,
        ring,
    }
}

/// Coverage reads, each written as a check record at `out[n]` while `n < cap`
struct CovReads {
    g: Geom,
    /// Each brick's coverage payload address, or 0 while it reads as 0
    table: *const u64,
    out: *mut u64,
    n: u32,
    cap: u32,
}

impl Reads for CovReads {
    #[inline(always)]
    fn read(&mut self, z: i32, y: i32, x: i32, kind: u8, t: [f64; 3]) -> u16 {
        let brick = self.g.brick(z, y, x);
        let offset = Geom::offset(z, y, x);
        // volatile: the host's commits change the table and the values while this runs, and each is read once
        let addr = unsafe { read_volatile(self.table.add(brick as usize)) };
        let seen = if addr == 0 { 0 } else { unsafe { read_volatile((addr as *const u16).add(offset as usize)) } };
        if self.n < self.cap {
            let rec = unsafe { self.out.add(self.n as usize * CHECK_WORDS) };
            unsafe {
                *rec = brick as u64 | (offset as u64) << 32 | (seen as u64) << 48;
                *rec.add(1) = kind as u64;
                *rec.add(2) = t[0].to_bits();
                *rec.add(3) = t[1].to_bits();
                *rec.add(4) = t[2].to_bits();
            }
        }
        self.n += 1;
        seen
    }
}

/// A survivor's path slot: `2 * cap + 1` points and tangents with the start point at `cap`, the forward half
/// after it and the back half before it, reversed, its tangents negated, so the slot from `cap - nb` is
/// `trace_seed`'s joined `pts_c`/`tans_c`
struct Slot {
    pts: *mut f32,
    tans: *mut f32,
    cap: u32,
}

impl Slot {
    #[inline(always)]
    fn at(&self, half: u32, i: u32) -> usize {
        (if half == 0 { self.cap + 1 + i } else { self.cap - 1 - i }) as usize * 3
    }
}

impl Path for Slot {
    #[inline(always)]
    fn pt(&self, half: u32, i: u32) -> [f32; 3] {
        let at = self.at(half, i);
        unsafe { [*self.pts.add(at), *self.pts.add(at + 1), *self.pts.add(at + 2)] }
    }

    #[inline(always)]
    fn push(&mut self, half: u32, i: u32, p: [f32; 3], t: [f32; 3]) {
        let at = self.at(half, i);
        let t = if half == 0 { t } else { [-t[0], -t[1], -t[2]] };
        unsafe {
            *self.pts.add(at) = p[0];
            *self.pts.add(at + 1) = p[1];
            *self.pts.add(at + 2) = p[2];
            *self.tans.add(at) = t[0];
            *self.tans.add(at + 1) = t[1];
            *self.tans.add(at + 2) = t[2];
        }
    }
}

/// A survivor's joined path from its slot, with its pending stack, `cum_s` and accepted spans in global scratch
struct Joined {
    pts: *const f32,
    stack: *mut u32,
    depth: u32,
    room: u32,
    cum: *mut f32,
    spans: *mut u32,
    n_spans: u32,
}

impl Curve for Joined {
    #[inline(always)]
    fn pt(&self, i: u32) -> [f32; 3] {
        let at = i as usize * 3;
        unsafe { [*self.pts.add(at), *self.pts.add(at + 1), *self.pts.add(at + 2)] }
    }

    #[inline(always)]
    fn push(&mut self, a: u32, b: u32) -> bool {
        if self.depth >= self.room {
            return false;
        }
        unsafe { *self.stack.add(self.depth as usize) = a << 16 | b };
        self.depth += 1;
        true
    }

    #[inline(always)]
    fn pop(&mut self) -> (u32, u32) {
        if self.depth == 0 {
            return (u32::MAX, u32::MAX);
        }
        self.depth -= 1;
        let v = unsafe { *self.stack.add(self.depth as usize) };
        (v >> 16, v & 0xffff)
    }

    #[inline(always)]
    fn cum(&self, i: u32) -> f32 {
        unsafe { *self.cum.add(i as usize) }
    }

    #[inline(always)]
    fn set_cum(&mut self, i: u32, v: f32) {
        unsafe { *self.cum.add(i as usize) = v };
    }

    #[inline(always)]
    fn accept(&mut self, a: u32, b: u32) -> bool {
        if self.n_spans >= SPAN_CAP {
            return false;
        }
        unsafe { *self.spans.add(self.n_spans as usize) = a << 16 | b };
        self.n_spans += 1;
        true
    }
}

#[inline(always)]
fn thread_1d() -> u32 {
    blockIdx_x() * blockDim_x() + threadIdx_x()
}

#[inline(always)]
fn min_u32(a: u32, b: u32) -> u32 {
    if a < b { a } else { b }
}

/// `trace_seed`'s start for seeds `first..first + n` of the catalogue: per seed its state ([`EXITED`],
/// [`SURVIVOR`] with its index, or [`CPU`] when the survivor list is full), its prefix checks (three records per
/// seed) and their count; survivors get their recentred start and direction in `surv_cu` and their seed (chunk
/// index) in `surv_seed`. `seeds` holds z, y, x and a primary flag per catalogue seed. Thread 0 records `n` in
/// `totals`, for the scans
///
/// Grid: x over the chunk's seeds
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fb_walk_seeds(
    d: u32,
    h: u32,
    w: u32,
    gy: u32,
    gx: u32,
    pres_table: *const u32,
    pres_pool: *const u8,
    cent_table: *const u32,
    cent_pool: *const u16,
    dir_table: *const u32,
    dir_pool: *const u16,
    decode: *const f64,
    ring: *const f64,
    cov_table: *const u64,
    seeds: *const i32,
    first: u32,
    n: u32,
    surv_cap: u32,
    state: *mut u32,
    pre: *mut u64,
    n_checks: *mut u32,
    totals: *mut u32,
    surv_seed: *mut u32,
    surv_cu: *mut f64,
) {
    let s = thread_1d();
    if s == 0 {
        unsafe { *totals.add(T_SEEDS) = n };
    }
    if s >= n {
        return;
    }
    let f = unsafe { vols(d, h, w, gy, gx, pres_table, pres_pool, cent_table, cent_pool, dir_table, dir_pool, decode, ring) };
    let at = unsafe { seeds.add((first + s) as usize * 4) };
    let seed = unsafe { [*at, *at.add(1), *at.add(2)] };
    let primary = unsafe { *at.add(3) } != 0;
    let out = unsafe { pre.add(s as usize * PREFIX_CHECKS as usize * CHECK_WORDS) };
    let mut r = CovReads { g: f.g, table: cov_table, out, n: 0, cap: PREFIX_CHECKS };
    let start = walk::prefix::<Rn, Vols, CovReads>(&f, &mut r, seed, primary);
    unsafe { *n_checks.add(s as usize) = r.n };
    let pre_count = r.n << PRE_SHIFT;
    let Some((c, u)) = start else {
        unsafe { *state.add(s as usize) = EXITED | pre_count };
        return;
    };
    let slot = unsafe { DeviceAtomicU32::from_ptr(totals.add(T_SURV)).fetch_add(1, AtomicOrdering::Relaxed) };
    if slot >= surv_cap {
        unsafe {
            *state.add(s as usize) = CPU;
            *n_checks.add(s as usize) = 0;
        }
        return;
    }
    unsafe {
        *state.add(s as usize) = SURVIVOR | pre_count | slot;
        *surv_seed.add(slot as usize) = s;
        let cu = surv_cu.add(slot as usize * CU_WORDS);
        *cu = c[0];
        *cu.add(1) = c[1];
        *cu.add(2) = c[2];
        *cu.add(3) = u[0];
        *cu.add(4) = u[1];
        *cu.add(5) = u[2];
    }
}

/// Both halves of each survivor, `cap` steps each at most, into its path slot (`2 * cap + 1` points from
/// `slot * (2 * cap + 1)`), its trace checks into `surv_checks` (`2 * cap` records per survivor), and nf, nb and
/// the check count into `surv_len`; a half that runs out of steps sends the seed back to the CPU
///
/// Grid: x over `surv_cap`
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fb_walk_trace(
    d: u32,
    h: u32,
    w: u32,
    gy: u32,
    gx: u32,
    pres_table: *const u32,
    pres_pool: *const u8,
    cent_table: *const u32,
    cent_pool: *const u16,
    dir_table: *const u32,
    dir_pool: *const u16,
    decode: *const f64,
    ring: *const f64,
    cov_table: *const u64,
    cap: u32,
    surv_cap: u32,
    totals: *const u32,
    surv_cu: *const f64,
    surv_len: *mut u32,
    path_pts: *mut f32,
    path_tans: *mut f32,
    surv_checks: *mut u64,
) {
    let i = thread_1d();
    if i >= min_u32(unsafe { *totals.add(T_SURV) }, surv_cap) {
        return;
    }
    let f = unsafe { vols(d, h, w, gy, gx, pres_table, pres_pool, cent_table, cent_pool, dir_table, dir_pool, decode, ring) };
    let cu = unsafe { surv_cu.add(i as usize * CU_WORDS) };
    let (c, u) = unsafe { ([*cu, *cu.add(1), *cu.add(2)], [*cu.add(3), *cu.add(4), *cu.add(5)]) };
    let width = (2 * cap + 1) as usize;
    let at = i as usize * width * 3;
    let mut slot = Slot { pts: unsafe { path_pts.add(at) }, tans: unsafe { path_tans.add(at) }, cap };
    let checks = unsafe { surv_checks.add(i as usize * 2 * cap as usize * CHECK_WORDS) };
    let mut r = CovReads { g: f.g, table: cov_table, out: checks, n: 0, cap: 2 * cap };
    let traced = walk::trace_halves::<Rn, Vols, CovReads, Slot>(&f, &mut r, &mut slot, c, u, cap);
    let len = unsafe { surv_len.add(i as usize * SURV_WORDS) };
    let Some((nf, nb)) = traced else {
        unsafe { *len = 1 };
        return;
    };
    let center = cap as usize * 3;
    let (cf, uf) = ([c[0] as f32, c[1] as f32, c[2] as f32], [u[0] as f32, u[1] as f32, u[2] as f32]);
    unsafe {
        *slot.pts.add(center) = cf[0];
        *slot.pts.add(center + 1) = cf[1];
        *slot.pts.add(center + 2) = cf[2];
        *slot.tans.add(center) = uf[0];
        *slot.tans.add(center + 1) = uf[1];
        *slot.tans.add(center + 2) = uf[2];
        *len = 0;
        *len.add(1) = nf;
        *len.add(2) = nb;
        *len.add(3) = r.n;
    }
}

/// Each survivor's accepted spans (`valid_curve_spans` and the length filter) into `spans` ([`SPAN_CAP`] packed
/// `a << 16 | b` per survivor, indices into its joined path), and per seed its checks, accepted points and curves
/// in `n_checks`, `n_pts` and `n_curves` (left as they are for other seeds); a survivor whose spans fail goes back
/// to the CPU with zero counts. `stack` and `cum` hold `2 * cap + 1` scratch values per survivor
///
/// Grid: x over `surv_cap`
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fb_walk_spans(
    cap: u32,
    surv_cap: u32,
    totals: *const u32,
    seeds: *const i32,
    first: u32,
    surv_seed: *const u32,
    surv_len: *mut u32,
    path_pts: *const f32,
    stack: *mut u32,
    cum: *mut f32,
    spans: *mut u32,
    n_checks: *mut u32,
    n_pts: *mut u32,
    n_curves: *mut u32,
) {
    let i = thread_1d();
    if i >= min_u32(unsafe { *totals.add(T_SURV) }, surv_cap) {
        return;
    }
    let s = unsafe { *surv_seed.add(i as usize) } as usize;
    let len = unsafe { surv_len.add(i as usize * SURV_WORDS) };
    let cpu = || unsafe {
        *len = 1;
        *n_checks.add(s) = 0;
        *n_pts.add(s) = 0;
        *n_curves.add(s) = 0;
    };
    if unsafe { *len } != 0 {
        cpu();
        return;
    }
    let (nf, nb, checks) = unsafe { (*len.add(1), *len.add(2), *len.add(3)) };
    let pre = unsafe { *n_checks.add(s) };
    let ntot = nb + 1 + nf;
    let mut accepted = 0;
    let mut points = 0;
    if ntot >= 8 {
        let width = (2 * cap + 1) as usize;
        let primary = unsafe { *seeds.add((first as usize + s) * 4 + 3) } != 0;
        let mut j = Joined {
            pts: unsafe { path_pts.add((i as usize * width + (cap - nb) as usize) * 3) },
            stack: unsafe { stack.add(i as usize * width) },
            depth: 0,
            room: width as u32,
            cum: unsafe { cum.add(i as usize * width) },
            spans: unsafe { spans.add(i as usize * SPAN_CAP as usize) },
            n_spans: 0,
        };
        let res: Result<(), SpansFail> = walk::spans::<Rn, Joined>(&mut j, ntot, nb, primary);
        if res.is_err() {
            cpu();
            return;
        }
        accepted = j.n_spans;
        let mut k = 0;
        while k < accepted {
            let v = unsafe { *j.spans.add(k as usize) };
            points += (v & 0xffff) - (v >> 16);
            k += 1;
        }
    }
    unsafe {
        *len.add(4) = accepted;
        *len.add(5) = points;
        *n_checks.add(s) = pre + checks;
        *n_pts.add(s) = points;
        *n_curves.add(s) = accepted;
    }
}

/// Exclusive prefix sum in place of `data[..count]`, `count = min(*count_at, cap)`, its total into `*total`: one
/// block of 1024 threads, each summing a contiguous run
///
/// Grid: one block of 1024
#[kernel]
pub unsafe fn fb_walk_scan(data: *mut u32, count_at: *const u32, cap: u32, total: *mut u32) {
    static mut SUMS: SharedArray<u32, 1024> = SharedArray::UNINIT;
    let t = threadIdx_x();
    let n = min_u32(unsafe { *count_at }, cap);
    let per = n.div_ceil(1024);
    let lo = min_u32(t * per, n);
    let hi = min_u32(lo + per, n);
    let mut sum = 0u32;
    let mut i = lo;
    while i < hi {
        sum += unsafe { *data.add(i as usize) };
        i += 1;
    }
    let sums = unsafe { SharedArray::as_raw_mut_ptr(&raw mut SUMS) };
    unsafe { *sums.add(t as usize) = sum };
    sync_threads();
    let mut off = 1;
    while off < 1024 {
        let v = if t >= off { unsafe { *sums.add((t - off) as usize) } } else { 0 };
        sync_threads();
        unsafe { *sums.add(t as usize) += v };
        sync_threads();
        off <<= 1;
    }
    let mut run = unsafe { *sums.add(t as usize) } - sum;
    if t == 1023 {
        unsafe { *total = run + sum };
    }
    let mut i = lo;
    while i < hi {
        let v = unsafe { *data.add(i as usize) };
        unsafe { *data.add(i as usize) = run };
        run += v;
        i += 1;
    }
}

/// Copy each seed's checks, accepted points, tangents and curves out in seed order, from the scanned offsets in
/// `n_checks`, `n_pts` and `n_curves`; nothing past a buffer's cap is written
///
/// Grid: x over the chunk's seeds
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fb_walk_gather(
    cap: u32,
    n: u32,
    state: *const u32,
    pre: *const u64,
    n_checks: *const u32,
    n_pts: *const u32,
    n_curves: *const u32,
    surv_len: *const u32,
    surv_checks: *const u64,
    path_pts: *const f32,
    path_tans: *const f32,
    spans: *const u32,
    check_cap: u32,
    pts_cap: u32,
    curve_cap: u32,
    out_checks: *mut u64,
    out_pts: *mut f32,
    out_tans: *mut f32,
    out_curves: *mut u32,
) {
    let s = thread_1d();
    if s >= n {
        return;
    }
    let st = unsafe { *state.add(s as usize) };
    if st & CPU != 0 {
        return;
    }
    let surv = st & SURVIVOR != 0;
    let i = (st & INDEX_BITS) as usize;
    let len = unsafe { surv_len.add(i * SURV_WORDS) };
    if surv && unsafe { *len } != 0 {
        // the survivor went back to the CPU in the trace or spans
        return;
    }
    let copy_checks = |src: *const u64, count: u32, at: u32| {
        let mut k = 0;
        while k < count && at + k < check_cap {
            let from = unsafe { src.add(k as usize * CHECK_WORDS) };
            let to = unsafe { out_checks.add((at + k) as usize * CHECK_WORDS) };
            let mut wd = 0;
            while wd < CHECK_WORDS {
                unsafe { *to.add(wd) = *from.add(wd) };
                wd += 1;
            }
            k += 1;
        }
    };
    let c_out = unsafe { *n_checks.add(s as usize) };
    let n_pre = (st >> PRE_SHIFT) & 3;
    copy_checks(unsafe { pre.add(s as usize * PREFIX_CHECKS as usize * CHECK_WORDS) }, n_pre, c_out);
    if !surv {
        return;
    }
    let width = (2 * cap + 1) as usize;
    copy_checks(unsafe { surv_checks.add(i * 2 * cap as usize * CHECK_WORDS) }, unsafe { *len.add(3) }, c_out + n_pre);
    let nb = unsafe { *len.add(2) };
    let base = i * width + (cap - nb) as usize;
    let mut p_out = unsafe { *n_pts.add(s as usize) };
    let mut cv_out = unsafe { *n_curves.add(s as usize) };
    let accepted = unsafe { *len.add(4) };
    let mut k = 0;
    while k < accepted {
        let v = unsafe { *spans.add(i * SPAN_CAP as usize + k as usize) };
        let (a, b) = (v >> 16, v & 0xffff);
        if cv_out < curve_cap {
            unsafe {
                *out_curves.add(cv_out as usize * 2) = p_out;
                *out_curves.add(cv_out as usize * 2 + 1) = p_out + (b - a);
            }
        }
        cv_out += 1;
        let mut j = a;
        while j < b {
            if p_out < pts_cap {
                let (from, to) = ((base + j as usize) * 3, p_out as usize * 3);
                unsafe {
                    *out_pts.add(to) = *path_pts.add(from);
                    *out_pts.add(to + 1) = *path_pts.add(from + 1);
                    *out_pts.add(to + 2) = *path_pts.add(from + 2);
                    *out_tans.add(to) = *path_tans.add(from);
                    *out_tans.add(to + 1) = *path_tans.add(from + 1);
                    *out_tans.add(to + 2) = *path_tans.add(from + 2);
                }
            }
            p_out += 1;
            j += 1;
        }
        k += 1;
    }
}

/// Each seed's result: `[status, checks, curves, tans]` as status 0 (done) or 1 (back to the CPU) and three
/// start, end pairs into the chunk's outputs (tans and points share their indices); a seed whose outputs reach
/// past a buffer's cap goes back to the CPU, and a seed back to the CPU has empty ranges
///
/// Grid: x over the chunk's seeds
#[kernel]
#[allow(clippy::too_many_arguments)]
pub unsafe fn fb_walk_results(
    n: u32,
    state: *const u32,
    surv_len: *const u32,
    n_checks: *const u32,
    n_pts: *const u32,
    n_curves: *const u32,
    totals: *const u32,
    check_cap: u32,
    pts_cap: u32,
    curve_cap: u32,
    results: *mut u32,
) {
    let s = thread_1d();
    if s >= n {
        return;
    }
    let out = unsafe { results.add(s as usize * RESULT_WORDS) };
    let st = unsafe { *state.add(s as usize) };
    let surv = st & SURVIVOR != 0;
    let cpu_early = st & CPU != 0 || (surv && unsafe { *surv_len.add((st & INDEX_BITS) as usize * SURV_WORDS) } != 0);
    // the end of seed s's run in a scanned array is the next seed's start, or the total after the last seed
    let end = |a: *const u32, total: usize| if s + 1 < n { unsafe { *a.add(s as usize + 1) } } else { unsafe { *totals.add(total) } };
    let (c0, c1) = (unsafe { *n_checks.add(s as usize) }, end(n_checks, T_CHECKS));
    let (p0, p1) = (unsafe { *n_pts.add(s as usize) }, end(n_pts, T_PTS));
    let (v0, v1) = (unsafe { *n_curves.add(s as usize) }, end(n_curves, T_CURVES));
    let fits = c1 <= check_cap && p1 <= pts_cap && v1 <= curve_cap;
    let done = !cpu_early && fits;
    let words = if done { [0, c0, c1, v0, v1, p0, p1] } else { [1, 0, 0, 0, 0, 0, 0] };
    let mut k = 0;
    while k < RESULT_WORDS {
        unroll_full!();
        unsafe { *out.add(k) = words[k] };
        k += 1;
    }
}

/// Point coverage bricks at new payloads: `list` holds `n` pairs of brick index and payload address, each payload
/// zeroed before this runs
///
/// Grid: x over the pairs
#[kernel]
pub unsafe fn fb_walk_cov_bricks(table: *mut u64, list: *const u64, n: u32) {
    let i = thread_1d();
    if i >= n {
        return;
    }
    let (brick, addr) = unsafe { (*list.add(i as usize * 2), *list.add(i as usize * 2 + 1)) };
    unsafe { write_volatile(table.add(brick as usize), addr) };
}

/// Apply committed coverage changes: `list` holds `n` pairs of brick index and `offset | value << 16`, every brick
/// already pointed at a payload
///
/// A batch may change one voxel twice and its pairs run in any order, so a value is applied by coverage's own
/// order: a voxel goes only from 0 to a direction (at least 256) to 1 (a count of 2), so 1 always wins and a
/// direction is set only over 0. The u32 holding the voxel is updated whole, by compare and swap, so its
/// neighbour's value is kept
///
/// Grid: x over the pairs
#[kernel]
pub unsafe fn fb_walk_cov_set(table: *const u64, list: *const u32, n: u32) {
    let i = thread_1d();
    if i >= n {
        return;
    }
    let (brick, packed) = unsafe { (*list.add(i as usize * 2), *list.add(i as usize * 2 + 1)) };
    let (offset, value) = (packed & 0xffff, packed >> 16);
    let addr = unsafe { read_volatile(table.add(brick as usize)) };
    let word = unsafe { (addr as *mut u32).add((offset >> 1) as usize) };
    let shift = (offset & 1) * 16;
    let a = unsafe { DeviceAtomicU32::from_ptr(word) };
    let mut cur = a.load(AtomicOrdering::Relaxed);
    loop {
        let old = (cur >> shift) & 0xffff;
        if old == value || (value != 1 && old != 0) {
            return;
        }
        let next = (cur & !(0xffff << shift)) | value << shift;
        match a.compare_exchange(cur, next, AtomicOrdering::Relaxed, AtomicOrdering::Relaxed) {
            Ok(_) => return,
            Err(now) => cur = now,
        }
    }
}
