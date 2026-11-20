//! The `fb_walk_*` kernels' buffer layout, shared with the host's wrapper (`src/gpu/cuda/walk.rs`) through
//! `#[path]`, so both size and read the buffers alike

/// u64 words per check record: `brick | offset << 32 | seen << 48`, the kind, then the tangent's three f64 bits
pub const CHECK_WORDS: usize = 5;
/// Most prefix checks of one seed: `is_covered` and two `covered_same_dir`
pub const PREFIX_CHECKS: u32 = 3;
/// u32 per survivor in `surv_len`: status (0 traced, 1 back to the CPU), nf, nb, checks, accepted spans, accepted
/// points, then two spare
pub const SURV_WORDS: usize = 8;
/// f64 per survivor in `surv_cu`: the recentred start point and its aligned direction
pub const CU_WORDS: usize = 6;
/// Largest step cap per half the kernels take
pub const MAX_CAP: u32 = 1024;
/// Room for accepted spans per survivor: each has at least 8 of at most `2 * cap + 1` points and neighbours share
/// one, so `(2 * cap + 1) / 7 + 1` is enough for any cap up to [`MAX_CAP`]
pub const SPAN_CAP: u32 = (2 * MAX_CAP + 1) / 7 + 1;
/// u32 per point in the mark masks: one bit per voxel of the 7³ box `mark_point` visits
pub const MASK_WORDS: u32 = 343u32.div_ceil(32);
/// u32 per seed result: status (0 done, 1 back to the CPU), then the start and end of its checks, curves, tans
/// and marks
pub const RESULT_WORDS: usize = 9;
/// Slots of the totals array
pub const TOTALS: usize = 8;
pub const T_CHECKS: usize = 0;
pub const T_PTS: usize = 1;
pub const T_CURVES: usize = 2;
pub const T_MARKS: usize = 3;
pub const T_SURV: usize = 4;
pub const T_SEEDS: usize = 5;
