//! The records that pass between the CPU geometry and the GPU engine, one per decision
//!
//! Frames are 3x3 matrices `m[row][col]` in world (trace-voxel) xyz whose columns are the crop axes (u, v, f), as
//! villa's `item['frame']`; positions are trace-voxel xyz. Float widths follow villa's: geometry in f64, model
//! inputs in f32

/// A 3x3 frame, `m[row][col]`, columns (u, v, f)
pub type Mat3 = [[f64; 3]; 3];

/// Fibre family, as the frame model's embedding index: H 0, V 1
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// Horizontal
    H = 0,
    /// Vertical
    V = 1,
}

/// What the frame model needs for one decision (villa batched.py `FrameRequest`, one row)
#[derive(Clone, Debug)]
pub struct FrameRequest {
    /// The decision head, trace voxels
    pub pos: [f64; 3],
    /// The prior frame `frame_from_heading(prior)`, which orients the 32^3 crop
    pub frame: Mat3,
    /// `path_features(observed_path, pos, frame, 32)`: 32 x (u, v, f, valid), newest first, flattened
    pub paths: [f32; 128],
    /// The fibre family
    pub family: Family,
}

/// The frame model's output for one decision, in the request frame's local axes
#[derive(Clone, Copy, Debug)]
pub struct FrameResult {
    /// Unit heading (residual on +f)
    pub heading: [f32; 3],
    /// Unit sheet normal
    pub normal: [f32; 3],
}

/// Most tube points one decision carries (villa `polyline_points(..., 2048)`)
pub const TUBE_POINTS: usize = 2048;
/// Reference history points (villa `n_history`)
pub const HISTORY: usize = 128;
/// Path-geometry rows (28 arclength offsets plus the fibre start)
pub const GEOMETRY_ROWS: usize = 29;
/// Strip rows along the fibre (`strip_grid` f) and columns across it (v)
pub const STRIP_ROWS: usize = 36;
/// Strip columns across the fibre
pub const STRIP_COLS: usize = 25;

/// The strip model's inputs for one decision (villa batched.py `finish`, one row of the batch)
#[derive(Clone, Debug)]
pub struct StripRequest {
    /// The crop centre, trace voxels (the decision head)
    pub pos: [f64; 3],
    /// The final frame, which orients the 144x128x128 crop
    pub frame: Mat3,
    /// `hist_local`, newest first, crop-local
    pub hist: [[f32; 3]; HISTORY],
    /// History validity, 0 or 1
    pub hmask: [f32; HISTORY],
    /// `visible_seed`, crop-local (zero when masked)
    pub seed: [f32; 3],
    /// `visible_seed_mask`, 0 or 1
    pub seed_mask: f32,
    /// `visible_seed_age`
    pub seed_age: f32,
    /// `visible_seed_tangent`, crop-local
    pub seed_tangent: [f32; 3],
    /// Path-geometry features, 29 x 8
    pub path_geometry: [[f32; 8]; GEOMETRY_ROWS],
    /// Path-geometry row validity
    pub path_geometry_valid: [bool; GEOMETRY_ROWS],
    /// Tube points, crop-local (u, v, f), at most [`TUBE_POINTS`]
    pub tube: Vec<[f32; 3]>,
    /// `crop_orientation`: radial . u_xy, radial . v_xy, u_z, v_z, f_z
    pub orientation: [f32; 5],
}

/// The strip model's output for one decision: the last pass's heights and log error scales on the strip grid
#[derive(Clone, Debug)]
pub struct StripResult {
    /// Heights u(f, v), crop-local trace voxels
    pub heights: [[f32; STRIP_COLS]; STRIP_ROWS],
    /// Log error scale per grid point
    pub log_scale: [[f32; STRIP_COLS]; STRIP_ROWS],
}
