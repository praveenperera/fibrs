//! A build probe: proves the PTX pipeline and the host's launch checks end to end

use cuda_device::{DisjointSlice, kernel, thread};

/// `out[i] = a[i] * scale + i`
#[kernel]
pub fn sp_probe(a: &[f32], scale: f32, mut out: DisjointSlice<f32>) {
    let t = thread::index_1d();
    let i = t.get();
    if let Some(o) = out.get_mut(t) {
        *o = a[i] * scale + i as f32;
    }
}
