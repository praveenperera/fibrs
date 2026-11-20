//! The Python environment fibrs runs in, where the shared libraries it loads at run time from other wheels live
//! (numpy's OpenBLAS, NVIDIA's nvCOMP)

use std::path::{Path, PathBuf};

/// The `site-packages` directories of the venv this executable was started from (`VENV/bin/avf-spline-generator`),
/// then of `$VIRTUAL_ENV`, each venv's in the order its `lib` directory lists them
///
/// Directories that do not exist are left in, so a caller's lookup inside them just fails
pub fn site_packages() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(arg0) = std::env::args_os().next() {
        let arg0 = PathBuf::from(arg0);
        let arg0 = if arg0.components().count() > 1 { Some(arg0) } else { which(&arg0) };
        if let Some(bin) = arg0.as_deref().and_then(Path::parent) {
            roots.push(bin.join(".."));
        }
    }
    if let Some(v) = std::env::var_os("VIRTUAL_ENV") {
        roots.push(PathBuf::from(v));
    }
    let mut sites = Vec::new();
    for root in roots {
        let Ok(libs) = std::fs::read_dir(root.join("lib")) else { continue };
        sites.extend(libs.flatten().map(|py| py.path().join("site-packages")));
    }
    sites
}

fn which(name: &Path) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(name)).find(|p| p.is_file())
}
