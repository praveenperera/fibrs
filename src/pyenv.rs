//! The Python environment fibrs runs in, where the shared libraries it loads at run time from other wheels live
//! (numpy's OpenBLAS, NVIDIA's nvCOMP)

use std::path::{Path, PathBuf};

use libloading::Library;

/// The `site-packages` directories of the venv this executable was started from (`VENV/bin/avf-spline-generator`),
/// then of `$VIRTUAL_ENV`, each venv's in the order its `lib` directory lists them. On Windows a venv keeps the
/// executable in `Scripts` and its one `site-packages` in `Lib`
///
/// Directories that do not exist are left in, so a caller's lookup inside them just fails
pub fn site_packages() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Some(bin) = started_from() {
        roots.push(bin.join(".."));
    }
    if let Some(v) = std::env::var_os("VIRTUAL_ENV") {
        roots.push(PathBuf::from(v));
    }
    let mut sites = Vec::new();
    for root in roots {
        sites.extend(sites_in(&root));
    }
    sites
}

/// The directory of the path this executable was run by, not the file it resolves to, since the drop-in can be a
/// link into the venv from elsewhere
#[cfg(unix)]
fn started_from() -> Option<PathBuf> {
    let arg0 = PathBuf::from(std::env::args_os().next()?);
    let arg0 = if arg0.components().count() > 1 { Some(arg0) } else { which(&arg0) };
    arg0.as_deref().and_then(Path::parent).map(Path::to_path_buf)
}

/// The executable's own directory: on Windows the drop-in is a copy, and the command that ran it may leave out
/// the `.exe` a `PATH` lookup would need
#[cfg(windows)]
fn started_from() -> Option<PathBuf> {
    std::env::current_exe().ok()?.parent().map(Path::to_path_buf)
}

#[cfg(unix)]
fn which(name: &Path) -> Option<PathBuf> {
    std::env::split_paths(&std::env::var_os("PATH")?).map(|d| d.join(name)).find(|p| p.is_file())
}

/// `lib/python3.*/site-packages`, in the order the directory lists them
#[cfg(unix)]
fn sites_in(root: &Path) -> Vec<PathBuf> {
    let Ok(libs) = std::fs::read_dir(root.join("lib")) else { return Vec::new() };
    libs.flatten().map(|py| py.path().join("site-packages")).collect()
}

/// `Lib\site-packages`, where a venv or an installed Python keeps it, then `site-packages`, where a per-user
/// install (`%APPDATA%\Python\Python3*`) keeps it beside its `Scripts`
#[cfg(windows)]
fn sites_in(root: &Path) -> Vec<PathBuf> {
    vec![root.join("Lib").join("site-packages"), root.join("site-packages")]
}

/// Opens the shared library at `path`, or by a bare name, the one the system's loader finds
///
/// # Safety
///
/// Loading a library runs its initialisers
#[cfg(unix)]
pub unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    unsafe { Library::new(path) }
}

/// Opens the DLL at `path`, resolving the DLLs it imports from its own directory, then the executable's and
/// System32, since the binary takes the current directory and `PATH` out of the search at startup. A relative path
/// is taken from the current directory, as it names a file there; a bare name is found in the executable's
/// directory or System32, or reuses a DLL of that name already loaded
///
/// # Safety
///
/// Loading a library runs its initialisers
#[cfg(windows)]
pub unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::windows::{LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR};

    // the DLL's own directory can only be searched for a full path; with a bare name the flag is an error
    let bare = path.parent().is_none_or(|p| p.as_os_str().is_empty()) && !path.has_root();
    if bare {
        return unsafe { Library::new(path) };
    }

    // the restricted search would look for a relative path under its own directories, not the current one; a path
    // with no absolute form (an empty one) is left for the load to refuse
    let path = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let flags = LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS;
    unsafe { libloading::os::windows::Library::load_with_flags(&path, flags) }.map(Library::from)
}
