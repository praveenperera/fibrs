//! `fibrs predict`: AFV fibre model V2 (Qualzz20/afv_fiber_9um_v2) probabilities for a CT, natively on an
//! NVIDIA GPU
//!
//! The same command line, output group and records as villa `lasagna/qualzz_inference.py` and fibrs's band
//! reference: the CT is cut into 128^3 tiles on a stride-96 lattice, grouped into xy columns of the output
//! (`contract`); the tiles are read ahead on reader threads and batched (`feed`); volkit's `AfvV2` normalises,
//! runs the network and blends and finalises each column on the GPU (`engine`); one writer thread stores the
//! columns and their progress records (`out`). `--mode villa` swaps the air rule for villa's storage-based one
//! (`villa`)

mod contract;
mod ct;
mod engine;
mod feed;
mod out;
mod run;
mod villa;

use std::ffi::OsString;

pub use run::main;

/// Whether `argv` asks for `fibrs predict`. Run as the drop-in `avf-spline-generator`, the binary keeps the
/// reference CLI, which has no subcommands
pub fn requested(argv: &[OsString]) -> bool {
    let Some(arg0) = argv.first() else { return false };
    requested_bytes(arg0.as_encoded_bytes(), argv.get(1).map(|a| a.as_encoded_bytes()))
}

/// [`requested`] on the raw bytes of the first two arguments, for the global allocator, which decides before
/// `std::env::args` can be collected
pub fn requested_bytes(arg0: &[u8], arg1: Option<&[u8]>) -> bool {
    !crate::run::dropin(arg0) && arg1 == Some(b"predict".as_slice())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::requested_bytes;

    /// The allocator's check takes the drop-in name as `Path::file_name` does, so it agrees with the dispatch in
    /// `main` on every command line
    #[test]
    fn requested_bytes_names_the_dropin_as_path_does() {
        let cases: [(&str, Option<&str>, bool); 8] = [
            ("fibrs", Some("predict"), true),
            ("/venv/bin/fibrs", Some("predict"), true),
            ("fibrs", Some("trace"), false),
            ("fibrs", None, false),
            ("avf-spline-generator", Some("predict"), false),
            ("/opt/avf-spline-generator/.", Some("predict"), false),
            ("/opt/avf-spline-generator//", Some("predict"), false),
            ("/opt/avf-spline-generator/..", Some("predict"), true),
        ];

        // Windows paths, and the extension the system adds, in any case
        #[cfg(windows)]
        let cases = [
            &cases[..],
            &[
                (r"C:\venv\Scripts\fibrs.exe", Some("predict"), true),
                (r"C:\venv\Scripts\avf-spline-generator.exe", Some("predict"), false),
                (r"C:\venv\Scripts\AVF-SPLINE-GENERATOR.EXE", Some("predict"), false),
                ("avf-spline-generator.EXE", Some("predict"), false),
                ("C:avf-spline-generator.exe", Some("predict"), false),
                (r"C:\opt\avf-spline-generator\.", Some("predict"), false),
                ("avf-spline-generator.exe.exe", Some("predict"), true),
            ],
        ]
        .concat();

        for (arg0, arg1, want) in cases {
            let name = Path::new(arg0).file_name().map(|n| n.to_string_lossy().into_owned());
            #[cfg(windows)]
            let name = name.map(|n| {
                let n = n.to_ascii_lowercase();
                n.strip_suffix(".exe").map(str::to_owned).unwrap_or(n)
            });
            let dropin = name.as_deref() == Some("avf-spline-generator");
            assert_eq!(!dropin && arg1 == Some("predict"), want, "{arg0} {arg1:?}");
            assert_eq!(requested_bytes(arg0.as_bytes(), arg1.map(str::as_bytes)), want, "{arg0} {arg1:?}");
        }
    }
}
