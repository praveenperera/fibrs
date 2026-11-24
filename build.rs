//! Records which volkit this build links, from Cargo.lock, for `fibrs predict`'s provenance: `volkit <version>`
//! from crates.io, or `volkit <version> @ <commit>` from a git source. Also sets `fibrs_engine` to the GPU engine
//! the target builds in: `cuda` on Linux and Windows x86_64, `metal` on macOS arm64, unset elsewhere

use std::fs;

fn main() {
    println!("cargo::rerun-if-changed=Cargo.lock");
    println!("cargo::rustc-check-cfg=cfg(fibrs_engine, values(none(), \"cuda\", \"metal\"))");
    // the target's, not the host's: build scripts see the target through CARGO_CFG_*
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let engine = match (os.as_str(), arch.as_str()) {
        ("linux" | "windows", "x86_64") => Some("cuda"),
        ("macos", "aarch64") => Some("metal"),
        _ => None,
    };
    // bare `fibrs_engine` says some engine is built in, `fibrs_engine = "..."` which one
    if let Some(engine) = engine {
        println!("cargo::rustc-cfg=fibrs_engine");
        println!("cargo::rustc-cfg=fibrs_engine=\"{engine}\"");
    }
    let lock = fs::read_to_string("Cargo.lock").unwrap_or_default();
    // the package entry is `name = "volkit"`, then `version = ...`, then `source = ...`: a registry URL, or
    // `git+<url>?rev=<rev>#<commit>`
    let mut lines = lock.lines().skip_while(|l| *l != "name = \"volkit\"").skip(1);
    let field = |l: Option<&str>, key: &str| {
        l.and_then(|l| l.strip_prefix(key)).map(|v| v.trim_matches('"').to_owned()).unwrap_or_default()
    };
    let version = field(lines.next(), "version = ");
    let source = field(lines.next(), "source = ");
    let volkit = match source.strip_prefix("git+").and_then(|s| s.rsplit_once('#')) {
        Some((_, commit)) => format!("volkit {version} @ {}", &commit[..commit.len().min(9)]),
        None => format!("volkit {version}"),
    };
    println!("cargo::rustc-env=FIBRS_VOLKIT={volkit}");
}
