//! Records which volkit this build links, from Cargo.lock, for `fibrs predict`'s provenance: `volkit <version>`
//! from crates.io, or `volkit <version> @ <commit>` from a git source

use std::fs;

fn main() {
    println!("cargo::rerun-if-changed=Cargo.lock");
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
