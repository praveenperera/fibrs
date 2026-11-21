//! A complete `.afv` read back: its frame and one family's fibres, as the v2-run `afvfile.read` gives them

use std::path::{Path, PathBuf};

use color_eyre::eyre::{Result, bail, eyre};
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde_json::Value;

use crate::seams::Point;

/// Bytes of one stored point: x, y, z as little-endian float64
const POINT_BYTES: usize = 24;

/// What a trace's metadata says about the volume it was traced in
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// The voxel size in µm (`vc_open_data_source_original_resolution`)
    pub voxel_um: f64,
    /// The whole volume's shape, z, y, x (`coordinate_base_shape_zyx`)
    pub volume_shape: [i64; 3],
    /// The native coordinate of the traced window's first voxel, x, y, z (`tracing.origin_xyz`)
    pub origin_xyz: [i64; 3],
    /// The scan the volume is, when the trace names it (`vc_open_data_coordinate_space`)
    pub coordinate_space: Option<String>,
}

/// An open, complete `.afv`
pub struct AfvReader {
    db: Connection,
    path: PathBuf,
}

impl AfvReader {
    /// Open `path` read-only; a file whose metadata does not mark it complete is refused
    pub fn open(path: &Path) -> Result<Self> {
        let db = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX)
            .map_err(|e| eyre!("{}: {e}", path.display()))?;
        let reader = Self { db, path: path.to_path_buf() };
        if reader.metadata("complete")? != Some(Value::Bool(true)) {
            bail!("{} is not a complete AFV", path.display());
        }
        Ok(reader)
    }

    /// The metadata value under `key`, parsed from its JSON
    pub fn metadata(&self, key: &str) -> Result<Option<Value>> {
        let text: Option<String> = self
            .db
            .query_row("SELECT value FROM metadata WHERE key = ?", [key], |r| r.get(0))
            .optional()
            .map_err(|e| eyre!("{}: {e}", self.path.display()))?;
        text.map(|t| serde_json::from_str(&t).map_err(|e| eyre!("{}: metadata {key}: {e}", self.path.display())))
            .transpose()
    }

    /// The volume and window the file's fibres were traced in
    pub fn frame(&self) -> Result<Frame> {
        let missing = |what: &str| eyre!("{}: no {what} in the metadata", self.path.display());
        let frame = self.metadata("frame")?.ok_or_else(|| missing("frame"))?;
        let tracing = self.metadata("tracing")?.ok_or_else(|| missing("tracing"))?;
        let voxel_um = frame["vc_open_data_source_original_resolution"]
            .as_f64()
            .ok_or_else(|| missing("frame.vc_open_data_source_original_resolution"))?;
        let triple = |v: &Value, what: &str| -> Result<[i64; 3]> {
            let items: Option<Vec<i64>> = v.as_array().map(|a| a.iter().filter_map(Value::as_i64).collect());
            items.and_then(|a| a.try_into().ok()).ok_or_else(|| missing(what))
        };
        let volume_shape = triple(&frame["coordinate_base_shape_zyx"], "frame.coordinate_base_shape_zyx")?;
        let origin_xyz = triple(&tracing["origin_xyz"], "tracing.origin_xyz")?;
        let coordinate_space = frame["vc_open_data_coordinate_space"].as_str().map(str::to_owned);
        Ok(Frame { voxel_um, volume_shape, origin_xyz, coordinate_space })
    }

    /// Every fibre of `family` ("V" or "H") in id order, as native x, y, z points
    ///
    /// Each block after a fibre's first repeats the point the block before it ends on; the repeats are dropped
    pub fn fibres(&self, family: &str) -> Result<Vec<Vec<Point>>> {
        let fail = |e: rusqlite::Error| eyre!("{}: {e}", self.path.display());
        let mut rows = self
            .db
            .prepare(
                "SELECT f.id, f.point_count, b.first_segment, b.points FROM fibers f \
                 LEFT JOIN blocks b ON b.fiber_id = f.id WHERE f.family = ? ORDER BY f.id, b.first_segment",
            )
            .map_err(fail)?;
        let mut rows = rows.query([family]).map_err(fail)?;
        let mut out: Vec<Vec<Point>> = Vec::new();
        // the fibre being read: its id and point count
        let mut current: Option<(i64, usize)> = None;
        while let Some(row) = rows.next().map_err(fail)? {
            let id: i64 = row.get(0).map_err(fail)?;
            let count: i64 = row.get(1).map_err(fail)?;
            let first: Option<i64> = row.get(2).map_err(fail)?;
            let Some(first) = first else {
                bail!("{}: fibre {id} has no blocks", self.path.display());
            };
            if current.map(|(at, _)| at) != Some(id) {
                self.check_last(current, &out)?;
                current = Some((id, count as usize));
                out.push(Vec::with_capacity(count as usize));
            }

            let points = out.last_mut().expect("a fibre is open");
            // a block's first segment is the index of its first point, which the block before it ends on
            let expected = points.len().saturating_sub(1);
            if first as usize != expected {
                bail!("{}: fibre {id}'s block at segment {first} does not follow its last", self.path.display());
            }
            let blob = row.get_ref(3).map_err(fail)?.as_blob().map_err(|e| eyre!("{}: {e}", self.path.display()))?;
            if blob.is_empty() || !blob.len().is_multiple_of(POINT_BYTES) {
                bail!("{}: fibre {id} has a block of {} bytes", self.path.display(), blob.len());
            }
            let stored: Vec<Point> =
                blob.as_chunks::<8>().0.as_chunks::<3>().0.iter().map(|p| p.map(f64::from_le_bytes)).collect();
            // a later block starts on the point the block before it ends on, which is dropped as a repeat only
            // when it is one, so a damaged boundary is refused rather than read as a different segment
            let skip = usize::from(first > 0);
            if skip == 1 && points.last() != stored.first() {
                bail!(
                    "{}: fibre {id}'s block at segment {first} does not start where the block before it ends",
                    self.path.display()
                );
            }
            points.extend_from_slice(&stored[skip..]);
        }
        self.check_last(current, &out)?;
        Ok(out)
    }

    /// That the fibre read last has the points its row counts
    fn check_last(&self, current: Option<(i64, usize)>, out: &[Vec<Point>]) -> Result<()> {
        let Some((id, count)) = current else { return Ok(()) };
        let got = out.last().map_or(0, Vec::len);
        if got != count {
            bail!("{}: fibre {id} has {got} points in its blocks, not {count}", self.path.display());
        }
        Ok(())
    }
}
