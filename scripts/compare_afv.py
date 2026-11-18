"""Compare two AFV files table by table, ignoring only the per-run metadata uuid

The AFV writer stores a fresh uuid4 in metadata on every run, and SQLite page layout may differ between
library versions, so file hashes cannot show parity. This compares the logical content instead: every
metadata value except uuid, every fibers row, every block blob and every spatial-index row, in id order.
Exit status 0 means identical content; the JSON line says where the first difference is otherwise

Usage: compare_afv.py A.afv B.afv
"""

import hashlib
import json
import sqlite3
import sys

QUERIES = {
    "fibers": "SELECT * FROM fibers ORDER BY id",
    "blocks": "SELECT * FROM blocks ORDER BY id",
    "block_bounds": "SELECT * FROM block_bounds ORDER BY id",
}


def open_ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def metadata(db):
    meta = {k: json.loads(v) for k, v in db.execute("SELECT key, value FROM metadata")}
    meta.pop("uuid", None)
    return meta


def main():
    a, b = open_ro(sys.argv[1]), open_ro(sys.argv[2])
    report = {"identical": True}
    for pragma in ("application_id", "user_version"):
        va = a.execute(f"PRAGMA {pragma}").fetchone()[0]
        vb = b.execute(f"PRAGMA {pragma}").fetchone()[0]
        if va != vb:
            report.update(identical=False, first_difference=f"pragma {pragma}: {va} != {vb}")
    if report["identical"] and metadata(a) != metadata(b):
        report.update(identical=False, first_difference="metadata", meta_a=metadata(a), meta_b=metadata(b))
    schema = "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY name"
    if report["identical"] and a.execute(schema).fetchall() != b.execute(schema).fetchall():
        report.update(identical=False, first_difference="schema")
    for table, query in QUERIES.items():
        ha, hb = hashlib.sha256(), hashlib.sha256()
        # zip stops at the shorter table; the counts below catch a length difference
        for row, (ra, rb) in enumerate(zip(a.execute(query), b.execute(query), strict=False)):
            ha.update(repr(ra).encode())
            hb.update(repr(rb).encode())
            if ra != rb and report["identical"]:
                report.update(identical=False, first_difference=f"{table} row {row}: {ra[:4]} != {rb[:4]}")
        na = a.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        nb = b.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        if na != nb and report["identical"]:
            report.update(identical=False, first_difference=f"{table} count {na} != {nb}")
        report[table] = {"rows_a": na, "rows_b": nb, "sha_a": ha.hexdigest()[:16], "sha_b": hb.hexdigest()[:16]}
    print(json.dumps(report))
    raise SystemExit(0 if report["identical"] else 1)


if __name__ == "__main__":
    main()
