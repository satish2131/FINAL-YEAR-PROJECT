"""
reindex_db.py - Phase 2 Re-indexing Script (InsightFace buffalo_l)
===================================================================
Run this ONCE after upgrading feature_extractor.py to InsightFace buffalo_l.

It wipes ALL existing encodings from the database and regenerates them
using the best available encoder (InsightFace -> DeepFace ArcFace fallback).

Usage:
    python reindex_db.py              # re-index everything
    python reindex_db.py --resume     # skip records that already have NEW encodings

Notes:
  - First run downloads InsightFace buffalo_l model weights (~200 MB)
  - 13,166 images will take ~30-60 minutes (CPU only)
  - GPU (CUDA) will be ~5x faster if available
"""

import os
import sys
import sqlite3
import argparse
import numpy as np

# Force UTF-8 output on Windows to avoid charmap codec errors
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Make sure root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.feature_extractor import get_face_encoding_from_image

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "criminals.db")

# InsightFace 512-D blob size: 4 bytes (shape int32) + 512*8 bytes (float64) = 4100
_INSIGHTFACE_BLOB_SIZE = 4100
# DeepFace ArcFace also 512-D -> same blob size
# Old dlib 128-D: 4 + 128*8 = 1028 bytes


def encode_blob(encoding: np.ndarray) -> bytes:
    """Pack a float64 ndarray into compact blob format."""
    enc = np.asarray(encoding, dtype="float64")
    shape = np.array(enc.shape).astype("int32")
    return shape.tobytes() + enc.tobytes()


def decode_blob(blob: bytes):
    """Decode blob back to ndarray. Returns None if invalid."""
    if blob is None or len(blob) < 4:
        return None
    try:
        return np.frombuffer(blob[4:], dtype="float64")
    except Exception:
        return None


def is_new_encoding(blob: bytes) -> bool:
    """Return True if the blob is already a 512-D (InsightFace/ArcFace) encoding."""
    return blob is not None and len(blob) == _INSIGHTFACE_BLOB_SIZE


def reindex(resume: bool = False):
    if not os.path.exists(DB_PATH):
        print(f"[Reindex] ERROR: DB not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    if resume:
        # Only re-encode records that DON'T have a 512-D encoding yet
        c.execute("SELECT id, name, filepath, encoding FROM criminals")
        all_rows = c.fetchall()
        rows = [(rid, name, fp, enc) for rid, name, fp, enc in all_rows if not is_new_encoding(enc)]
        print(f"[Reindex] RESUME mode: {len(rows)} records need re-encoding "
              f"(skipping {len(all_rows) - len(rows)} already 512-D).")
    else:
        c.execute("SELECT id, name, filepath FROM criminals")
        rows_raw = c.fetchall()
        rows = [(rid, name, fp, None) for rid, name, fp in rows_raw]
        print(f"[Reindex] FULL mode: {len(rows)} records to re-encode with InsightFace buffalo_l.")

    total   = len(rows)
    success = 0
    failed  = 0
    skipped = 0

    for i, row in enumerate(rows, start=1):
        rid, name, filepath = row[0], row[1], row[2]

        # Normalise filepath
        if not os.path.isabs(filepath):
            filepath = os.path.join(os.path.dirname(__file__), filepath)

        if not os.path.exists(filepath):
            print(f"  [{i}/{total}] SKIP (missing): {name} -> {os.path.basename(filepath)}")
            skipped += 1
            continue

        try:
            enc = get_face_encoding_from_image(filepath)
            if enc is None:
                print(f"  [{i}/{total}] NO FACE: {name} ({os.path.basename(filepath)})")
                failed += 1
                continue

            blob = encode_blob(enc)
            c.execute("UPDATE criminals SET encoding=? WHERE id=?", (blob, rid))
            conn.commit()
            success += 1
            # Print progress every 50 records or always show first 10
            if i <= 10 or i % 50 == 0:
                print(f"  [{i}/{total}] OK ({int(enc.shape[0])}-D): {name}")

        except Exception as e:
            print(f"  [{i}/{total}] ERROR {name}: {e}")
            failed += 1

    conn.close()
    print(f"\n[Reindex] Done!")
    print(f"  OK     : {success} encoded")
    print(f"  FAILED : {failed} (no face or error)")
    print(f"  SKIPPED: {skipped} (missing file)")
    print(f"\n[Reindex] Restart app.py to load new InsightFace embeddings into memory.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-index face DB with InsightFace buffalo_l.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip records that already have 512-D embeddings (saves time if partially done)."
    )
    args = parser.parse_args()
    reindex(resume=args.resume)
