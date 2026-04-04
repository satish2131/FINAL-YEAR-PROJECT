import sqlite3
import os
import numpy as np


def init_db(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS criminals (
                 id INTEGER PRIMARY KEY,
                 name TEXT,
                 filepath TEXT,
                 encoding BLOB
                 )''')
    conn.commit()
    conn.close()


def add_criminal(db_path, name, filepath, encoding):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    # convert numpy array to bytes with shape metadata
    try:
        enc = np.asarray(encoding, dtype='float64')
        shape = np.array(enc.shape).astype('int32')
        blob = shape.tobytes() + enc.tobytes()
    except Exception:
        blob = None
    c.execute('INSERT INTO criminals (name, filepath, encoding) VALUES (?,?,?)', (name, filepath, blob))
    conn.commit()
    conn.close()


def _decode_encoding(blob):
    if blob is None:
        return None
    try:
        # face_recognition produces 1D arrays. shape.tobytes() for (128,) is exactly 4 bytes.
        if len(blob) > 4:
            return np.frombuffer(blob[4:], dtype='float64')
        return None
    except Exception:
        return None


def get_all_criminals(db_path):
    if not os.path.exists(db_path):
        return []
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT id, name, filepath, encoding FROM criminals')
    rows = c.fetchall()
    conn.close()
    out = []
    for r in rows:
        enc = _decode_encoding(r[3])
        out.append({'id': r[0], 'name': r[1], 'filepath': r[2], 'encoding': enc})
    return out
