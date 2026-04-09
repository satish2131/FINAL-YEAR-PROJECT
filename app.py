import os
import time
import base64
from datetime import datetime
import numpy as np
import face_recognition
from flask import Flask, request, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from model.feature_extractor import get_face_encoding_from_image
from utils.image_processing import preprocess_image
from utils.database import (
    get_all_criminals,
    get_db_stats,
    search_criminals_by_name,
    get_paginated_criminals,
    delete_criminal,
    init_db,
    add_criminal,
    populate_gender_column,
)

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
DB_PATH = os.path.join(BASE_DIR, 'database', 'criminals.db')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── In-memory cache ──────────────────────────────────────────────────────────
all_criminals = []
_enc_matrix = None          # (N, 128) full matrix
_gender_index = {}          # {'male': [indices], 'female': [indices], 'unknown': [indices]}

# ── Search history (in-memory, persists while server runs) ───────────────────
search_history = []         # list of dicts, newest first
_search_counter = 0         # auto-increment sketch ID


def _load_encodings():
    """Load (or reload) all face encodings from SQLite into RAM with gender index."""
    global all_criminals, _enc_matrix, _gender_index
    print('[LFIM] Loading encodings from database into memory...')
    try:
        rows = get_all_criminals(DB_PATH)
        valid = [(r, r['encoding']) for r in rows if r['encoding'] is not None and len(r['encoding']) == 128]
        if valid:
            all_criminals = [r for r, _ in valid]
            _enc_matrix = np.array([enc for _, enc in valid], dtype=np.float64)
            # Build gender index: map gender -> list of row indices
            _gender_index = {'male': [], 'female': [], 'unknown': []}
            for i, r in enumerate(all_criminals):
                g = (r.get('gender') or 'unknown').lower()
                _gender_index.setdefault(g, []).append(i)
            counts = {k: len(v) for k, v in _gender_index.items()}
            print(f'[LFIM] Loaded {len(all_criminals)} encodings. Gender dist: {counts}')
        else:
            all_criminals = []
            _enc_matrix = None
            _gender_index = {}
            print('[LFIM] WARNING: No valid encodings found in database!')
    except Exception as e:
        print(f'[LFIM] ERROR loading database: {e}')
        all_criminals = []
        _enc_matrix = None
        _gender_index = {}


# Run gender migration once, then load cache
init_db(DB_PATH)
populate_gender_column(DB_PATH)
_load_encodings()
# ─────────────────────────────────────────────────────────────────────────────


def fast_match(query_encoding, top_k=5, gender_filter='any'):
    """Vectorized distance search with strict gender filtering.
    gender_filter: 'any' | 'male' | 'female'
    """
    if _enc_matrix is None or len(_enc_matrix) == 0:
        return []

    # Select subset indices — STRICT: only the requested gender, no unknowns mixed in
    if gender_filter in ('male', 'female'):
        idx_list = _gender_index.get(gender_filter, [])
        if not idx_list:
            return []
        subset_idx = np.array(idx_list, dtype=np.int64)
        sub_matrix = _enc_matrix[subset_idx]
        sub_criminals = [all_criminals[i] for i in subset_idx]
    else:
        sub_matrix = _enc_matrix
        sub_criminals = all_criminals
        subset_idx = None

    distances = face_recognition.face_distance(sub_matrix, query_encoding)
    k = min(top_k, len(distances))
    top_idx = np.argpartition(distances, k)[:k]
    top_idx = top_idx[np.argsort(distances[top_idx])]
    results = []
    for idx in top_idx:
        c = sub_criminals[idx]
        rel_file = c['filepath'].replace('\\', '/')
        if 'dataset/' in rel_file:
            rel_file = rel_file.split('dataset/', 1)[-1]
        else:
            rel_file = os.path.basename(c['filepath'])
        results.append({
            'id': c.get('id'),
            'name': c.get('name', os.path.basename(c['filepath'])),
            'file': rel_file,
            'distance': float(distances[idx]),
            'gender': c.get('gender', 'unknown'),
        })
    return results


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ── Core routes ───────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return redirect(url_for('admin_portal'))


@app.route('/admin')
def admin_portal():
    return send_from_directory(BASE_DIR, 'admin_dashboard.html')


@app.route('/interactive')
def interactive():
    return send_from_directory(BASE_DIR, 'facial_identity_interactive_parts.html')


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/dataset/<path:filename>')
def dataset_file(filename):
    return send_from_directory(DATASET_FOLDER, filename)


@app.route('/assets/<path:filename>')
def serve_asset(filename):
    return send_from_directory(BASE_DIR, filename)


# ── Face-matching routes ──────────────────────────────────────────────────────

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        preprocess_image(path)
        enc = get_face_encoding_from_image(path)
        if enc is None:
            return jsonify({'error': 'No face found in uploaded image.'}), 400
        matches = fast_match(enc, top_k=5)
        return jsonify({'query': filename, 'matches': matches})
    return redirect(url_for('index'))


@app.route('/save_composite', methods=['POST'])
def save_composite():
    """Receive a base64 PNG from the UI, save it, run gender-filtered in-memory matching, return JSON."""
    global _search_counter
    data = None
    if request.is_json:
        payload = request.get_json()
        data = payload.get('image')
        gender_filter = payload.get('gender', 'any')
    else:
        data = request.form.get('image')
        gender_filter = request.form.get('gender', 'any')
    if not data:
        return jsonify({'error': 'No image received'}), 400
    header, b64 = data.split(',', 1) if ',' in data else (None, data)
    try:
        img_bytes = base64.b64decode(b64)
    except Exception:
        return jsonify({'error': 'Invalid image data'}), 400
    fname = f'composite_{int(time.time())}.png'
    out_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    with open(out_path, 'wb') as f:
        f.write(img_bytes)
    enc = get_face_encoding_from_image(out_path)
    if enc is None:
        # Record failed search in history
        _search_counter += 1
        search_history.insert(0, {
            'id': f'SKT-{_search_counter:03d}',
            'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'gender': gender_filter.capitalize() if gender_filter != 'any' else 'Any',
            'status': 'No Match',
            'matchCount': 0,
            'topMatch': None,
        })
        return jsonify({'query': fname, 'matches': [], 'error': 'No face detected in composite sketch'})
    matches = fast_match(enc, top_k=5, gender_filter=gender_filter)

    # ── Record this search in history ─────────────────────────────────────
    _search_counter += 1
    top = None
    if matches:
        conf = max(0, min(100, round((1 - matches[0]['distance']) * 100, 1)))
        top = {'name': matches[0]['name'], 'confidence': conf}
    search_history.insert(0, {
        'id': f'SKT-{_search_counter:03d}',
        'date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'gender': gender_filter.capitalize() if gender_filter != 'any' else 'Any',
        'status': 'Matched' if matches else 'No Match',
        'matchCount': len(matches),
        'topMatch': top,
        'matches': matches,              # full match list for detail view
    })
    # ─────────────────────────────────────────────────────────────────────

    return jsonify({'query': fname, 'matches': matches, 'gender_filter': gender_filter})


# ── SQLite Database API ───────────────────────────────────────────────────────

@app.route('/api/search/history', methods=['GET'])
def api_search_history():
    """Return the search history (newest first). Used by the admin dashboard."""
    return jsonify({
        'total': len(search_history),
        'matched': sum(1 for h in search_history if h['status'] == 'Matched'),
        'history': search_history,
    })


@app.route('/api/db/stats', methods=['GET'])
def api_db_stats():
    """Return high-level statistics about the SQLite database."""
    stats = get_db_stats(DB_PATH)
    stats['in_memory'] = len(all_criminals)
    stats['cache_ready'] = _enc_matrix is not None
    return jsonify(stats)


@app.route('/api/db/records', methods=['GET'])
def api_db_records():
    """Return a paginated list of records (no encoding blobs).
    Query params: page (int), per_page (int, max 100)
    """
    try:
        page = max(1, int(request.args.get('page', 1)))
        per_page = min(100, max(1, int(request.args.get('per_page', 20))))
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid pagination params'}), 400

    records, total = get_paginated_criminals(DB_PATH, page=page, per_page=per_page)
    # Resolve relative filepath for the dataset image
    enriched = []
    for r in records:
        rel = r['filepath'].replace('\\', '/')
        if 'dataset/' in rel:
            rel = rel.split('dataset/', 1)[-1]
        else:
            rel = os.path.basename(r['filepath'])
        enriched.append({
            'id': r['id'],
            'name': r['name'],
            'file': rel,
        })
    return jsonify({
        'page': page,
        'per_page': per_page,
        'total': total,
        'pages': (total + per_page - 1) // per_page,
        'records': enriched,
    })


@app.route('/api/db/search', methods=['GET'])
def api_db_search():
    """Text search over the 'name' column.
    Query params: q (str), limit (int, max 50)
    """
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify({'error': 'Missing query param: q'}), 400
    try:
        limit = min(50, max(1, int(request.args.get('limit', 20))))
    except (ValueError, TypeError):
        limit = 20
    rows = search_criminals_by_name(DB_PATH, q, limit=limit)
    enriched = []
    for r in rows:
        rel = r['filepath'].replace('\\', '/')
        if 'dataset/' in rel:
            rel = rel.split('dataset/', 1)[-1]
        else:
            rel = os.path.basename(r['filepath'])
        enriched.append({'id': r['id'], 'name': r['name'], 'file': rel})
    return jsonify({'query': q, 'count': len(enriched), 'results': enriched})


@app.route('/api/db/records/<int:record_id>', methods=['DELETE'])
def api_db_delete(record_id):
    """Delete a single record by id and refresh the in-memory cache."""
    deleted = delete_criminal(DB_PATH, record_id)
    if not deleted:
        return jsonify({'error': f'Record {record_id} not found'}), 404
    _load_encodings()   # refresh RAM cache
    return jsonify({'deleted': record_id, 'in_memory': len(all_criminals)})


@app.route('/api/db/reload', methods=['POST'])
def api_db_reload():
    """Hot-reload the in-memory encoding cache from SQLite (e.g. after batch import)."""
    _load_encodings()
    return jsonify({
        'status': 'reloaded',
        'in_memory': len(all_criminals),
        'cache_ready': _enc_matrix is not None,
    })


@app.route('/api/db/add', methods=['POST'])
def api_db_add():
    """Add a new face image to the database via file upload.
    Form fields: file (required), name (optional – defaults to filename stem)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Unsupported file type'}), 400

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    enc = get_face_encoding_from_image(save_path)
    if enc is None:
        os.remove(save_path)
        return jsonify({'error': 'No face detected in the uploaded image'}), 400

    name = request.form.get('name', '').strip() or os.path.splitext(filename)[0]
    init_db(DB_PATH)
    add_criminal(DB_PATH, name, save_path, enc)
    _load_encodings()   # refresh RAM cache

    return jsonify({
        'status': 'added',
        'name': name,
        'file': filename,
        'in_memory': len(all_criminals),
    }), 201


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    app.run(debug=False, threaded=True)
