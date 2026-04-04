import os
import time
import base64
import numpy as np
import face_recognition
from flask import Flask, request, redirect, url_for, render_template_string, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from model.feature_extractor import get_face_encoding_from_image
from utils.image_processing import preprocess_image
from utils.database import get_all_criminals

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
DATASET_FOLDER = os.path.join(BASE_DIR, 'dataset')
DB_PATH = os.path.join(BASE_DIR, 'database', 'criminals.db')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ── Pre-load all encodings at startup ──────────────────────────────────────
print('[LFIM] Loading encodings from database into memory...')
_all_criminals = []
_enc_matrix = None   # numpy (N, 128) float64 array

try:
    rows = get_all_criminals(DB_PATH)
    valid = [(r, r['encoding']) for r in rows if r['encoding'] is not None and len(r['encoding']) == 128]
    if valid:
        _all_criminals = [r for r, _ in valid]
        _enc_matrix = np.array([enc for _, enc in valid], dtype=np.float64)
        print(f'[LFIM] Loaded {len(_all_criminals)} face encodings into RAM  shape={_enc_matrix.shape}')
    else:
        print('[LFIM] WARNING: No valid encodings found in database!')
except Exception as e:
    print(f'[LFIM] ERROR loading database: {e}')
# ───────────────────────────────────────────────────────────────────────────


def fast_match(query_encoding, top_k=5):
    """Run vectorized cosine distance against in-memory encoding matrix."""
    if _enc_matrix is None or len(_enc_matrix) == 0:
        return []
    distances = face_recognition.face_distance(_enc_matrix, query_encoding)
    k = min(top_k, len(distances))
    top_idx = np.argpartition(distances, k)[:k]
    top_idx = top_idx[np.argsort(distances[top_idx])]
    results = []
    for idx in top_idx:
        c = _all_criminals[idx]
        rel_file = c['filepath'].replace('\\', '/')
        if 'dataset/' in rel_file:
            rel_file = rel_file.split('dataset/', 1)[-1]
        else:
            rel_file = os.path.basename(c['filepath'])
        results.append({
            'name': c.get('name', os.path.basename(c['filepath'])),
            'file': rel_file,
            'distance': float(distances[idx])
        })
    return results


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return redirect(url_for('interactive'))


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


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/dataset/<path:filename>')
def dataset_file(filename):
    return send_from_directory(DATASET_FOLDER, filename)


@app.route('/save_composite', methods=['POST'])
def save_composite():
    """Receive a base64 PNG from the UI, save it, run in-memory matching, return JSON."""
    data = None
    if request.is_json:
        payload = request.get_json()
        data = payload.get('image')
    else:
        data = request.form.get('image')
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
        return jsonify({'query': fname, 'matches': [], 'error': 'No face detected in composite sketch'})
    matches = fast_match(enc, top_k=5)
    return jsonify({'query': fname, 'matches': matches})


@app.route('/interactive')
def interactive():
    return send_from_directory(BASE_DIR, 'facial_identity_interactive_parts.html')


if __name__ == '__main__':
    app.run(debug=False, threaded=True)

