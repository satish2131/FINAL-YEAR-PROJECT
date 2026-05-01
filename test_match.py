import os
import sqlite3
import numpy as np
from model.feature_extractor import get_face_encoding_from_image

# Load all encodings from DB
DB_PATH = os.path.join(os.path.dirname(__file__), 'database', 'criminals.db')

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("SELECT name, filepath, encoding FROM criminals WHERE encoding IS NOT NULL")
db_rows = c.fetchall()
conn.close()

db_encs = []
db_names = []
for name, fp, enc_blob in db_rows:
    enc = np.frombuffer(enc_blob, dtype=np.float64)
    if len(enc) == 512:
        db_encs.append(enc)
        db_names.append(name)

db_encs = np.array(db_encs)

print(f"Loaded {len(db_encs)} 512-D DB encodings.")

import glob
sketch_paths = glob.glob(r"c:\Users\dsaiv\OneDrive\Desktop\final year project\Latent_Facial_Identity_Mapping\uploads\composite_*.png")
print(f"Found {len(sketch_paths)} sketches.")

for path in sketch_paths:
    from utils.image_processing import preprocess_image
    print(f"\nProcessing {os.path.basename(path)}")
    # simulate what app.py does
    import tempfile, shutil
    temp_path = os.path.join(tempfile.gettempdir(), os.path.basename(path))
    shutil.copy(path, temp_path)
    preprocess_image(temp_path)
    
    enc = get_face_encoding_from_image(temp_path)
    if enc is None:
        print("Fail to encode.")
        continue
        
    distances = 1.0 - (db_encs @ enc)
    min_idx = np.argmin(distances)
    print(f"Top DB match for this sketch: {db_names[min_idx]} with Cosine Distance {distances[min_idx]:.3f}")
    
    # print top 3
    top3_idx = np.argsort(distances)[:3]
    for idx in top3_idx:
        print(f"  {db_names[idx]} \t Dist: {distances[idx]:.3f}")
