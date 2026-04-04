import sys
import os

# Ensure the root directory is in the Python path so it can find the 'utils' package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import face_recognition
from utils.database import init_db, add_criminal


def add_dataset_to_db(dataset_folder, db_path):
    init_db(db_path)
    for root, _, files in os.walk(dataset_folder):
        for fname in files:
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            fpath = os.path.join(root, fname)
            try:
                img = face_recognition.load_image_file(fpath)
                encs = face_recognition.face_encodings(img)
                if len(encs) == 0:
                    print(f'No face found in {fname}, skipping')
                    continue
                
                parent_dir = os.path.basename(root)
                if parent_dir == os.path.basename(os.path.normpath(dataset_folder)):
                    name = os.path.splitext(fname)[0]
                else:
                    name = parent_dir.replace('_', ' ')
                
                add_criminal(db_path, name, fpath, encs[0])
                print(f'Added {fname} to DB as {name}')
            except Exception as e:
                print(f'Failed {fname}: {e}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, help='Path to dataset folder')
    p.add_argument('--db', required=True, help='Path to output db file')
    args = p.parse_args()
    add_dataset_to_db(args.dataset, args.db)
