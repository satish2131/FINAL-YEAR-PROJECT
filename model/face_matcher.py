import os
import numpy as np
import face_recognition
from utils.database import get_all_criminals


def match_face(query_encoding, dataset_folder=None, db_path=None, top_k=5):
    """Match a query encoding to dataset images using fast vectorized numpy operations."""
    results = []

    if db_path:
        criminals = get_all_criminals(db_path)

        # Collect valid entries and their encodings into lists
        valid_entries = []
        encoding_list = []
        for c in criminals:
            enc = c['encoding']
            if enc is None:
                continue
            valid_entries.append(c)
            encoding_list.append(enc)

        if encoding_list:
            # Stack into a single (N, 128) numpy matrix for vectorized distance
            all_encodings = np.array(encoding_list)
            # Compute ALL distances in one fast numpy call
            distances = face_recognition.face_distance(all_encodings, query_encoding)

            # Get top_k indices using argpartition (O(n) instead of O(n log n) full sort)
            k = min(top_k, len(distances))
            top_indices = np.argpartition(distances, k)[:k]
            # Sort only those k for final ordering
            top_indices = top_indices[np.argsort(distances[top_indices])]

            for idx in top_indices:
                c = valid_entries[idx]
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
