import traceback
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model.face_matcher import match_face
import numpy as np

# create dummy enc
query = np.zeros((128,), dtype=np.float64)
try:
    print('running match_face...')
    matches = match_face(query, db_path='database/criminals.db', top_k=5)
    print('SUCCESS! matches:', len(matches))
except Exception as e:
    print('ERROR IN MATCH_FACE:')
    traceback.print_exc()
