import face_recognition
import numpy as np


def get_face_encoding_from_image(image_path):
    """Load an image and return the first face encoding found, or None."""
    try:
        img = face_recognition.load_image_file(image_path)
        encs = face_recognition.face_encodings(img)
        if len(encs) == 0:
            return None
        return encs[0]
    except Exception:
        return None
