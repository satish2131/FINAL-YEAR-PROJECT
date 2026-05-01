import numpy as np
import cv2
import os

# Suppress TF/Keras verbose output before importing DeepFace
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from deepface import DeepFace

# ── Model selection — DeepFace ArcFace (Phase 1) ──
_MODEL_NAME = "ArcFace"
_DETECTOR   = "opencv"    # Switched to opencv (Much faster on CPU than retinaface)

# Load model ONCE globally (Significant speed boost for batch processing)
MODEL = DeepFace.build_model(_MODEL_NAME)


def get_face_encoding_from_image(image_path):
    """
    Load an image and return a 512-D ArcFace face embedding, or None if no face found.

    Uses DeepFace with ArcFace model and RetinaFace detector.
    Falls back to skip detector for composite/sketch images where RetinaFace
    may fail due to the cartoon-style appearance.

    Returns:
        np.ndarray of shape (512,) and dtype float64, L2-normalised   — or None
    """
    try:
        # Read + resize (Massive speed boost for large images during detection)
        img = cv2.imread(image_path)
        if img is None:
            print(f"[FeatureExtractor] ERROR: Could not read image at {image_path}")
            return None
            
        # ── Optional: Scale down only if image is excessively large to save RAM ──
        if img.shape[0] > 1000 or img.shape[1] > 1000:
            h, w = img.shape[:2]
            scale = 1000.0 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        # ── Primary attempt: OpenCV detector (fast) ──────────────
        try:
            result = DeepFace.represent(
                img_path=img,
                model_name=_MODEL_NAME,
                detector_backend=_DETECTOR,
                enforce_detection=True,
                align=False, # CRITICAL: OpenCV eye-alignment fails on sketches and rotates them wildly
            )
            if result:
                embedding = np.array(result[0]["embedding"], dtype=np.float64)
                # L2-normalise so cosine similarity = dot product
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
                return embedding

        except Exception:
            pass  # Face not detected by primary detector — try fallback

        # ── Fallback: retinaface (Highly accurate on stylized sketches) ──
        try:
            result = DeepFace.represent(
                img_path=img,
                model_name=_MODEL_NAME,
                detector_backend='retinaface',
                enforce_detection=True,
                align=False,
            )
        except Exception:
            # If all else fails, attempt the 'skip' backend just to yield an embedding
            result = DeepFace.represent(
                img_path=img,
                model_name=_MODEL_NAME,
                detector_backend='skip',
                enforce_detection=False,   
                align=False,
            )
        if result:
            embedding = np.array(result[0]["embedding"], dtype=np.float64)
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            return embedding

    except Exception as e:
        print(f"[FeatureExtractor] ERROR: {e}")

    return None
