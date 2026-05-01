"""
image_processing.py — Phase 2: Enhanced preprocessing for InsightFace buffalo_l
=================================================================================
Key changes over Phase 1:
  - Output size increased to 800x800 (InsightFace detector works on 640x640
    internally but benefits from a higher-res input fed to it)
  - CLAHE-style contrast enhancement via OpenCV for better edge definition
  - Histogram equalisation on luminance channel (YUV) — preserves colour
    while boosting contrast in dark/flat composite sketches
  - Gentle Gaussian sharpening to surface facial structure
  - PIL-based fallback if OpenCV fails
"""

import os
import numpy as np

try:
    import cv2
    _CV2_OK = True
except ImportError:
    _CV2_OK = False

from PIL import Image, ImageOps, ImageEnhance, ImageFilter


def _enhance_with_cv2(img_path: str, output_size=(800, 800)) -> bool:
    """
    OpenCV-based preprocessing pipeline (preferred path):
      1. CLAHE on L-channel of LAB colour space  — localised contrast boost
      2. Unsharp-mask sharpening
      3. Pad-resize preserving aspect ratio
    Returns True on success, False on failure.
    """
    # Read with alpha channel
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return False

    # ── Handle Transparent Backgrounds (Composites) ──────────────────────────
    if len(img.shape) == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3] / 255.0
        # Blend onto white background instead of black
        white_bg = np.ones_like(img[:, :, :3]) * 255
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * alpha + white_bg[:, :, c] * (1.0 - alpha)).astype(np.uint8)
        img = img[:, :, :3]
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # ── CLAHE on LAB L-channel (localised contrast, no colour shift) ─────────
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ── Unsharp-mask: sharpen edges (facial contours) ─────────────────────────
    gaussian = cv2.GaussianBlur(img, (0, 0), sigmaX=2.0)
    img = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

    # ── Pad-resize: keep aspect ratio, pad with neutral grey ─────────────────
    h, w = img.shape[:2]
    scale = min(output_size[0] / w, output_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    pad_top    = (output_size[1] - new_h) // 2
    pad_bottom = output_size[1] - new_h - pad_top
    pad_left   = (output_size[0] - new_w) // 2
    pad_right  = output_size[0] - new_w - pad_left
    img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                             cv2.BORDER_CONSTANT, value=(200, 200, 200))

    # Use correct imwrite params based on file extension
    # IMWRITE_JPEG_QUALITY (key=1) is invalid for PNG and causes a warning
    ext = os.path.splitext(img_path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        # PNG: compression 1 = fast, low compression (preserves quality)
        cv2.imwrite(img_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return True


def _enhance_with_pil(img_path: str, output_size=(800, 800)):
    """
    PIL fallback preprocessing pipeline (used when OpenCV is unavailable):
      Contrast + sharpness boost + unsharp mask + pad-resize
    """
    img_raw = Image.open(img_path)
    if img_raw.mode in ('RGBA', 'LA') or (img_raw.mode == 'P' and 'transparency' in img_raw.info):
        alpha = img_raw.convert('RGBA').split()[-1]
        img = Image.new("RGB", img_raw.size, (255, 255, 255))
        img.paste(img_raw, mask=alpha)
    else:
        img = img_raw.convert("RGB")
    
    img = ImageOps.exif_transpose(img)

    img = ImageEnhance.Contrast(img).enhance(1.7)
    img = ImageEnhance.Brightness(img).enhance(1.05)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = img.filter(ImageFilter.UnsharpMask(radius=1.5, percent=140, threshold=3))

    img.thumbnail(output_size, Image.LANCZOS)
    background = Image.new("RGB", output_size, (200, 200, 200))
    offset = (
        (output_size[0] - img.size[0]) // 2,
        (output_size[1] - img.size[1]) // 2,
    )
    background.paste(img, offset)
    background.save(img_path, quality=95)


def preprocess_image(image_path: str, output_size=(800, 800)) -> str:
    """
    Enhanced preprocessing pipeline for composite/latent sketch images.

    Preferred path: OpenCV CLAHE + unsharp mask + pad-resize (800x800)
    Fallback path:  PIL contrast/sharpness boost + pad-resize

    Both paths produce an image that InsightFace buffalo_l and DeepFace ArcFace
    can process more reliably than the raw composite SVG render.

    Returns:
        str: the (modified) image_path
    """
    if _CV2_OK:
        success = _enhance_with_cv2(image_path, output_size)
        if success:
            return image_path
    # PIL fallback
    _enhance_with_pil(image_path, output_size)
    return image_path
