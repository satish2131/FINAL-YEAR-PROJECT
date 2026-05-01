import cv2
import numpy as np
from deepface import DeepFace
import glob

# Load ArcFace once
MODEL = DeepFace.build_model("ArcFace")
_DETECTOR = "opencv"

images = glob.glob(r"c:\Users\dsaiv\OneDrive\Desktop\final year project\Latent_Facial_Identity_Mapping\**\*.png", recursive=True)

for img_path in images[:3]:
    print(f"\n--- Testing {img_path} ---")
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    if img is None: continue
    
    print(f"Original shape: {img.shape}")
    
    # Simulate current method
    img_bgr = cv2.imread(img_path)
    img_bgr = cv2.resize(img_bgr, (500, 500))
    
    # Try primary detector
    try:
        res = DeepFace.represent(img_bgr, model_name="ArcFace", detector_backend=_DETECTOR, enforce_detection=True)
        print("Primary detector (opencv) SUCCESS")
    except Exception as e:
        print("Primary detector (opencv) FAILED:", e)
        
    # See if converting transparent to white helps
    if len(img.shape) == 3 and img.shape[2] == 4:
        print("Image has alpha channel. Creating white background variant...")
        alpha = img[:, :, 3] / 255.0
        bg = np.ones_like(img[:, :, :3], dtype=np.uint8) * 255
        for c in range(3):
            bg[:, :, c] = (alpha * img[:, :, c] + (1 - alpha) * bg[:, :, c]).astype(np.uint8)
        
        bg = cv2.resize(bg, (500, 500))
        try:
            res_bg = DeepFace.represent(bg, model_name="ArcFace", detector_backend=_DETECTOR, enforce_detection=True)
            print("Primary detector (opencv) ON WHITE BG SUCCESS")
        except Exception as e:
            print("Primary detector (opencv) ON WHITE BG FAILED")
